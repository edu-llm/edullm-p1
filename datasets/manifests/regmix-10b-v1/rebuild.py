#!/usr/bin/env python3
"""Rebuild regmix-10b-v1 byte-for-byte from allenai/olmo-mix-1124 and check it.

Inputs (next to this file): sources.json, outputs.json, documents/<domain>.tsv.gz.
Outputs under --out:
  tokenized/<d>/<d>.npy                    headerless uint32 LE token stream per domain
  publish-stage/tokens/<d>/*.u32le.bin     the published pretrain/regmix-10b v1 objects
  verify.json                              per-object sha256 comparison
  work/                                    intermediates (parts, per-task status)

Single machine:
  python rebuild.py --out OUT --cache CACHE [--only NAME[,NAME...]] [--workers N]

Distributed (e.g. Slurm arrays; run the stages in this order, each over
--task 0..count-1, where `--stage plan` prints the counts):
  --stage fetch      download + sha256-check one source file (task 0 also fetches the tokenizer)
  --stage tokenize   tokenize one slice of one source file into work/parts/
  --stage assemble   write one publish object from the parts
  --stage npy        concatenate one domain's publish objects into tokenized/<d>/<d>.npy
  --stage verify     compare every sha256 with outputs.json (exit 1 on any mismatch)

--only accepts domain names (e.g. wiki), tokenized names (tokenized/wiki/wiki.npy)
or publish names (tokens/wiki/val-00000.u32le.bin); the whole domain is rebuilt.
Exit status is non-zero if any output differs; the first mismatching document
(by per-document token count / crc32 from the manifest) is printed.
"""

from __future__ import annotations

import argparse
import errno
import functools
import gzip
import hashlib
import io
import json
import math
import os
import sys
import time
import zlib
from pathlib import Path

# Network filesystems (this was written against FarmShare's NFS-backed /scratch)
# occasionally surface a transient ESTALE/ENOENT on a write, fsync, or rename under
# heavy concurrent load from unrelated processes, even though nothing is wrong with
# the operation itself. Observed in practice: outages of up to ~2 minutes affecting
# many concurrent workers at once (a shared-mount hiccup, not a per-file issue), so
# the backoff budget below is sized to survive that, capped so it doesn't grow
# unboundedly. Retrying costs nothing when the filesystem is healthy.
_RETRYABLE_ERRNOS = (errno.ESTALE, errno.ENOENT, errno.EBUSY)


def _retry_io(fn, attempts: int = 12, base_delay: float = 2.0, max_delay: float = 45.0):
    last: OSError | None = None
    for i in range(attempts):
        try:
            return fn()
        except OSError as exc:
            if exc.errno not in _RETRYABLE_ERRNOS or i == attempts - 1:
                raise
            last = exc
            wait = min(max_delay, base_delay * (2**i))
            print(f"[retry] {fn}: {type(exc).__name__} errno={exc.errno}; retry {i+1}/{attempts} in {wait:.0f}s", flush=True)
            time.sleep(wait)
    raise last  # pragma: no cover - loop always returns or raises above


def _reopen_retry(path: Path, flags: int, attempts: int = 10, base_delay: float = 2.0, max_delay: float = 30.0) -> int:
    """os.open with retry: the outage that makes a handle go stale can itself make
    the reopen transiently fail too (observed: ENOENT on a file that is still there
    seconds later -- a client-side view lagging a server-side hiccup), so the reopen
    needs its own retry, not just the write it's trying to recover."""
    last: OSError | None = None
    for i in range(attempts):
        try:
            return os.open(path, flags)
        except OSError as exc:
            if exc.errno not in _RETRYABLE_ERRNOS or i == attempts - 1:
                raise
            last = exc
            wait = min(max_delay, base_delay * (2**i))
            print(f"[retry] reopen {path}: {type(exc).__name__} errno={exc.errno}; retry {i+1}/{attempts} in {wait:.0f}s", flush=True)
            time.sleep(wait)
    raise last  # pragma: no cover - loop always returns or raises above

import numpy as np

HERE = Path(__file__).resolve().parent
EOS = 100257
TOKENS_PER_TASK = 200_000_000  # tokenize-stage slice size (tokens incl. EOS)
BUF = 64 << 20
BATCH_DOCS = 512
BATCH_CHARS = 16 << 20


def _local_dir() -> Path:
    """A node-local scratch dir for this process: SLURM_TMPDIR/TMPDIR/tmp, never the
    shared NFS mount. Every write stage does its actual I/O here -- ftruncate,
    pwrite, fsync, the works -- with no other process sharing the path, then copies
    the single finished file to its NFS destination in one shot. That trades many
    small concurrent writes against a flaky shared mount (what caused the ESTALE
    failures under heavy parallel load) for one big write per output file, which is
    both faster on local disk and far less exposed to that class of failure."""
    base = os.environ.get("SLURM_TMPDIR") or os.environ.get("TMPDIR") or "/tmp"
    d = Path(base) / f"rebuild-{os.getpid()}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _publish_from_local(local_path: Path, dest: Path) -> None:
    """Copy a fully-written local file to `dest` on the (possibly NFS) output tree:
    write to a `.tmp<pid>` sibling, fsync, atomic rename -- the whole copy retried
    as one unit on a transient ESTALE/ENOENT/EBUSY, which is simpler and safer than
    resuming a partial copy. Removes `local_path` once published."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + f".tmp{os.getpid()}")

    def _copy() -> None:
        with open(local_path, "rb") as src, open(tmp, "wb", buffering=BUF) as out:
            while True:
                b = src.read(BUF)
                if not b:
                    break
                out.write(b)
            out.flush()
            os.fsync(out.fileno())

    _retry_io(_copy)
    _retry_io(lambda: tmp.replace(dest))
    local_path.unlink(missing_ok=True)


# --------------------------------------------------------------------------- manifest


@functools.lru_cache(maxsize=None)
def sources() -> dict:
    return json.loads((HERE / "sources.json").read_text(encoding="utf-8"))


@functools.lru_cache(maxsize=None)
def outputs() -> dict:
    return json.loads((HERE / "outputs.json").read_text(encoding="utf-8"))


def all_domains() -> list[str]:
    return list(outputs()["domains"])


@functools.lru_cache(maxsize=None)
def load_docs(domain: str):
    """(src, row, ntok, crc32) arrays in output order, plus token offsets."""
    path = HERE / outputs()["domains"][domain]["documents_file"]
    with gzip.open(path, "rt", encoding="ascii") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        if header != ["src", "row", "ntok", "crc32"]:
            raise SystemExit(f"{path}: unexpected header {header}")
        cells = fh.read().split()
    if len(cells) % 4:
        raise SystemExit(f"{path}: ragged rows")
    src = np.array(cells[0::4], dtype=np.int64)
    row = np.array(cells[1::4], dtype=np.int64)
    ntok = np.array(cells[2::4], dtype=np.int64)
    crc = np.array([int(x, 16) for x in cells[3::4]], dtype=np.uint32)
    tok_off = np.concatenate([[0], np.cumsum(ntok + 1)])
    meta = outputs()["domains"][domain]
    if len(ntok) != meta["docs"] or int(tok_off[-1]) != meta["tokens_with_eos"]:
        raise SystemExit(f"{path}: doc/token totals disagree with outputs.json")
    return src, row, ntok, crc, tok_off


@functools.lru_cache(maxsize=None)
def source_pieces(domain: str) -> dict:
    """For each source used by `domain`: its docs (output indices) and slice bounds."""
    src, _, ntok, _, _ = load_docs(domain)
    out = {}
    for s in np.unique(src).tolist():
        idx = np.flatnonzero(src == s)
        w = ntok[idx] + 1
        cw = np.concatenate([[0], np.cumsum(w)])
        n_pieces = max(1, math.ceil(int(cw[-1]) / TOKENS_PER_TASK))
        # contiguous slices of the source's output-ordered docs, balanced by tokens
        bounds = [0]
        for p in range(1, n_pieces):
            bounds.append(int(np.searchsorted(cw, cw[-1] * p / n_pieces, side="left")))
        bounds.append(len(idx))
        bounds = sorted(set(bounds))
        out[s] = {"idx": idx, "cw": cw, "bounds": bounds}
    return out


def domains_for(only: str | None) -> list[str]:
    doms = all_domains()
    if not only:
        return doms
    names = {o["name"]: o["domain"] for o in outputs()["outputs"]}
    chosen = set()
    for n in only.split(","):
        n = n.strip()
        if n in doms:
            chosen.add(n)
        elif n in names:
            chosen.add(names[n])
        else:
            raise SystemExit(f"--only: unknown name {n!r}")
    return [d for d in doms if d in chosen]


def plan(only: str | None) -> dict:
    doms = domains_for(only)
    fetch = sorted({int(s) for d in doms for s in np.unique(load_docs(d)[0]).tolist()})
    tokenize = []
    for d in doms:
        for s, sp in sorted(source_pieces(d).items()):
            for p in range(len(sp["bounds"]) - 1):
                tokenize.append((d, s, p))
    assemble = [o["name"] for o in outputs()["outputs"] if o["layout"] == "publish" and o["domain"] in doms]
    return {"domains": doms, "fetch": fetch, "tokenize": tokenize, "assemble": assemble, "npy": doms}


def safe(name: str) -> str:
    return name.replace("/", "__")


def write_json_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1) + "\n", encoding="utf-8")
    _retry_io(lambda: tmp.replace(path))


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(BUF)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------- fetch


def _download(repo_id, repo_type, revision, filename, cache):
    from huggingface_hub import hf_hub_download

    last = None
    for attempt in range(8):
        try:
            return Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    repo_type=repo_type,
                    revision=revision,
                    cache_dir=str(cache),
                )
            )
        except Exception as exc:  # HTTP 429 and other transient errors: back off
            last = exc
            wait = min(600, 15 * 2**attempt)
            print(f"[fetch] {filename}: {type(exc).__name__}: {exc}; retry in {wait}s", flush=True)
            time.sleep(wait)
    raise SystemExit(f"[fetch] giving up on {filename}: {last}")


def ensure_source(i: int, cache: Path, out: Path) -> Path:
    f = sources()["files"][i]
    path = _download(f["repo_id"], f["repo_type"], f["revision"], f["path"], cache)
    status = out / "work" / "status" / "fetch" / f"src{i:02d}.json"
    size = path.stat().st_size
    if status.is_file():
        st = json.loads(status.read_text())
        if st.get("ok") and st.get("local") == str(path) and st.get("size") == size:
            return path
    sha = sha256_path(path)
    ok = sha == f["sha256"] and size == f["size"]
    write_json_atomic(status, {"src": i, "path": f["path"], "local": str(path), "size": size, "sha256": sha, "ok": ok})
    if not ok:
        raise SystemExit(f"[fetch] sha256/size mismatch for {f['path']}: got {sha} {size}, want {f['sha256']} {f['size']}")
    print(f"[fetch] src{i:02d} ok {f['path']} ({size:,} bytes)", flush=True)
    return path


def ensure_tokenizer(cache: Path) -> None:
    t = sources()["tokenizer"]
    for f in t["files"]:
        p = _download(t["repo_id"], t["repo_type"], t["revision"], f["path"], cache)
        sha = sha256_path(p)
        if sha != f["sha256"]:
            raise SystemExit(f"[fetch] tokenizer file {f['path']} sha256 {sha} != {f['sha256']}")
    load_tokenizer(cache)
    print("[fetch] tokenizer ok", flush=True)


@functools.lru_cache(maxsize=None)
def load_tokenizer(cache: Path):
    from transformers import AutoTokenizer

    t = sources()["tokenizer"]
    return AutoTokenizer.from_pretrained(t["repo_id"], revision=t["revision"], cache_dir=str(cache), use_fast=True)


def stage_fetch(task: int, args) -> None:
    p = plan(args.only)
    if task == 0:
        ensure_tokenizer(args.cache)
    ensure_source(p["fetch"][task], args.cache, args.out)


# --------------------------------------------------------------------------- reading
# Same reading rules as the build (trim_and_tokenize_regmix.py): Python text
# iteration over the decompressed file (universal newlines), line.strip(),
# json.loads, text = first non-empty of text/content/code/body. `row` in the
# manifest is the 0-based index of the line in that iteration.


def open_text_stream(path: Path, name: str):
    if name.endswith(".jsonl.zstd") or name.endswith(".jsonl.zst"):
        import zstandard as zstd

        raw = open(path, "rb")
        reader = zstd.ZstdDecompressor().stream_reader(raw)
        return io.TextIOWrapper(reader, encoding="utf-8"), raw
    if name.endswith(".json.gz") or name.endswith(".jsonl.gz"):
        return gzip.open(path, "rt", encoding="utf-8"), None
    if name.endswith(".jsonl") or name.endswith(".json"):
        return open(path, "rt", encoding="utf-8"), None
    raise ValueError(f"unsupported format: {path}")


def doc_text(obj: dict) -> str:
    for key in ("text", "content", "code", "body"):
        val = obj.get(key)
        if isinstance(val, str) and val:
            return val
    return "\n".join(v for v in obj.values() if isinstance(v, str))


# --------------------------------------------------------------------------- tokenize


def part_path(out: Path, domain: str, s: int, p: int) -> Path:
    return out / "work" / "parts" / domain / f"src{s:02d}.p{p:02d}.u32"


def tok_status_path(out: Path, domain: str, s: int, p: int) -> Path:
    return out / "work" / "status" / "tokenize" / f"{domain}.src{s:02d}.p{p:02d}.json"


def stage_tokenize(task: int, args) -> dict:
    domain, s, p = plan(args.only)["tokenize"][task]
    t0 = time.time()
    src_arr, row_arr, ntok_arr, crc_arr, _ = load_docs(domain)
    sp = source_pieces(domain)[s]
    a, b = sp["bounds"][p], sp["bounds"][p + 1]
    idx = sp["idx"][a:b]  # output indices, increasing
    w = ntok_arr[idx] + 1
    local_off = np.concatenate([[0], np.cumsum(w)])  # token offsets inside this part
    rows = row_arr[idx]
    order = np.argsort(rows, kind="stable")
    rows_sorted = rows[order]
    if len(rows_sorted) > 1 and not (np.diff(rows_sorted) > 0).all():
        raise SystemExit(f"{domain} src{s}: duplicate rows in manifest")

    local = ensure_source(s, args.cache, args.out)
    tok = load_tokenizer(args.cache)
    name = Path(sources()["files"][s]["path"]).name

    ppath = part_path(args.out, domain, s, p)
    local_dir = _local_dir()
    tmp = local_dir / f"{domain}.src{s:02d}.p{p:02d}.u32"  # node-local; never on the shared NFS mount
    fd = os.open(tmp, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    os.ftruncate(fd, int(local_off[-1]) * 4)
    mismatches: list[dict] = []
    n_bad = 0
    n_done = 0

    def pwrite_retry(data: bytes, offset: int, attempts: int = 12, base_delay: float = 2.0, max_delay: float = 45.0) -> None:
        """Write at offset, reopening `tmp` on ESTALE.

        Retrying the identical os.pwrite call on the same fd cannot recover from a
        stale NFS file handle -- once a handle goes stale it stays stale, however
        long you wait. The file itself is fine; only closing and reopening the path
        gets a fresh handle. Never reopens with O_TRUNC (that would discard sibling
        offsets already written by earlier flush() calls in this same piece).
        """
        nonlocal fd
        for i in range(attempts):
            try:
                os.pwrite(fd, data, offset)
                return
            except OSError as exc:
                if exc.errno not in _RETRYABLE_ERRNOS or i == attempts - 1:
                    raise
                wait = min(max_delay, base_delay * (2**i))
                print(f"[retry] pwrite {tmp} @ {offset}: {type(exc).__name__} errno={exc.errno}; reopen + retry {i+1}/{attempts} in {wait:.0f}s", flush=True)
                time.sleep(wait)
                try:
                    os.close(fd)
                except OSError:
                    pass
                fd = _reopen_retry(tmp, os.O_RDWR)

    def flush(batch: list[tuple[int, str]]) -> None:
        nonlocal n_bad, n_done
        if not batch:
            return
        enc = tok([t for _, t in batch], add_special_tokens=False, padding=False, truncation=False)
        for (q, text), ids in zip(batch, enc["input_ids"]):
            k = int(idx[q])
            n = len(ids)
            arr = np.empty(n + 1, dtype="<u4")
            arr[:n] = ids
            arr[n] = EOS
            want_n = int(ntok_arr[k])
            got_crc = zlib.crc32(arr.tobytes())
            if n != want_n or got_crc != int(crc_arr[k]):
                n_bad += 1
                if len(mismatches) < 20:
                    mismatches.append(
                        {
                            "domain": domain,
                            "doc": k,
                            "src": s,
                            "row": int(row_arr[k]),
                            "ntok_expected": want_n,
                            "ntok_got": n,
                            "crc32_expected": f"{int(crc_arr[k]):08x}",
                            "crc32_got": f"{got_crc:08x}",
                            "text_head": text[:300],
                            "ids_head": [int(x) for x in ids[:40]],
                        }
                    )
                # keep the slot size so later documents stay aligned
                slot = np.zeros(want_n + 1, dtype="<u4")
                m = min(n + 1, want_n + 1)
                slot[:m] = arr[:m]
                slot[-1] = EOS
                arr = slot
            pwrite_retry(arr.tobytes(), int(local_off[q]) * 4)
            n_done += 1

    stream, raw = open_text_stream(local, name)
    batch: list[tuple[int, str]] = []
    chars = 0
    j = 0
    total = len(rows_sorted)
    try:
        if total:
            want = int(rows_sorted[0])
            for i, line in enumerate(stream):
                if i != want:
                    continue
                obj = json.loads(line.strip())
                text = doc_text(obj) if isinstance(obj, dict) else ""
                if not text:
                    raise SystemExit(f"{domain} src{s} row {i}: manifest row is not a document")
                batch.append((int(order[j]), text))
                chars += len(text)
                j += 1
                if len(batch) >= BATCH_DOCS or chars >= BATCH_CHARS:
                    flush(batch)
                    batch, chars = [], 0
                if j == total:
                    break
                want = int(rows_sorted[j])
        flush(batch)
    finally:
        stream.close()
        if raw is not None:
            raw.close()
    if j != total:
        raise SystemExit(f"{domain} src{s}: file ended after {j} of {total} manifest rows")
    os.fsync(fd)  # local disk: no NFS staleness to retry around here
    os.close(fd)
    _publish_from_local(tmp, ppath)
    st = {
        "domain": domain,
        "src": s,
        "piece": p,
        "docs": int(len(idx)),
        "docs_written": n_done,
        "tokens_with_eos": int(local_off[-1]),
        "mismatched_docs": n_bad,
        "mismatches": mismatches,
        "seconds": round(time.time() - t0, 1),
    }
    write_json_atomic(tok_status_path(args.out, domain, s, p), st)
    print(f"[tokenize] {domain} src{s:02d} piece {p}: {n_done:,} docs, {n_bad} mismatched, {st['seconds']}s", flush=True)
    return st


# --------------------------------------------------------------------------- assemble


def publish_path(out: Path, name: str) -> Path:
    return out / "publish-stage" / name


def sha_record_path(out: Path, name: str) -> Path:
    return out / "work" / "sha" / f"{safe(name)}.json"


def stage_assemble(task: int, args) -> dict:
    name = plan(args.only)["assemble"][task]
    o = {x["name"]: x for x in outputs()["outputs"]}[name]
    domain = o["domain"]
    t_start, t_end = o["domain_token_range"]
    src_arr, _, _, _, tok_off = load_docs(domain)
    k0 = int(np.searchsorted(tok_off, t_start, side="right") - 1)
    k1 = int(np.searchsorted(tok_off, t_end, side="left"))  # exclusive
    docs = np.arange(k0, k1)
    pieces = source_pieces(domain)
    bufs: dict[int, tuple[memoryview, int]] = {}
    for s in np.unique(src_arr[docs]).tolist():
        sp = pieces[s]
        idx, cw, bounds = sp["idx"], sp["cw"], sp["bounds"]
        j0 = int(np.searchsorted(idx, k0, side="left"))
        j1 = int(np.searchsorted(idx, k1, side="left"))
        chunks = []
        for p in range(len(bounds) - 1):
            pa, pb = bounds[p], bounds[p + 1]
            lo, hi = max(j0, pa), min(j1, pb)
            if lo >= hi:
                continue
            st = tok_status_path(args.out, domain, s, p)
            if not st.is_file():
                raise SystemExit(f"[assemble] {name}: tokenize output missing for {domain} src{s} piece {p}")
            with open(part_path(args.out, domain, s, p), "rb") as fh:
                fh.seek(int(cw[lo] - cw[pa]) * 4)
                chunks.append(fh.read(int(cw[hi] - cw[lo]) * 4))
        bufs[s] = [memoryview(b"".join(chunks)), 0]
    dest = publish_path(args.out, name)
    local_tmp = _local_dir() / safe(name)
    h = hashlib.sha256()
    nbytes = 0
    with open(local_tmp, "wb", buffering=BUF) as fh:
        for k in docs.tolist():
            s = int(src_arr[k])
            mv, cur = bufs[s]
            n = int(tok_off[k + 1] - tok_off[k]) * 4
            piece = mv[cur : cur + n]
            bufs[s][1] = cur + n
            lo = max(0, (t_start - int(tok_off[k])) * 4)
            hi = min(n, (t_end - int(tok_off[k])) * 4)
            seg = piece[lo:hi]
            h.update(seg)
            fh.write(seg)
            nbytes += len(seg)
        fh.flush()
        os.fsync(fh.fileno())  # local disk
    _publish_from_local(local_tmp, dest)
    rec = {"name": name, "bytes": nbytes, "sha256": h.hexdigest(), "docs_touched": int(len(docs))}
    write_json_atomic(sha_record_path(args.out, name), rec)
    print(f"[assemble] {name}: {nbytes:,} bytes sha256 {rec['sha256'][:12]}", flush=True)
    return rec


def stage_npy(task: int, args) -> dict:
    domain = plan(args.only)["npy"][task]
    objs = [o for o in outputs()["outputs"] if o["domain"] == domain and o["layout"] == "publish"]
    objs.sort(key=lambda o: o["domain_token_range"][0])
    name = f"tokenized/{domain}/{domain}.npy"
    dest = args.out / name
    local_tmp = _local_dir() / safe(name)
    h_out = hashlib.sha256()
    with open(local_tmp, "wb") as out_fh:
        for o in objs:
            h_in = hashlib.sha256()
            with open(publish_path(args.out, o["name"]), "rb") as in_fh:
                while True:
                    b = in_fh.read(BUF)
                    if not b:
                        break
                    h_in.update(b)
                    h_out.update(b)
                    out_fh.write(b)
            rec = json.loads(sha_record_path(args.out, o["name"]).read_text())
            if rec["sha256"] != h_in.hexdigest():
                raise SystemExit(f"[npy] {o['name']} changed on disk after assemble")
            rec["sha256_from_disk"] = h_in.hexdigest()
            write_json_atomic(sha_record_path(args.out, o["name"]), rec)
        out_fh.flush()
        os.fsync(out_fh.fileno())  # local disk
    disk = sha256_path(local_tmp)
    if disk != h_out.hexdigest():
        raise SystemExit(f"[npy] {name}: sha256 of written bytes != sha256 re-read from local disk")
    _publish_from_local(local_tmp, dest)
    disk = sha256_path(dest)
    if disk != h_out.hexdigest():
        raise SystemExit(f"[npy] {name}: sha256 of written bytes != sha256 re-read from disk")
    rec = {"name": name, "bytes": dest.stat().st_size, "sha256": disk, "sha256_from_disk": disk}
    write_json_atomic(sha_record_path(args.out, name), rec)
    print(f"[npy] {name}: {rec['bytes']:,} bytes sha256 {disk[:12]}", flush=True)
    return rec


# --------------------------------------------------------------------------- verify


def first_bad_doc_from_npy(out: Path, domain: str) -> dict | None:
    """Scan a rebuilt .npy against per-document crc32 from the manifest."""
    path = out / "tokenized" / domain / f"{domain}.npy"
    if not path.is_file():
        return None
    src, row, ntok, crc, tok_off = load_docs(domain)
    if path.stat().st_size != int(tok_off[-1]) * 4:
        return {"domain": domain, "note": f"size {path.stat().st_size} != expected {int(tok_off[-1]) * 4}"}
    mm = np.memmap(path, dtype="<u4", mode="r")
    for k in range(len(ntok)):
        a, b = int(tok_off[k]), int(tok_off[k + 1])
        if zlib.crc32(mm[a:b]) != int(crc[k]):
            return {
                "domain": domain,
                "doc": k,
                "src": int(src[k]),
                "row": int(row[k]),
                "ntok_expected": int(ntok[k]),
                "crc32_expected": f"{int(crc[k]):08x}",
                "crc32_got": f"{zlib.crc32(mm[a:b]):08x}",
                "ids_head_got": mm[a : min(b, a + 40)].tolist(),
            }
    return None


def stage_verify(args) -> int:
    doms = domains_for(args.only)
    rows = []
    ok_all = True
    for o in outputs()["outputs"]:
        if o["domain"] not in doms:
            continue
        rec_path = sha_record_path(args.out, o["name"])
        got = json.loads(rec_path.read_text()) if rec_path.is_file() else None
        match = bool(got) and got["sha256"] == o["sha256"] and got["bytes"] == o["bytes"]
        if got and "sha256_from_disk" in got:
            match = match and got["sha256_from_disk"] == o["sha256"]
        ok_all &= match
        rows.append(
            {
                "name": o["name"],
                "bytes": o["bytes"],
                "expected_sha256": o["sha256"],
                "rebuilt_sha256": got["sha256"] if got else None,
                "match": match,
            }
        )
    tok_status = {}
    for p in (args.out / "work" / "status" / "tokenize").glob("*.json"):
        st = json.loads(p.read_text())
        tok_status[p.name] = st
    doc_mismatch = [m for st in tok_status.values() for m in st["mismatches"]]
    doc_mismatch.sort(key=lambda m: (m["domain"], m["doc"]))
    n_bad_docs = sum(st["mismatched_docs"] for st in tok_status.values())
    first_bad = {}
    for d in doms:
        cand = [m for m in doc_mismatch if m["domain"] == d]
        if cand:
            first_bad[d] = cand[0]
        elif not all(r["match"] for r in rows if r["name"].startswith((f"tokens/{d}/", f"tokenized/{d}/"))):
            first_bad[d] = first_bad_doc_from_npy(args.out, d)
    print(f"{'output':<44} {'expected':<12} {'rebuilt':<12} match")
    for r in rows:
        print(f"{r['name']:<44} {r['expected_sha256'][:12]:<12} {(r['rebuilt_sha256'] or '-')[:12]:<12} {'yes' if r['match'] else 'NO'}")
    n_ok = sum(r["match"] for r in rows)
    print(f"{n_ok}/{len(rows)} outputs sha256-identical; {n_bad_docs} documents differ from the manifest")
    for d, m in first_bad.items():
        print(f"FIRST MISMATCHING DOCUMENT in {d}: {json.dumps(m, ensure_ascii=False)[:2000]}")
    write_json_atomic(
        args.out / "verify.json",
        {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "domains": doms,
            "outputs": rows,
            "all_match": ok_all,
            "mismatched_documents": n_bad_docs,
            "first_mismatching_document": first_bad,
            "tokenize_seconds": sum(st["seconds"] for st in tok_status.values()),
        },
    )
    return 0 if ok_all else 1


# --------------------------------------------------------------------------- driver


def _pool_run(stage: str, task: int, args) -> None:
    {"tokenize": stage_tokenize, "assemble": stage_assemble, "npy": stage_npy}[stage](task, args)


def run_all(args) -> int:
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
    import multiprocessing as mp

    p = plan(args.only)
    ensure_tokenizer(args.cache)
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(p["fetch"])))) as ex:
        list(ex.map(lambda s: ensure_source(s, args.cache, args.out), p["fetch"]))
    ctx = mp.get_context("spawn")
    for stage in ("tokenize", "assemble", "npy"):
        n = len(p[stage])
        with ProcessPoolExecutor(max_workers=min(args.workers, n), mp_context=ctx) as ex:
            futs = [ex.submit(_pool_run, stage, i, args) for i in range(n)]
            for f in futs:
                f.result()
    rc = stage_verify(args)
    if rc == 0 and not args.keep_work:
        import shutil

        shutil.rmtree(args.out / "work" / "parts", ignore_errors=True)
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True, help="Hugging Face cache dir for sources and tokenizer")
    ap.add_argument("--only", default=None)
    ap.add_argument("--stage", default="all", choices=["all", "plan", "fetch", "tokenize", "assemble", "npy", "verify"])
    ap.add_argument("--task", type=int, default=None, help="task index for a single-stage run (default: $SLURM_ARRAY_TASK_ID)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--keep-work", action="store_true", help="keep work/parts after a successful run")
    args = ap.parse_args()
    args.out = args.out.resolve()
    args.cache = args.cache.resolve()
    if args.stage == "all":
        return run_all(args)
    if args.stage == "plan":
        p = plan(args.only)
        print(json.dumps({k: len(v) for k, v in p.items()}))
        return 0
    if args.stage == "verify":
        return stage_verify(args)
    task = args.task if args.task is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
    if args.stage == "fetch":
        stage_fetch(task, args)
    else:
        _pool_run(args.stage, task, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
