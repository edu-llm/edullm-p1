#!/usr/bin/env python3
"""Rebuild olmo-127b-v1 (edullm-data ``pretrain/olmo-127b`` v1) byte-for-byte.

Inputs are public: 512 files of ``allenai/olmo-mix-1124`` and the
``allenai/dolma2-tokenizer`` tokenizer, both at pinned commits (sources.json).
Every document of every file is kept, in file order (documents_per_file.tsv.gz).
Each document is dolma2-tokenized and followed by EOS (100257); the per-file
token streams are joined per source in join order, cut into 1 GiB
``train-NNNNN.u32le.bin`` objects, and the last 0.15% of each source stream is
``val-00000.u32le.bin`` (outputs.json).

Usage::

    python rebuild.py --out OUT --cache CACHE                 # all 481 objects
    python rebuild.py --out OUT --cache CACHE --only tokens/wiki/val-00000.u32le.bin
    python rebuild.py --out OUT --cache CACHE --only wiki     # every object of one source
    python rebuild.py --out OUT --cache CACHE --only src:511 --hash-only   # one input file

``--only`` accepts object names, sources (``wiki`` or ``tokens/wiki``), and input
files (``src:<index>`` or the HF path); it may be repeated or comma-separated.
Object selections tokenize exactly the input files that overlap the objects.
File selections check that file's token stream against its recorded sha256.
``--hash-only`` verifies without keeping outputs (objects are hashed while
streamed; with ``--jobs > 1`` per-file token files are written to OUT/.work and
deleted after use).

Exit status is non-zero on any mismatch; the first mismatching document window
(and, with ``--reference``, the exact document) is printed.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import gzip
import hashlib
import io
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TOKEN_BYTES = 4
DTYPE = np.dtype("<u4")


# --------------------------------------------------------------------------- manifest


def _read_tsv_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


class Manifest:
    def __init__(self, mdir: Path) -> None:
        self.dir = mdir
        self.sources_json = json.loads((mdir / "sources.json").read_text(encoding="utf-8"))
        self.outputs = json.loads((mdir / "outputs.json").read_text(encoding="utf-8"))
        fmt = self.outputs["format"]
        if fmt["dtype"] != "uint32" or fmt["endianness"] != "little":
            raise SystemExit("outputs.json: unsupported format")
        self.eos = int(fmt["eos_token_id"])
        self.window = int(self.outputs["doc_windows"]["eos_per_window"])
        self.chunk_docs = int(self.sources_json["tokenizer"]["call"]["batch_docs"])

        files = {int(f["src"]): dict(f) for f in self.sources_json["files"]}
        for row in _read_tsv_gz(mdir / "documents_per_file.tsv.gz"):
            f = files[int(row["src"])]
            if f["path"] != row["hf_path"]:
                raise SystemExit(f"documents_per_file: src {row['src']} path mismatch")
            f["row_start"] = int(row["row_start"])
            f["row_end"] = int(row["row_end"])
            f["skipped"] = _parse_skipped(row["skipped_rows"])
            f["ndocs"] = int(row["ndocs"])
            f["ntok"] = int(row["ntok"])
            f["eos_in_text"] = int(row["eos_in_text"])
        for fs in self.outputs["file_streams"]:
            f = files[int(fs["src"])]
            f["stream_start"] = int(fs["stream_start_token"])
            f["tokens"] = int(fs["tokens_with_eos"])
            f["stream_sha256"] = fs["sha256"]
            if f["ntok"] + f["ndocs"] != f["tokens"]:
                raise SystemExit(f"src {f['src']}: ntok+ndocs != tokens_with_eos")
        self.files = [files[i] for i in sorted(files)]
        if [f["src"] for f in self.files] != list(range(len(self.files))):
            raise SystemExit("sources.json: src indices must be 0..N-1")
        self.file_by_path = {f["path"]: f for f in self.files}

        self.windows: dict[int, list[tuple[int, int, str]]] = {}
        for row in _read_tsv_gz(mdir / "doc_windows.tsv.gz"):
            self.windows.setdefault(int(row["src"]), []).append(
                (int(row["start_token"]), int(row["ntokens"]), row["sha256_16"])
            )

        self.source_names = list(self.outputs["layout"]["sources"])
        self.objects = list(self.outputs["objects"])
        self.obj_by_name = {o["name"]: o for o in self.objects}
        self.files_of: dict[str, list[dict]] = {s: [] for s in self.source_names}
        for f in self.files:
            self.files_of[f["source"]].append(f)
        self.objects_of: dict[str, list[dict]] = {s: [] for s in self.source_names}
        for o in self.objects:
            self.objects_of[o["source"]].append(o)
        self._check_geometry()

    def _check_geometry(self) -> None:
        for s in self.source_names:
            pos = 0
            for f in sorted(self.files_of[s], key=lambda f: f["join_key"]):
                if f["stream_start"] != pos:
                    raise SystemExit(f"{s}: file streams do not tile the source stream")
                pos += f["tokens"]
            total = pos
            pos = 0
            for o in self.objects_of[s]:
                if int(o["stream_start_token"]) != pos or int(o["bytes"]) != int(o["tokens"]) * 4:
                    raise SystemExit(f"{s}: objects do not tile the source stream")
                pos += int(o["tokens"])
            if pos != total:
                raise SystemExit(f"{s}: objects cover {pos} tokens, files {total}")

    def files_overlapping(self, obj: dict) -> list[dict]:
        a = int(obj["stream_start_token"])
        b = a + int(obj["tokens"])
        return [
            f
            for f in self.files_of[obj["source"]]
            if f["stream_start"] < b and f["stream_start"] + f["tokens"] > a
        ]


def _parse_skipped(text: str) -> dict[int, str]:
    out: dict[int, str] = {}
    for item in filter(None, (text or "").split(",")):
        row, reason = item.split(":", 1)
        out[int(row)] = reason
    return out


# --------------------------------------------------------------------------- downloads


def sha256_file(path: Path, bufsize: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _with_backoff(fn, what: str, tries: int = 8):
    delay = 10.0
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - retried below, re-raised at the end
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if attempt == tries:
                raise
            wait = delay * (3 if status == 429 else 1)
            print(f"[download] {what}: {type(exc).__name__} status={status}; retry in {wait:.0f}s",
                  flush=True)
            time.sleep(wait)
            delay = min(delay * 2, 300.0)


def fetch_source_file(f: dict, cache: Path) -> Path:
    from huggingface_hub import hf_hub_download

    path = Path(
        _with_backoff(
            lambda: hf_hub_download(
                repo_id=f["repo_id"],
                repo_type=f["repo_type"],
                revision=f["revision"],
                filename=f["path"],
                cache_dir=str(cache / "hf"),
            ),
            f["path"],
        )
    )
    size = path.stat().st_size
    if size != int(f["size"]):
        raise SystemExit(f"{f['path']}: size {size} != {f['size']}")
    got = sha256_file(path)
    if got != f["sha256"]:
        raise SystemExit(f"{f['path']}: sha256 {got} != {f['sha256']}")
    return path


def fetch_tokenizer(man: Manifest, cache: Path) -> Path:
    from huggingface_hub import snapshot_download

    spec = man.sources_json["tokenizer"]
    names = [x["path"] for x in spec["files"]]
    local = Path(
        _with_backoff(
            lambda: snapshot_download(
                repo_id=spec["repo_id"],
                repo_type=spec["repo_type"],
                revision=spec["revision"],
                allow_patterns=names,
                cache_dir=str(cache / "hf"),
            ),
            spec["repo_id"],
        )
    )
    for x in spec["files"]:
        p = local / x["path"]
        got = sha256_file(p)
        if got != x["sha256"] or p.stat().st_size != int(x["size"]):
            raise SystemExit(f"tokenizer file {x['path']}: sha256 {got} != {x['sha256']}")
    return local


def load_tokenizer(local: Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(local), use_fast=True)


# --------------------------------------------------------------------------- reading
# The reading rules below are exactly those of the original tokenizer
# (datasets/olmo/tokenize_olmo_shard.py, sha256 d5e7bd97...): text-mode
# iteration with universal newlines, UTF-8 with errors="replace", zstd read
# with python-zstandard's default stream_reader (first frame only; every file
# here is single-frame, see README), and rows skipped when blank, invalid
# JSON, or with an empty/missing "text".


def open_text_stream(path: Path, name: str):
    name = name.lower()
    if name.endswith(".json.gz") or name.endswith(".jsonl.gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    if name.endswith(".jsonl.zstd") or name.endswith(".jsonl.zst") or name.endswith(".zstd"):
        import zstandard as zstd

        fh = open(path, "rb")
        reader = zstd.ZstdDecompressor().stream_reader(fh)
        return io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
    if name.endswith(".jsonl") or name.endswith(".json"):
        return open(path, "rt", encoding="utf-8", errors="replace")
    raise ValueError(f"unsupported format: {name}")


def iter_rows(path: Path, name: str):
    """Yield (row, text or None, reason) for every line in native order."""
    with open_text_stream(path, name) as fh:
        for row, line in enumerate(fh):
            line = line.strip()
            if not line:
                yield row, None, "blank"
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                yield row, None, "badjson"
                continue
            text = obj.get("text")
            if not text:
                yield row, None, "notext"
                continue
            yield row, text, ""


def audit_counts(path: Path, name: str) -> dict:
    """Newline counts from binary decompression (all frames), for the README audit."""
    out: dict = {}
    n = 0
    last = b""
    if name.endswith(".zstd") or name.endswith(".zst"):
        import zstandard as zstd

        for across in (False, True):
            n, total, last = 0, 0, b""
            with open(path, "rb") as fh:
                r = zstd.ZstdDecompressor().stream_reader(fh, read_across_frames=across)
                while True:
                    b = r.read(1 << 24)
                    if not b:
                        break
                    n += b.count(b"\n")
                    total += len(b)
                    last = b[-1:]
            key = "all_frames" if across else "first_frame"
            out[key] = {"newlines": n, "bytes": total, "ends_with_newline": last == b"\n"}
    else:
        total = 0
        with gzip.open(path, "rb") as fh:
            while True:
                b = fh.read(1 << 24)
                if not b:
                    break
                n += b.count(b"\n")
                total += len(b)
                last = b[-1:]
        out["all_members"] = {"newlines": n, "bytes": total, "ends_with_newline": last == b"\n"}
    return out


# --------------------------------------------------------------------------- published reference


class ReferenceReader:
    """Sequential reader over the published objects of one source (diagnosis only)."""

    def __init__(self, ref_dir: Path, man: Manifest, source: str, start_token: int) -> None:
        self.objs = [(int(o["stream_start_token"]), int(o["tokens"]), ref_dir / o["name"])
                     for o in man.objects_of[source]]
        self.pos = start_token

    def read(self, n: int) -> np.ndarray:
        parts = []
        while n > 0:
            for a, t, p in self.objs:
                if a <= self.pos < a + t:
                    take = min(n, a + t - self.pos)
                    with open(p, "rb") as fh:
                        fh.seek((self.pos - a) * TOKEN_BYTES)
                        parts.append(np.frombuffer(fh.read(take * TOKEN_BYTES), dtype=DTYPE))
                    self.pos += take
                    n -= take
                    break
            else:
                break
        return np.concatenate(parts) if parts else np.zeros(0, dtype=DTYPE)


# --------------------------------------------------------------------------- one file


class WindowHasher:
    """sha256 of consecutive windows ending after every W-th EOS token of a file stream."""

    def __init__(self, eos: int, w: int) -> None:
        self.eos, self.w = eos, w
        self.count = 0  # EOS tokens seen
        self.pos = 0  # tokens seen
        self.start = 0
        self.h = hashlib.sha256()
        self.out: list[tuple[int, int, str]] = []

    def update(self, arr: np.ndarray) -> None:
        idx = np.flatnonzero(arr == self.eos)
        cut = 0
        if len(idx):
            ords = self.count + np.arange(len(idx))
            for p in idx[(ords + 1) % self.w == 0]:
                self.h.update(arr[cut : p + 1])
                end = self.pos + int(p) + 1
                self.out.append((self.start, end - self.start, self.h.hexdigest()[:16]))
                self.start = end
                self.h = hashlib.sha256()
                cut = int(p) + 1
            self.count += len(idx)
        self.h.update(arr[cut:])
        self.pos += len(arr)

    def finish(self) -> list[tuple[int, int, str]]:
        if self.pos > self.start:
            self.out.append((self.start, self.pos - self.start, self.h.hexdigest()[:16]))
            self.start = self.pos
        return self.out


def process_file(
    f: dict,
    man: Manifest,
    cache: Path,
    tok_dir: Path,
    *,
    sink_path: Path | None = None,
    reference: Path | None = None,
    stats_dir: Path | None = None,
    audit: bool = False,
    chunk_docs: int | None = None,
    emit=None,
) -> dict:
    """Tokenize one input file; verify its stream; optionally write/emit tokens."""
    t0 = time.time()
    path = fetch_source_file(f, cache)
    t_dl = time.time() - t0
    tok = load_tokenizer(tok_dir)
    eos = man.eos
    chunk_docs = chunk_docs or man.chunk_docs
    stream = hashlib.sha256()
    windows = WindowHasher(eos, man.window)
    out_fh = open(sink_path, "wb") if sink_path else None
    ref = ReferenceReader(reference, man, f["source"], f["stream_start"]) if reference else None
    first_diff: dict | None = None
    problems: list[str] = []

    obj_bounds = [
        (o["name"], int(o["stream_start_token"]) - f["stream_start"],
         int(o["stream_start_token"]) + int(o["tokens"]) - f["stream_start"])
        for o in man.objects_of[f["source"]]
        if int(o["stream_start_token"]) < f["stream_start"] + f["tokens"]
        and int(o["stream_start_token"]) + int(o["tokens"]) > f["stream_start"]
    ]
    docs_ending = {name: 0 for name, _, _ in obj_bounds}

    n_rows = n_docs = n_tok = eos_in_text = 0
    skipped: dict[int, str] = {}
    pos = 0  # token offset within this file's stream
    batch_texts: list[str] = []
    batch_rows: list[int] = []

    def flush() -> None:
        nonlocal pos, n_docs, n_tok, eos_in_text, first_diff
        if not batch_texts:
            return
        enc = tok(
            batch_texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )
        for row, text, ids in zip(batch_rows, batch_texts, enc["input_ids"]):
            arr = np.empty(len(ids) + 1, dtype=DTYPE)
            arr[:-1] = ids
            arr[-1] = eos
            k = int(np.count_nonzero(arr[:-1] == eos))
            eos_in_text += k
            stream.update(arr)
            windows.update(arr)
            if out_fh is not None:
                out_fh.write(arr.tobytes())
            if emit is not None:
                emit(pos, arr)
            end = pos + len(arr)
            for name, a, b in obj_bounds:
                if a < end <= b:
                    docs_ending[name] += 1
            if ref is not None and first_diff is None:
                pub = ref.read(len(arr))
                if len(pub) != len(arr) or not np.array_equal(pub, arr):
                    m = min(len(pub), len(arr))
                    neq = np.flatnonzero(pub[:m] != arr[:m])
                    j = int(neq[0]) if len(neq) else m
                    first_diff = {
                        "doc": n_docs, "row": row, "token_in_doc": j,
                        "file_token_offset": pos + j,
                        "rebuilt_tokens": arr[max(0, j - 8): j + 8].tolist(),
                        "published_tokens": pub[max(0, j - 8): j + 8].tolist(),
                        "rebuilt_text": tok.decode(arr[max(0, j - 32): j + 32].tolist()),
                        "published_text": tok.decode(pub[max(0, j - 32): j + 32].tolist()),
                        "source_text_head": text[:200],
                    }
            pos = end
            n_docs += 1
            n_tok += len(ids)
        batch_texts.clear()
        batch_rows.clear()

    for row, text, reason in iter_rows(path, f["path"]):
        n_rows += 1
        if text is None:
            skipped[row] = reason
            continue
        batch_texts.append(text)
        batch_rows.append(row)
        if len(batch_texts) >= chunk_docs:
            flush()
    flush()
    if out_fh is not None:
        out_fh.close()

    got = stream.hexdigest()
    win = windows.finish()
    ok = got == f["stream_sha256"] and pos == f["tokens"]
    if f["row_start"] != 0 or n_rows != f["row_end"]:
        problems.append(f"rows: file has {n_rows}, manifest {f['row_start']}..{f['row_end']}")
    if skipped != f["skipped"]:
        problems.append(f"skipped rows differ: file {sorted(skipped.items())[:10]}, "
                        f"manifest {sorted(f['skipped'].items())[:10]}")
    if n_docs != f["ndocs"] or n_tok != f["ntok"] or eos_in_text != f["eos_in_text"]:
        problems.append(f"counts: docs {n_docs}/{f['ndocs']} ntok {n_tok}/{f['ntok']} "
                        f"eos_in_text {eos_in_text}/{f['eos_in_text']}")
    first_bad_window = None
    exp_win = man.windows.get(f["src"], [])
    for k, (w_got, w_exp) in enumerate(zip(win, exp_win)):
        if w_got != w_exp:
            first_bad_window = {"window": k, "start_token": w_exp[0],
                                "eos_ordinals": [k * man.window, (k + 1) * man.window - 1]}
            break
    if first_bad_window is None and len(win) != len(exp_win):
        k = min(len(win), len(exp_win))
        first_bad_window = {"window": k, "eos_ordinals": [k * man.window, (k + 1) * man.window - 1]}

    result = {
        "src": f["src"], "source": f["source"], "path": f["path"],
        "ok": ok and not problems, "stream_ok": ok,
        "sha256": got, "expected_sha256": f["stream_sha256"],
        "tokens_with_eos": pos, "expected_tokens_with_eos": f["tokens"],
        "rows": n_rows, "docs": n_docs, "ntok": n_tok, "eos_in_text": eos_in_text,
        "skipped": {str(k): v for k, v in sorted(skipped.items())},
        "docs_ending_in_object": docs_ending,
        "windows": len(win), "first_bad_window": first_bad_window,
        "first_diff_vs_reference": first_diff,
        "problems": problems,
        "download_s": round(t_dl, 1), "total_s": round(time.time() - t0, 1),
    }
    if audit:
        result["audit"] = audit_counts(path, f["path"])
    if stats_dir is not None:
        stats_dir.mkdir(parents=True, exist_ok=True)
        (stats_dir / f"src-{f['src']:05d}.json").write_text(json.dumps(result, indent=1) + "\n")
    return result


def report_file(r: dict, man: Manifest) -> None:
    tag = "OK      " if r["ok"] else "MISMATCH"
    print(f"{tag} src={r['src']:>3} {r['source']:<15} docs={r['docs']:>9} "
          f"tokens={r['tokens_with_eos']:>11} sha256={r['sha256'][:12]} "
          f"expected={r['expected_sha256'][:12]} ({r['total_s']}s) {r['path']}", flush=True)
    for p in r["problems"]:
        print(f"         problem: {p}", flush=True)
    if not r["stream_ok"]:
        w = r["first_bad_window"]
        if w:
            lo, hi = w["eos_ordinals"]
            print(f"         first mismatching document window: #{w['window']} "
                  f"(documents {lo}..{hi} of this file if no EOS occurs inside text)", flush=True)
        d = r.get("first_diff_vs_reference")
        if d:
            print(f"         first mismatching document: doc {d['doc']} (row {d['row']}), "
                  f"token {d['token_in_doc']} of the document", flush=True)
            print(f"           rebuilt  : {d['rebuilt_tokens']} {d['rebuilt_text']!r}", flush=True)
            print(f"           published: {d['published_tokens']} {d['published_text']!r}",
                  flush=True)


# --------------------------------------------------------------------------- objects


class ObjectRouter:
    """Receive a source's token stream in order; write/hash the selected objects."""

    def __init__(self, man: Manifest, source: str, names: set[str], out: Path, hash_only: bool):
        self.objs = [o for o in man.objects_of[source] if o["name"] in names]
        self.out, self.hash_only = out, hash_only
        self.state: dict[str, dict] = {}
        self.results: list[dict] = []

    def feed(self, src_pos: int, arr: np.ndarray) -> None:
        end = src_pos + len(arr)
        for o in self.objs:
            a = int(o["stream_start_token"])
            b = a + int(o["tokens"])
            if end <= a or src_pos >= b:
                continue
            st = self.state.get(o["name"])
            if st is None:
                if max(src_pos, a) != a:
                    raise RuntimeError(f"{o['name']}: stream did not start at object start")
                fh = None
                if not self.hash_only:
                    p = self.out / o["name"]
                    p.parent.mkdir(parents=True, exist_ok=True)
                    fh = open(p.with_suffix(p.suffix + ".partial"), "wb")
                st = self.state[o["name"]] = {"h": hashlib.sha256(), "fh": fh, "pos": a}
            lo, hi = max(src_pos, a), min(end, b)
            if lo != st["pos"]:
                raise RuntimeError(f"{o['name']}: out-of-order bytes")
            piece = arr[lo - src_pos : hi - src_pos]
            st["h"].update(piece)
            if st["fh"] is not None:
                st["fh"].write(piece.tobytes())
            st["pos"] = hi
            if hi == b:
                self._finish(o, st)

    def _finish(self, o: dict, st: dict) -> None:
        got = st["h"].hexdigest()
        if st["fh"] is not None:
            st["fh"].close()
            p = self.out / o["name"]
            p.with_suffix(p.suffix + ".partial").replace(p)
        ok = got == o["sha256"]
        r = {"name": o["name"], "ok": ok, "sha256": got, "expected_sha256": o["sha256"],
             "bytes": int(o["bytes"])}
        self.results.append(r)
        print(f"{'OK      ' if ok else 'MISMATCH'} object {o['name']} sha256={got[:12]} "
              f"expected={o['sha256'][:12]}", flush=True)
        st["done"] = True

    def incomplete(self) -> list[str]:
        done = {r["name"] for r in self.results}
        return [o["name"] for o in self.objs if o["name"] not in done]


# --------------------------------------------------------------------------- driver


def resolve_only(man: Manifest, items: list[str]) -> tuple[set[str], set[int]]:
    objs: set[str] = set()
    files: set[int] = set()
    names = [n.strip() for item in items for n in item.split(",") if n.strip()]
    if not names:
        objs = {o["name"] for o in man.objects}
    for n in names:
        n = n.strip("/")
        if n.startswith("src:"):
            files.add(int(n[4:]))
        elif n in man.obj_by_name:
            objs.add(n)
        elif (n[len("tokens/"):] if n.startswith("tokens/") else n) in man.objects_of:
            s = n[len("tokens/"):] if n.startswith("tokens/") else n
            objs.update(o["name"] for o in man.objects_of[s])
        elif n in man.file_by_path:
            files.add(man.file_by_path[n]["src"])
        else:
            raise SystemExit(f"--only: unknown name {n!r}")
    for name in objs:
        files.update(f["src"] for f in man.files_overlapping(man.obj_by_name[name]))
    return objs, files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="output dir (v1 layout: tokens/<source>/...)")
    ap.add_argument("--cache", type=Path, required=True, help="download cache dir")
    ap.add_argument("--only", action="append", default=[], help="objects, sources, or src:<i> files")
    ap.add_argument("--manifest-dir", type=Path, default=HERE)
    ap.add_argument("--hash-only", action="store_true", help="verify without keeping outputs")
    ap.add_argument("--download-only", action="store_true")
    ap.add_argument("--jobs", type=int, default=1, help="input files tokenized in parallel")
    ap.add_argument("--chunk-docs", type=int, default=None, help="documents per tokenizer call")
    ap.add_argument("--reference", type=Path, default=None,
                    help="published v1 dir (tokens/<source>/...) for document-level diagnosis")
    ap.add_argument("--stats-dir", type=Path, default=None, help="write per-file JSON results")
    ap.add_argument("--audit", action="store_true", help="also count raw newlines per file")
    ap.add_argument("--keep-work", action="store_true")
    args = ap.parse_args()

    man = Manifest(args.manifest_dir)
    objs, file_ids = resolve_only(man, args.only)
    files = [man.files[i] for i in sorted(file_ids)]
    print(f"manifest: {len(man.files)} files, {len(man.objects)} objects; selected "
          f"{len(objs)} object(s), {len(files)} input file(s)", flush=True)

    args.cache.mkdir(parents=True, exist_ok=True)
    tok_dir = fetch_tokenizer(man, args.cache)
    if args.download_only:
        for f in files:
            fetch_source_file(f, args.cache)
            print(f"downloaded+verified src={f['src']} {f['path']}", flush=True)
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    work = args.out / ".work"
    kw = dict(reference=args.reference, stats_dir=args.stats_dir, audit=args.audit,
              chunk_docs=args.chunk_docs)
    file_results: list[dict] = []
    obj_results: list[dict] = []
    failed = False

    by_source: dict[str, list[dict]] = {}
    for f in files:
        by_source.setdefault(f["source"], []).append(f)

    for source in man.source_names:
        todo = sorted(by_source.get(source, []), key=lambda f: f["join_key"])
        if not todo:
            continue
        names = {n for n in objs if man.obj_by_name[n]["source"] == source}
        router = ObjectRouter(man, source, names, args.out, args.hash_only) if names else None

        def emit_for(f):
            return (lambda pos, arr: router.feed(f["stream_start"] + pos, arr)) if router else None

        if args.jobs <= 1 or len(todo) == 1:
            for f in todo:
                try:
                    r = process_file(f, man, args.cache, tok_dir, emit=emit_for(f), **kw)
                except Exception:  # noqa: BLE001
                    traceback.print_exc()
                    return 2
                report_file(r, man)
                file_results.append(r)
        else:
            work.mkdir(parents=True, exist_ok=True)
            with cf.ProcessPoolExecutor(max_workers=args.jobs) as ex:
                futs = []
                for f in todo:
                    sink = work / f"src-{f['src']:05d}.u32" if router else None
                    futs.append((f, sink, ex.submit(process_file, f, man, args.cache, tok_dir,
                                                    sink_path=sink, **kw)))
                for f, sink, fut in futs:  # consume in join order
                    r = fut.result()
                    report_file(r, man)
                    file_results.append(r)
                    if router is not None:
                        with open(sink, "rb") as fh:
                            pos = f["stream_start"]
                            while True:
                                b = fh.read(1 << 26)
                                if not b:
                                    break
                                arr = np.frombuffer(b, dtype=DTYPE)
                                router.feed(pos, arr)
                                pos += len(arr)
                        if not args.keep_work:
                            sink.unlink()
        if router is not None:
            obj_results.extend(router.results)
            for name in router.incomplete():
                print(f"MISMATCH object {name}: incomplete stream", flush=True)
                failed = True

    bad_files = [r for r in file_results if not r["ok"]]
    bad_objs = [r for r in obj_results if not r["ok"]]
    failed = failed or bool(bad_files) or bool(bad_objs)
    print(f"summary: files {len(file_results) - len(bad_files)}/{len(file_results)} ok, "
          f"objects {len(obj_results) - len(bad_objs)}/{len(obj_results)} ok", flush=True)
    if bad_files:
        r = bad_files[0]
        print(f"first mismatching input file: src={r['src']} {r['path']}", flush=True)
        report_file(r, man)
    if work.is_dir() and not args.keep_work and not any(work.iterdir()):
        work.rmdir()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
