#!/usr/bin/env python3
"""Derive the regmix-10b-v1 document manifest from the surviving build artifacts.

One-off provenance script that ran on FarmShare (paths below are FarmShare
paths). It is not needed to rebuild the corpus; rebuild.py only reads the
files this script wrote (sources.json, outputs.json, documents/*.tsv.gz).

What survived the build: plan/ (trim_results.json with the per-domain
materialization order and doc counts), logs/, the tokenized memmaps
tokenized/<d>/<d>.npy, and the publish stage. The intermediate text
(data/, trim/) was deleted, so selection is re-derived and then checked:

  hash      one array task per trained-on file: sha256, and for each .npy the
            EOS positions, a crc32 per document, and the sha256 of every byte
            range the publish layout cuts out of it.
  index     one array task per source file: download from HF at the pinned
            revision, check the LFS sha256, then iterate the file exactly as
            trim_and_tokenize_regmix.py did and record which physical lines
            it yielded as documents.
  manifest  single task: replay the trim (materialize the domain's files in
            the logged order, random.Random(42).shuffle over document
            indices, keep the first docs_after), map each kept document to
            (source file, row), attach ntok/crc32 from the .npy, and write
            documents/<d>.tsv.gz, sources.json, outputs.json.

Run with the original build venv's python (read-only use, PYTHONDONTWRITEBYTECODE=1)
so file reading matches what ran (zstandard 0.25.0, Python 3.12.3).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import random
import sys
import time
import zlib
from pathlib import Path

import numpy as np

BUILD = Path("/scratch/users/nzhao2/agent-runs/regmix-10b-20260725-124810")
PUBLISH = Path(
    "/scratch/users/nzhao2/agent-runs/regmix-edullm-publish-20260730T224545Z/publish-stage"
)
WORK = Path("/scratch/users/nzhao2/agent-runs/repro-manifests-20260925/regmix-10b-v1")
MB = WORK / "mb"

REPO_ID = "allenai/olmo-mix-1124"
REPO_TYPE = "dataset"
REVISION = "99ee6aaace88779d1ef099d36251b91101c1679b"
TOKENIZER_REPO = "allenai/dolma2-tokenizer"
TOKENIZER_REVISION = "5292e5d6c0f40b67cc765fe41bec991cf4345b5c"
EOS = 100257
SEED = 42
SHARD_BYTES = 1 << 30
VAL_FRACTION = 0.0015
# Trim order of trim_regmix_domain.sbatch (DOMAIN_LIST); publish uses sorted order.
DOMAINS = ["dclm", "arxiv", "starcoder", "pes2o", "open-web-math", "algebraic-stack", "wiki"]
CHUNK = 256 << 20  # bytes, multiple of 4
BATCH_DOCS = 64  # matches trim_and_tokenize_regmix.py --batch-size 64

# `hash` scans each tokenized .npy for occurrences of the EOS id and treats every
# occurrence as a document boundary. That undercounts nothing but OVERcounts whenever
# a document's own text contains the literal string "<|endoftext|>": the pinned
# dolma2-tokenizer has that string registered as special token id 100257 (== EOS), and
# HF fast tokenizers still recognize a special token's literal text inside the input
# even with add_special_tokens=False (that flag only controls tokens the tokenizer adds
# itself, not ones already spelled out in the raw text). Comparing eos_count in each
# domain's hash record against trim_results.json's docs_after found exactly this, in
# five of seven domains:
#   arxiv +1, starcoder +3, pes2o +1, open-web-math +20, algebraic-stack +39
# (dclm and wiki matched exactly: 0 affected documents). For these five domains we
# retokenize every document directly (`--stage retok`, reduced by `load_retok`) instead
# of trusting the EOS scan, so a document's true length is whatever the tokenizer
# actually produced for its text -- embedded EOS-valued tokens and all. See
# domains_meta[d]["embedded_eos_note"] in outputs.json for the affected documents.
PIECES_PER_FILE = {"arxiv": 4, "starcoder": 1, "pes2o": 5, "open-web-math": 2, "algebraic-stack": 2}
AFFECTED_DOMAINS = list(PIECES_PER_FILE)


# --------------------------------------------------------------------------- helpers


def publish_layout(total_bytes: int) -> list[tuple[str, int, int]]:
    """(name, start_byte, end_byte) of each publish object cut from one domain stream.

    Equivalent to publish_regmix_edullm_data.py: split into 1 GiB train shards,
    then carve int(bytes * 0.0015) (rounded down to 4) off the tail into val.
    """
    val = int(total_bytes * VAL_FRACTION)
    val -= val % 4
    train_end = total_bytes - val
    segs = []
    start, i = 0, 0
    while start < train_end:
        end = min(start + SHARD_BYTES, train_end)
        segs.append((f"train-{i:05d}.u32le.bin", start, end))
        start, i = end, i + 1
    segs.append(("val-00000.u32le.bin", train_end, total_bytes))
    return segs


def trained_on_files() -> list[dict]:
    """7 tokenized memmaps then the publish objects, in a fixed order."""
    out = []
    for d in DOMAINS:
        out.append({"name": f"tokenized/{d}/{d}.npy", "path": str(BUILD / "tokenized" / d / f"{d}.npy"), "domain": d, "kind": "npy"})
    for d in sorted(DOMAINS):
        for p in sorted((PUBLISH / "tokens" / d).iterdir()):
            out.append({"name": f"tokens/{d}/{p.name}", "path": str(p), "domain": d, "kind": "bin"})
    return out


def sources_list() -> list[dict]:
    return json.loads((MB / "hf_pathsinfo.json").read_text())


def safe(name: str) -> str:
    return name.replace("/", "__")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_json_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n")
    tmp.replace(path)


# Exact copies of the reading logic in scripts/trim_and_tokenize_regmix.py
# (sha256 4e8584f0...), except that the physical line index is kept.


def open_text_stream(path: Path, name: str):
    if name.endswith(".jsonl.zstd") or name.endswith(".jsonl.zst"):
        import zstandard as zstd

        raw = open(path, "rb")
        reader = zstd.ZstdDecompressor().stream_reader(raw)
        return io.TextIOWrapper(reader, encoding="utf-8"), raw
    if name.endswith(".json.gz") or name.endswith(".jsonl.gz"):
        return gzip.open(path, "rt", encoding="utf-8"), None
    raise ValueError(f"unsupported format: {name}")


def doc_text(obj: dict) -> str:
    for key in ("text", "content", "code", "body"):
        val = obj.get(key)
        if isinstance(val, str) and val:
            return val
    return "\n".join(v for v in obj.values() if isinstance(v, str))


# --------------------------------------------------------------------------- hash


def cmd_hash(task: int) -> None:
    f = trained_on_files()[task]
    path = Path(f["path"])
    size = path.stat().st_size
    t0 = time.time()
    rec = {"name": f["name"], "path": str(path), "domain": f["domain"], "kind": f["kind"], "bytes": size}
    h = hashlib.sha256()
    if f["kind"] == "bin":
        rec["sha256"] = sha256_file(path)
        rec["tokens"] = size // 4
        rec["seconds"] = round(time.time() - t0, 1)
        write_json_atomic(MB / "hash" / f"{safe(f['name'])}.json", rec)
        print(json.dumps(rec))
        return

    assert size % 4 == 0
    segs = publish_layout(size)
    seg_h = [hashlib.sha256() for _ in segs]
    eos_pos: list[np.ndarray] = []
    crcs: list[int] = []
    running = 0
    tok_base = 0
    with open(path, "rb") as fh:
        head = fh.read(8)
        rec["has_npy_header"] = head.startswith(b"\x93NUMPY")
        fh.seek(0)
        pos = 0
        while True:
            b = fh.read(CHUNK)
            if not b:
                break
            h.update(b)
            # feed publish segments
            for si, (_, s0, s1) in enumerate(segs):
                lo, hi = max(s0, pos), min(s1, pos + len(b))
                if lo < hi:
                    seg_h[si].update(memoryview(b)[lo - pos : hi - pos])
            arr = np.frombuffer(b, dtype="<u4")
            e = np.flatnonzero(arr == EOS)
            eos_pos.append(e.astype(np.int64) + tok_base)
            mv = memoryview(b)
            prev = 0
            for idx in e.tolist():
                end = (idx + 1) * 4
                crcs.append(zlib.crc32(mv[prev:end], running))
                running = 0
                prev = end
            if prev < len(b):
                running = zlib.crc32(mv[prev:], running)
            pos += len(b)
            tok_base += len(arr)
    positions = np.concatenate(eos_pos) if eos_pos else np.zeros(0, np.int64)
    rec["sha256"] = h.hexdigest()
    rec["tokens"] = size // 4
    rec["eos_count"] = int(len(positions))
    rec["ends_with_eos"] = bool(len(positions) and positions[-1] == size // 4 - 1)
    rec["trailing_crc_nonzero"] = bool(running != 0)
    rec["segments"] = [
        {"name": f"tokens/{f['domain']}/{n}", "start": s0, "end": s1, "sha256": seg_h[i].hexdigest()}
        for i, (n, s0, s1) in enumerate(segs)
    ]
    (MB / "eos").mkdir(parents=True, exist_ok=True)
    np.save(MB / "eos" / f"{f['domain']}.eos.npy", positions)
    np.save(MB / "eos" / f"{f['domain']}.crc.npy", np.asarray(crcs, dtype=np.uint32))
    rec["seconds"] = round(time.time() - t0, 1)
    write_json_atomic(MB / "hash" / f"{safe(f['name'])}.json", rec)
    print(json.dumps({k: v for k, v in rec.items() if k != "segments"}))


# --------------------------------------------------------------------------- index


def download(src: dict) -> Path:
    from huggingface_hub import hf_hub_download

    last = None
    for attempt in range(8):
        try:
            p = hf_hub_download(
                repo_id=REPO_ID,
                filename=src["path"],
                repo_type=REPO_TYPE,
                revision=REVISION,
                cache_dir=str(MB / "hf-cache"),
            )
            return Path(p)
        except Exception as exc:  # 429 / transient
            last = exc
            wait = min(600, 15 * 2**attempt)
            print(f"download retry {attempt} in {wait}s: {type(exc).__name__}: {exc}", flush=True)
            time.sleep(wait)
    raise SystemExit(f"download failed: {src['path']}: {last}")


def cmd_index(task: int) -> None:
    src = sources_list()[task]
    t0 = time.time()
    local = download(src)
    t_dl = time.time() - t0
    size = local.stat().st_size
    sha = sha256_file(local)
    stats = {
        "src": task,
        "path": src["path"],
        "size": size,
        "size_expected": src["size"],
        "sha256": sha,
        "sha256_expected": src["lfs_sha256"],
        "sha_ok": sha == src["lfs_sha256"] and size == src["size"],
        "download_seconds": round(t_dl, 1),
    }
    if not stats["sha_ok"]:
        write_json_atomic(MB / "index" / f"src{task:02d}.json", stats)
        raise SystemExit(f"sha256/size mismatch for {src['path']}")

    name = Path(src["path"]).name
    rows: list[int] = []
    n_lines = n_blank = n_nondict_or_empty = n_fallback = 0
    fallback_rows: list[int] = []
    text_chars = 0
    stream, raw = open_text_stream(local, name)
    try:
        for i, line in enumerate(stream):
            n_lines += 1
            s = line.strip()
            if not s:
                n_blank += 1
                continue
            obj = json.loads(s)
            t = doc_text(obj) if isinstance(obj, dict) else ""
            if not t:
                n_nondict_or_empty += 1
                continue
            if t is not obj.get("text"):
                n_fallback += 1
                if len(fallback_rows) < 50:
                    fallback_rows.append(i)
            text_chars += len(t)
            rows.append(i)
    finally:
        stream.close()
        if raw is not None:
            raw.close()
    stats.update(
        {
            "lines": n_lines,
            "blank_lines": n_blank,
            "skipped_no_text": n_nondict_or_empty,
            "docs": len(rows),
            "docs_text_from_fallback_field": n_fallback,
            "fallback_rows_head": fallback_rows,
            "text_chars": text_chars,
        }
    )
    if name.endswith((".zstd", ".zst")):
        # Diagnostic: the build read with stream_reader(read_across_frames=False).
        import zstandard as zstd

        def count(across: bool) -> tuple[int, int]:
            nb = nl = 0
            with open(local, "rb") as fh:
                r = zstd.ZstdDecompressor().stream_reader(fh, read_across_frames=across)
                while True:
                    b = r.read(CHUNK)
                    if not b:
                        break
                    nb += len(b)
                    nl += b.count(b"\n")
            return nb, nl

        stats["zstd_first_frame_bytes_lines"] = count(False)
        stats["zstd_all_frames_bytes_lines"] = count(True)
    (MB / "index").mkdir(parents=True, exist_ok=True)
    np.save(MB / "index" / f"src{task:02d}.rows.npy", np.asarray(rows, dtype=np.uint32))
    stats["seconds"] = round(time.time() - t0, 1)
    write_json_atomic(MB / "index" / f"src{task:02d}.json", stats)
    print(json.dumps(stats))


# --------------------------------------------------------------------------- manifest


def hf_path_of(build_path: str) -> str:
    marker = "/regmix-10b-20260725-124810/data/"
    assert marker in build_path, build_path
    return build_path.split(marker, 1)[1]


_DOMAIN_SELECTION_CACHE: dict[str, dict] = {}


def domain_selection(d: str) -> dict:
    """Replays the trim's file materialization + seed-42 shuffle for domain d.

    Independent of the tokenized .npy: only needs plan/trim_results.json and the
    per-source row lists from `--stage index`. Returns, per kept document in OUTPUT
    order: which source (src) and which physical row in that source's decompressed
    text stream (row). Cached because both cmd_manifest and cmd_retok call it.
    """
    if d in _DOMAIN_SELECTION_CACHE:
        return _DOMAIN_SELECTION_CACHE[d]
    srcs = sources_list()
    path_to_src = {s["path"]: i for i, s in enumerate(srcs)}
    trim = {t["domain"]: t for t in json.loads((BUILD / "plan" / "trim_results.json").read_text())}
    tr = trim[d]
    files = [hf_path_of(p) for p in tr["source_shards"]]
    src_ids = [path_to_src[f] for f in files]
    rows_list = [np.load(MB / "index" / f"src{s:02d}.rows.npy").astype(np.int64) for s in src_ids]
    counts = [len(r) for r in rows_list]
    n = sum(counts)
    assert n == tr["docs_before"], (d, n, tr["docs_before"])
    order = list(range(n))
    random.Random(SEED).shuffle(order)
    keep = np.asarray(order[: tr["docs_after"]], dtype=np.int64)
    cum = np.concatenate([[0], np.cumsum(counts)])
    fidx = np.searchsorted(cum, keep, side="right") - 1
    local = keep - cum[fidx]
    src = np.asarray(src_ids, dtype=np.int64)[fidx]
    row = np.empty(len(keep), dtype=np.int64)
    for k, r in enumerate(rows_list):
        m = fidx == k
        row[m] = r[local[m]]
    sel = {"tr": tr, "files": files, "src_ids": src_ids, "counts": counts, "src": src, "row": row}
    _DOMAIN_SELECTION_CACHE[d] = sel
    return sel


# --------------------------------------------------------------------------- retok
#
# Full retokenization of a domain known to be affected by the embedded-EOS hazard
# (see AFFECTED_DOMAINS above). One task per (source file used by the domain, piece
# of that file's needed rows); pieces split the row list by COUNT (not by token
# budget -- we don't know per-doc length ahead of time, that's what this computes),
# so each task streams its file from the start, skips rows outside its piece, and
# tokenizes only the rows inside it. Slower than the EOS-scan for domains where the
# scan is exact, which is why only the 5 affected domains go through this path.


def retok_units() -> list[dict]:
    units = []
    for d in AFFECTED_DOMAINS:
        sel = domain_selection(d)
        pieces = PIECES_PER_FILE[d]
        for file_local_idx in range(len(sel["src_ids"])):
            for piece_idx in range(pieces):
                units.append({"domain": d, "file_local_idx": file_local_idx, "piece_idx": piece_idx, "pieces": pieces})
    return units


def retok_part_path(d: str, file_local_idx: int, piece_idx: int) -> Path:
    return MB / "retok" / d / f"part-f{file_local_idx:02d}-p{piece_idx:02d}"


def cmd_retok(task: int) -> None:
    unit = retok_units()[task]
    d, file_local_idx, piece_idx, pieces = unit["domain"], unit["file_local_idx"], unit["piece_idx"], unit["pieces"]
    t0 = time.time()
    sel = domain_selection(d)
    src_id = sel["src_ids"][file_local_idx]
    idx_for_file = np.flatnonzero(sel["src"] == src_id)
    rows_for_file = sel["row"][idx_for_file]
    order_by_row = np.argsort(rows_for_file, kind="stable")
    rows_sorted = rows_for_file[order_by_row]
    j_sorted = idx_for_file[order_by_row]
    bounds = np.array_split(np.arange(len(rows_sorted)), pieces)[piece_idx]
    out_j: list[int] = []
    out_ntok: list[int] = []
    out_crc: list[int] = []
    embedded: list[dict] = []

    if len(bounds) > 0:
        rows_chunk = rows_sorted[bounds]
        j_chunk = j_sorted[bounds]
        srcs = sources_list()
        fmeta = srcs[src_id]
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer

        local = Path(
            hf_hub_download(
                repo_id=REPO_ID,
                filename=fmeta["path"],
                repo_type=REPO_TYPE,
                revision=REVISION,
                cache_dir=str(MB / "hf-cache"),
            )
        )
        name = Path(fmeta["path"]).name
        tok = AutoTokenizer.from_pretrained(TOKENIZER_REPO, revision=TOKENIZER_REVISION, cache_dir=str(MB / "hf-cache"), use_fast=True)

        batch_texts: list[str] = []
        batch_j: list[int] = []
        batch_rows: list[int] = []

        def flush() -> None:
            if not batch_texts:
                return
            enc = tok(batch_texts, add_special_tokens=False, padding=False, truncation=False)["input_ids"]
            for jj, rr, ids in zip(batch_j, batch_rows, enc):
                n = len(ids)
                arr = np.empty(n + 1, dtype="<u4")
                arr[:n] = ids
                arr[n] = EOS
                out_j.append(int(jj))
                out_ntok.append(n)
                out_crc.append(zlib.crc32(arr.tobytes()))
                eos_at = [p for p, v in enumerate(ids) if v == EOS]
                if eos_at and len(embedded) < 20:
                    p0 = eos_at[0]
                    ctx = tok.decode(ids[max(0, p0 - 12) : p0 + 4])
                    embedded.append({"src": src_id, "row": int(rr), "pos_in_doc": p0, "n_occurrences": len(eos_at), "context": ctx})
                elif eos_at:
                    embedded.append({"src": src_id, "row": int(rr), "pos_in_doc": eos_at[0], "n_occurrences": len(eos_at)})
            batch_texts.clear()
            batch_j.clear()
            batch_rows.clear()

        stream, raw = open_text_stream(local, name)
        k = 0
        total = len(rows_chunk)
        try:
            want = int(rows_chunk[0])
            for i, line in enumerate(stream):
                if i != want:
                    continue
                obj = json.loads(line.strip())
                text = doc_text(obj) if isinstance(obj, dict) else ""
                if not text:
                    raise SystemExit(f"{d} src{src_id} row {i}: manifest row is not a document")
                batch_texts.append(text)
                batch_j.append(int(j_chunk[k]))
                batch_rows.append(int(rows_chunk[k]))
                k += 1
                if len(batch_texts) >= BATCH_DOCS:
                    flush()
                if k == total:
                    break
                want = int(rows_chunk[k])
            flush()
        finally:
            stream.close()
            if raw is not None:
                raw.close()
        if k != total:
            raise SystemExit(f"{d} src{src_id} piece{piece_idx}: file ended after {k} of {total} rows")

    path = retok_part_path(d, file_local_idx, piece_idx)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path.with_suffix(".npz"),
        j=np.asarray(out_j, dtype=np.int64),
        ntok=np.asarray(out_ntok, dtype=np.int64),
        crc=np.asarray(out_crc, dtype=np.uint32),
    )
    write_json_atomic(
        path.with_suffix(".json"),
        {
            "domain": d,
            "file_local_idx": file_local_idx,
            "piece_idx": piece_idx,
            "docs": len(out_j),
            "embedded_eos_count": len(embedded),
            "embedded_eos_docs": embedded,
            "seconds": round(time.time() - t0, 1),
        },
    )
    print(f"[retok] {d} file{file_local_idx} piece{piece_idx}: {len(out_j)} docs, {len(embedded)} with embedded EOS, {round(time.time() - t0, 1)}s", flush=True)


def load_retok(d: str) -> tuple[np.ndarray, np.ndarray, dict]:
    sel = domain_selection(d)
    n = sel["tr"]["docs_after"]
    ntok = np.full(n, -1, dtype=np.int64)
    crc = np.zeros(n, dtype=np.uint32)
    examples: list[dict] = []
    total_embedded = 0
    parts = sorted((MB / "retok" / d).glob("part-*.npz"))
    expected_parts = len(sel["src_ids"]) * PIECES_PER_FILE[d]
    if len(parts) != expected_parts:
        raise SystemExit(f"{d}: expected {expected_parts} retok parts, found {len(parts)}")
    for p in parts:
        z = np.load(p)
        j = z["j"]
        if len(j):
            if (ntok[j] != -1).any():
                raise SystemExit(f"{d}: {p.name} overlaps a previous retok part")
            ntok[j] = z["ntok"]
            crc[j] = z["crc"]
        meta = json.loads(p.with_name(p.stem + ".json").read_text())
        total_embedded += meta["embedded_eos_count"]
        examples.extend(meta["embedded_eos_docs"])
    missing = int((ntok < 0).sum())
    if missing:
        raise SystemExit(f"{d}: retok covers {n - missing} of {n} documents; {missing} missing")
    return ntok, crc, {"embedded_eos_docs_total": total_embedded, "embedded_eos_examples": examples[:20]}


def cmd_manifest(out_dir: Path) -> None:
    srcs = sources_list()
    hashes = {}
    for f in trained_on_files():
        hashes[f["name"]] = json.loads((MB / "hash" / f"{safe(f['name'])}.json").read_text())
    idx_stats = [json.loads((MB / "index" / f"src{i:02d}.json").read_text()) for i in range(len(srcs))]
    assert all(s["sha_ok"] for s in idx_stats)

    (out_dir / "documents").mkdir(parents=True, exist_ok=True)
    report: dict = {"domains": {}}
    outputs: list[dict] = []
    domains_meta: dict = {}
    for d in DOMAINS:
        sel = domain_selection(d)
        tr, src_ids, counts, src, row = sel["tr"], sel["src_ids"], sel["counts"], sel["src"], sel["row"]
        keep = np.arange(tr["docs_after"], dtype=np.int64)  # output-order positions; len == docs_after
        global_to_local = {s: k for k, s in enumerate(src_ids)}
        fidx = np.asarray([global_to_local[int(s)] for s in src], dtype=np.int64)
        h = hashes[f"tokenized/{d}/{d}.npy"]
        assert h["ends_with_eos"] and not h["has_npy_header"]
        embedded_info = None
        if d in AFFECTED_DOMAINS:
            # EOS-scan overcounts boundaries for these domains (see AFFECTED_DOMAINS
            # comment): retokenize every document directly instead of trusting it.
            ntok, crc, embedded_info = load_retok(d)
        else:
            pos = np.load(MB / "eos" / f"{d}.eos.npy")
            crc = np.load(MB / "eos" / f"{d}.crc.npy")
            assert len(pos) == tr["docs_after"] == len(crc), (d, len(pos), tr["docs_after"], len(crc))
            ntok = np.diff(np.concatenate([[-1], pos])) - 1
        assert int(ntok.sum()) == tr["tokens_after"], d
        assert int((ntok + 1).sum()) == tr["tokens_with_eos"] == h["tokens"], d
        assert (ntok > 0).all(), d

        # documents/<d>.tsv.gz (deterministic gzip)
        lines = ["src\trow\tntok\tcrc32\n"]
        lines += [f"{a}\t{b}\t{c}\t{e:08x}\n" for a, b, c, e in zip(src.tolist(), row.tolist(), ntok.tolist(), crc.tolist())]
        payload = "".join(lines).encode()
        gz_path = out_dir / "documents" / f"{d}.tsv.gz"
        with open(gz_path, "wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as gz:
                gz.write(payload)

        # where each publish object starts/ends in document terms
        tok_off = np.concatenate([[0], np.cumsum(ntok + 1)])
        seg_info = []
        for seg in h["segments"]:
            t0, t1 = seg["start"] // 4, seg["end"] // 4
            k0 = int(np.searchsorted(tok_off, t0, side="right") - 1)
            k1 = int(np.searchsorted(tok_off, t1, side="left") - 1)
            seg_info.append(
                {
                    "name": seg["name"],
                    "token_start": t0,
                    "token_end": t1,
                    "first_doc": k0,
                    "first_doc_token_offset": int(t0 - tok_off[k0]),
                    "last_doc": k1,
                    "last_doc_tokens_included": int(t1 - tok_off[k1]),
                    "last_doc_tokens_with_eos": int(ntok[k1] + 1),
                }
            )

        per_file = []
        for k, s in enumerate(src_ids):
            m = fidx == k
            per_file.append(
                {
                    "src": s,
                    "path": srcs[s]["path"],
                    "docs_in_file": counts[k],
                    "docs_selected": int(m.sum()),
                    "tokens_selected": int(ntok[m].sum()),
                    "rows_min": int(row[m].min()) if m.any() else None,
                    "rows_max": int(row[m].max()) if m.any() else None,
                    "lines_in_file": idx_stats[s]["lines"],
                    "lines_not_yielded": idx_stats[s]["lines"] - idx_stats[s]["docs"],
                }
            )
        domains_meta[d] = {
            "documents_file": f"documents/{d}.tsv.gz",
            "docs": int(len(keep)),
            "tokens_content": int(ntok.sum()),
            "tokens_with_eos": int((ntok + 1).sum()),
            "target_tokens": tr["target_tokens"],
            "selection": {
                "rule": (
                    "materialize the files below in this order (every JSON line whose "
                    "doc_text() is non-empty, in file order) into one list of docs_before "
                    "documents; order = list(range(docs_before)); "
                    "random.Random(42).shuffle(order); keep order[:docs_after] in that order. "
                    "No document is truncated: the trim stops after the first whole document "
                    "that brings content tokens to >= target_tokens; the 0.98/1.02 skip rule "
                    "never fired (docs_scanned == docs_after)."
                ),
                "docs_before": tr["docs_before"],
                "docs_scanned": tr["docs_scanned"],
                "docs_after": tr["docs_after"],
                "files_in_materialization_order": per_file,
            },
            "publish_objects": seg_info,
        }
        if embedded_info:
            domains_meta[d]["embedded_eos_note"] = (
                f"{embedded_info['embedded_eos_docs_total']} document(s) in this domain contain the "
                "tokenizer's own EOS token (100257, '<|endoftext|>') inside their content, not only as "
                "the appended document separator: the pinned dolma2-tokenizer has that literal string "
                "registered as a special token, and HF fast tokenizers still recognize it inside raw "
                "text even with add_special_tokens=False (that flag only controls tokens the tokenizer "
                "adds itself). Scanning the archived .npy for EOS occurrences therefore overcounts "
                "document boundaries here; ntok/crc32 below come from retokenizing every document in "
                "this domain directly (build_manifest.py --stage retok) instead."
            )
            domains_meta[d]["embedded_eos_examples"] = embedded_info["embedded_eos_examples"]
        outputs.append(
            {
                "name": f"tokenized/{d}/{d}.npy",
                "domain": d,
                "layout": "tokenized",
                "bytes": h["bytes"],
                "sha256": h["sha256"],
                "tokens": h["tokens"],
                "docs": int(len(keep)),
                "docs_complete": int(len(keep)),
            }
        )
        report["domains"][d] = {"docs": int(len(keep)), "gz_bytes": gz_path.stat().st_size}

    for d in sorted(DOMAINS):
        for seg in domains_meta[d]["publish_objects"]:
            name = seg["name"]
            hb = hashes[name]
            hn = {s["name"]: s for s in hashes[f"tokenized/{d}/{d}.npy"]["segments"]}[name]
            assert hb["sha256"] == hn["sha256"], f"publish object {name} != npy byte range"
            assert hb["bytes"] == hn["end"] - hn["start"]
            outputs.append(
                {
                    "name": name,
                    "domain": d,
                    "layout": "publish",
                    "split": "val" if name.endswith("val-00000.u32le.bin") else "train",
                    "bytes": hb["bytes"],
                    "sha256": hb["sha256"],
                    "tokens": hb["tokens"],
                    "domain_token_range": [seg["token_start"], seg["token_end"]],
                    "docs_touched": seg["last_doc"] - seg["first_doc"] + 1,
                }
            )

    tok_snap = BUILD / "hf-cache" / "hub" / "models--allenai--dolma2-tokenizer" / "snapshots" / TOKENIZER_REVISION
    tok_files = [
        {"path": p.name, "size": p.stat().st_size, "sha256": sha256_file(p)}
        for p in sorted(tok_snap.iterdir())
    ]
    sources = {
        "note": "src in documents/*.tsv.gz indexes files[]. All files are unmodified allenai/olmo-mix-1124 files at the pinned revision; sha256 is the LFS oid and was re-checked on download.",
        "files": [
            {
                "src": i,
                "repo_id": REPO_ID,
                "repo_type": REPO_TYPE,
                "revision": REVISION,
                "path": s["path"],
                "size": s["size"],
                "sha256": s["lfs_sha256"],
                "last_commit": s.get("last_commit"),
                "lines": idx_stats[i]["lines"],
                "docs": idx_stats[i]["docs"],
            }
            for i, s in enumerate(srcs)
        ],
        "tokenizer": {
            "repo_id": TOKENIZER_REPO,
            "repo_type": "model",
            "revision": TOKENIZER_REVISION,
            "files": tok_files,
            "load": "transformers.AutoTokenizer.from_pretrained(repo_id, revision=revision, use_fast=True)",
            "encode": "tok(texts, add_special_tokens=False, padding=False, truncation=False)['input_ids']",
            "eos_token_id": EOS,
        },
    }
    write_json_atomic(out_dir / "sources.json", sources)
    total_docs = sum(m["docs"] for m in domains_meta.values())
    total_tok = sum(m["tokens_with_eos"] for m in domains_meta.values())
    val_tok = sum(o["tokens"] for o in outputs if o.get("split") == "val")
    outputs_doc = {
        "corpus_id": "regmix-10b-v1",
        "published_as": "pretrain/regmix-10b v1 (edullm-data), tokens/<source>/ layout",
        "format": {
            "dtype": "uint32",
            "endianness": "little",
            "header": "none (raw token stream; the .npy files are headerless numpy memmaps despite the extension)",
            "document_separator": f"EOS token {EOS} appended after every document; no BOS",
            "eos_token_id": EOS,
        },
        "layout": {
            "tokenized": "tokenized/<d>/<d>.npy = the documents of domain d in documents/<d>.tsv.gz order, each as tokens + [EOS].",
            "publish": (
                "tokens/<d>/ cut from the same stream: val_bytes = int(stream_bytes * 0.0015) rounded down to a "
                "multiple of 4; the last val_bytes form val-00000.u32le.bin; the remaining prefix is split into "
                "train-NNNNN.u32le.bin shards of exactly 1 GiB (1073741824 bytes) except the last. Shard and "
                "train/val boundaries fall at token boundaries, not document boundaries."
            ),
            "shard_bytes": SHARD_BYTES,
            "val_fraction": VAL_FRACTION,
        },
        "totals": {
            "docs": total_docs,
            "tokens_content": total_tok - total_docs,
            "tokens_with_eos": total_tok,
            "tokens_val": val_tok,
            "tokens_train": total_tok - val_tok,
            "objects": len(outputs),
        },
        "outputs": outputs,
        "domains": domains_meta,
    }
    write_json_atomic(out_dir / "outputs.json", outputs_doc)
    report["totals"] = outputs_doc["totals"]
    report["zstd_frames"] = [
        {"src": s["src"], "first": s["zstd_first_frame_bytes_lines"], "all": s["zstd_all_frames_bytes_lines"]}
        for s in idx_stats
        if "zstd_first_frame_bytes_lines" in s
        and s["zstd_first_frame_bytes_lines"] != s["zstd_all_frames_bytes_lines"]
    ]
    report["files_with_unyielded_lines"] = [
        {k: s[k] for k in ("src", "path", "lines", "docs", "blank_lines", "skipped_no_text")}
        for s in idx_stats
        if s["lines"] != s["docs"]
    ]
    report["docs_text_from_fallback_field"] = sum(s["docs_text_from_fallback_field"] for s in idx_stats)
    write_json_atomic(out_dir / "build_report.json", report)
    print(json.dumps(report, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["hash", "index", "manifest", "count", "retok", "retok_count"])
    ap.add_argument("--task", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "-1")))
    ap.add_argument("--out", type=Path, default=MB / "manifest_out")
    a = ap.parse_args()
    if a.stage == "count":
        print(len(trained_on_files()), len(sources_list()))
    elif a.stage == "retok_count":
        print(len(retok_units()))
    elif a.stage == "hash":
        cmd_hash(a.task)
    elif a.stage == "index":
        cmd_index(a.task)
    elif a.stage == "retok":
        cmd_retok(a.task)
    else:
        cmd_manifest(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
