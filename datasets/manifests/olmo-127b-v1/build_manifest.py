#!/usr/bin/env python3
"""Derive the olmo-127b-v1 document manifest, and hash the published objects.

One-off provenance script that ran on FarmShare. Not needed to rebuild the
corpus (rebuild.py only reads sources.json / documents_per_file.tsv.gz /
doc_windows.tsv.gz / outputs.json, which this script writes). Paths below are
FarmShare paths.

What survives on FarmShare scratch (read-only) for olmo-127b-v1:

  /scratch/users/nzhao2/agent-runs/olmo127b-edullm-publish-20260730T233445Z/
    publish-stage/tokens/<source>/{train-NNNNN,val-00000}.u32le.bin  (481 files,
    the actual trained-on objects; still present even though the run also
    promoted them into edullm-data)
    cache/plan/tokenized_manifest.json  (512 input shards, no per-file hash)

Nothing about document/row selection or per-file hashes survives from the
original per-file tokenize jobs (edullm-dataset-olmo*-dolma2-tok-*,
olmohq-topup-*): only their venvs/scripts/logs (used to pin tokenizer +
library versions and confirm reading rules), not their `tokenized/` outputs.
So "documents are whole files" is re-derived from scratch here by actually
downloading and tokenizing every one of the 512 public source files (same
public HF data and same tokenizer rebuild.py uses), then compared against the
real published bytes on FarmShare to get the manifest's expected hashes.

Three passes, run as separate Slurm array/steps (see sbatch/ under this
FarmShare work dir):

  tokenize      one array task per source file (512 total). Downloads from
                HF at the pinned revision (sha256-checked against
                sources.json), tokenizes exactly as
                datasets/olmo/tokenize_olmo_shard.py did (dolma2, EOS per
                document, rows skipped when blank/invalid JSON/no "text"),
                and writes stats/src-NNNNN.json: row/doc/skip counts, ntok,
                eos_in_text, tokens_with_eos, and the sha256 of this file's
                own token stream (content + one EOS per document, in row
                order). No output tokens are kept on disk.

  scan-domain   one array task per source (7 total). Reads every stats/src-*
                for that source, sorts by sources.json join_key to get each
                file's stream_start (cumulative tokens_with_eos), lists the
                domain's real published objects (train-NNNNN.u32le.bin +
                val-00000.u32le.bin, from `ls`, recorded in objects.json), then
                makes ONE sequential pass over those published bytes,
                incrementally hashing every file-range (-> the manifest's
                per-file "published stream sha256", the ground truth
                rebuild.py's own tokenization is checked against) and every
                object-range (-> outputs.json) at the same time. This is the
                "per-file check" and the "layout check" of the review spec,
                done in one read of each domain's ~15-119 GB.

  assemble      single task. Combines stats/*.json + domain_scan/*.json with
                sources.json into documents_per_file.tsv.gz, doc_windows.tsv.gz,
                and outputs.json, after checking every domain's file streams
                tile its published stream exactly (no gap/overlap) and every
                object tiles the same stream.

Run with the venv built from this directory's requirements.txt (Python 3.12.3,
matching FarmShare's /usr/bin/python3.12) -- the same venv used for the
rebuild.py verification, so there is exactly one dependency set in play.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import rebuild as rb  # noqa: E402  (reuse download/read helpers, not Manifest)

EOS_PER_WINDOW = 65536  # doc_windows.tsv.gz granularity (diagnostic only)


# --------------------------------------------------------------------------- tokenize


class _SourcesShim:
    """Just enough of rebuild.Manifest for rb.fetch_tokenizer/fetch_source_file."""

    def __init__(self, sources_json: dict) -> None:
        self.sources_json = sources_json


def load_sources(mdir: Path) -> tuple[dict, list[dict]]:
    sj = json.loads((mdir / "sources.json").read_text(encoding="utf-8"))
    files = sorted(sj["files"], key=lambda f: f["src"])
    if [f["src"] for f in files] != list(range(len(files))):
        raise SystemExit("sources.json: src indices must be 0..N-1")
    return sj, files


def cmd_tokenize(args: argparse.Namespace) -> int:
    sj, files = load_sources(args.manifest_dir)
    f = files[args.src]
    shim = _SourcesShim(sj)
    args.cache.mkdir(parents=True, exist_ok=True)
    tok_dir = rb.fetch_tokenizer(shim, args.cache)
    tok = rb.load_tokenizer(tok_dir)
    eos = int(sj["tokenizer"]["eos_token_id"])
    chunk_docs = int(sj["tokenizer"]["call"]["batch_docs"])

    t0 = time.time()
    path = rb.fetch_source_file(f, args.cache)
    t_dl = time.time() - t0

    stream = hashlib.sha256()
    windows = rb.WindowHasher(eos, EOS_PER_WINDOW)
    n_rows = n_docs = n_tok = eos_in_text = 0
    skipped: dict[int, str] = {}
    batch_texts: list[str] = []
    batch_rows: list[int] = []

    def flush() -> None:
        nonlocal n_docs, n_tok, eos_in_text
        if not batch_texts:
            return
        enc = tok(
            batch_texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )
        for ids in enc["input_ids"]:
            arr = np.empty(len(ids) + 1, dtype=rb.DTYPE)
            arr[:-1] = ids
            arr[-1] = eos
            eos_in_text += int(np.count_nonzero(arr[:-1] == eos))
            stream.update(arr)
            windows.update(arr)
            n_docs += 1
            n_tok += len(ids)
        batch_texts.clear()
        batch_rows.clear()

    for row, text, reason in rb.iter_rows(path, f["path"]):
        n_rows += 1
        if text is None:
            skipped[row] = reason
            continue
        batch_texts.append(text)
        batch_rows.append(row)
        if len(batch_texts) >= chunk_docs:
            flush()
    flush()

    tokens_with_eos = n_tok + n_docs
    result = {
        "src": f["src"],
        "source": f["source"],
        "path": f["path"],
        "rows": n_rows,
        "docs": n_docs,
        "ntok": n_tok,
        "eos_in_text": eos_in_text,
        "tokens_with_eos": tokens_with_eos,
        "skipped": {str(k): v for k, v in sorted(skipped.items())},
        "stream_sha256": stream.hexdigest(),
        "windows": windows.finish(),
        "download_s": round(t_dl, 1),
        "total_s": round(time.time() - t0, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result) + "\n", encoding="utf-8")
    print(f"OK src={f['src']} {f['path']} rows={n_rows} docs={n_docs} "
          f"tokens_with_eos={tokens_with_eos} sha256={stream.hexdigest()[:16]} "
          f"({result['total_s']}s)", flush=True)
    return 0


# --------------------------------------------------------------------------- scan-domain


def cmd_scan_domain(args: argparse.Namespace) -> int:
    sj, files = load_sources(args.manifest_dir)
    dom = args.domain
    dom_files = [f for f in files if f["source"] == dom]
    dom_files.sort(key=lambda f: f["join_key"])

    stats = {}
    for f in dom_files:
        p = args.stats_dir / f"src-{f['src']:05d}.json"
        stats[f["src"]] = json.loads(p.read_text(encoding="utf-8"))

    pos = 0
    file_bounds = []  # (src, start, end)
    for f in dom_files:
        tok = stats[f["src"]]["tokens_with_eos"]
        file_bounds.append((f["src"], pos, pos + tok))
        pos += tok
    total_from_files = pos

    objs = json.loads((args.objects_json).read_text(encoding="utf-8"))[dom]

    def obj_key(o):
        return (o["name"].startswith("val"), o["name"])

    objs = sorted(objs, key=obj_key)
    opos = 0
    obj_bounds = []  # (name, start, end, bytes)
    for o in objs:
        toks = o["bytes"] // 4
        if o["bytes"] % 4 != 0:
            raise SystemExit(f"{dom}/{o['name']}: bytes not uint32-aligned")
        obj_bounds.append((o["name"], opos, opos + toks, o["bytes"]))
        opos += toks
    total_from_objects = opos

    if total_from_files != total_from_objects:
        raise SystemExit(
            f"{dom}: file streams total {total_from_files:,} tokens, "
            f"objects total {total_from_objects:,} tokens -- mismatch"
        )

    file_h = {src: hashlib.sha256() for src, _, _ in file_bounds}
    obj_h = {name: hashlib.sha256() for name, _, _, _ in obj_bounds}
    file_done: set[int] = set()
    obj_done: set[str] = set()

    def feed(pos0: int, arr: np.ndarray) -> None:
        end0 = pos0 + len(arr)
        for src, a, b in file_bounds:
            if src in file_done or end0 <= a or pos0 >= b:
                continue
            lo, hi = max(pos0, a), min(end0, b)
            file_h[src].update(arr[lo - pos0 : hi - pos0])
            if hi == b:
                file_done.add(src)
        for name, a, b, _nbytes in obj_bounds:
            if name in obj_done or end0 <= a or pos0 >= b:
                continue
            lo, hi = max(pos0, a), min(end0, b)
            obj_h[name].update(arr[lo - pos0 : hi - pos0])
            if hi == b:
                obj_done.add(name)

    pos = 0
    t0 = time.time()
    for o in objs:
        p = args.publish_dir / "tokens" / dom / o["name"]
        with open(p, "rb") as fh:
            while True:
                b = fh.read(1 << 27)
                if not b:
                    break
                arr = np.frombuffer(b, dtype=rb.DTYPE)
                feed(pos, arr)
                pos += len(arr)
    if pos != total_from_objects:
        raise SystemExit(f"{dom}: read {pos} tokens, expected {total_from_objects}")
    if file_done != {src for src, _, _ in file_bounds}:
        raise SystemExit(f"{dom}: {len(file_bounds) - len(file_done)} file range(s) never closed")
    if obj_done != {name for name, _, _, _ in obj_bounds}:
        raise SystemExit(f"{dom}: {len(obj_bounds) - len(obj_done)} object range(s) never closed")

    out = {
        "source": dom,
        "elapsed_s": round(time.time() - t0, 1),
        "total_tokens": total_from_objects,
        "file_streams": [
            {"src": src, "stream_start_token": a, "tokens_with_eos": b - a,
             "sha256": file_h[src].hexdigest()}
            for src, a, b in file_bounds
        ],
        "objects": [
            {"name": name, "stream_start_token": a, "tokens": b - a, "bytes": nbytes,
             "sha256": obj_h[name].hexdigest()}
            for name, a, b, nbytes in obj_bounds
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"OK domain={dom} files={len(file_bounds)} objects={len(obj_bounds)} "
          f"tokens={total_from_objects:,} ({out['elapsed_s']}s)", flush=True)
    return 0


# --------------------------------------------------------------------------- assemble


def cmd_assemble(args: argparse.Namespace) -> int:
    import csv

    sj, files = load_sources(args.manifest_dir)
    by_src = {f["src"]: f for f in files}

    stats = {}
    for f in files:
        p = args.stats_dir / f"src-{f['src']:05d}.json"
        stats[f["src"]] = json.loads(p.read_text(encoding="utf-8"))

    domains = sorted({f["source"] for f in files})
    scans = {}
    for dom in domains:
        p = args.domain_scan_dir / f"{dom}.json"
        scans[dom] = json.loads(p.read_text(encoding="utf-8"))

    mismatches = []
    for dom in domains:
        scan = scans[dom]
        by_file_src = {fs["src"]: fs for fs in scan["file_streams"]}
        for f in [f for f in files if f["source"] == dom]:
            st = stats[f["src"]]
            fs = by_file_src[f["src"]]
            if st["tokens_with_eos"] != fs["tokens_with_eos"]:
                mismatches.append((f["src"], "tokens_with_eos differ",
                                    st["tokens_with_eos"], fs["tokens_with_eos"]))
            if st["stream_sha256"] != fs["sha256"]:
                mismatches.append((f["src"], "stream_sha256 differs vs published bytes",
                                    st["stream_sha256"], fs["sha256"]))
    if mismatches:
        for m in mismatches[:20]:
            print(f"MISMATCH src={m[0]} {m[1]}: {m[2]} != {m[3]}", flush=True)
        raise SystemExit(f"{len(mismatches)} file(s) do not match the published stream; "
                          f"the manifest was NOT written")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # documents_per_file.tsv.gz
    doc_path = args.out_dir / "documents_per_file.tsv.gz"
    with gzip.open(doc_path, "wt", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["src", "hf_path", "row_start", "row_end", "skipped_rows",
                    "ndocs", "ntok", "eos_in_text"])
        for f in files:
            st = stats[f["src"]]
            skipped_str = ",".join(f"{k}:{v}" for k, v in st["skipped"].items())
            w.writerow([f["src"], f["path"], 0, st["rows"], skipped_str,
                        st["docs"], st["ntok"], st["eos_in_text"]])

    # doc_windows.tsv.gz
    win_path = args.out_dir / "doc_windows.tsv.gz"
    with gzip.open(win_path, "wt", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["src", "start_token", "ntokens", "sha256_16"])
        for f in files:
            st = stats[f["src"]]
            for start, ntok_w, h16 in st["windows"]:
                w.writerow([f["src"], start, ntok_w, h16])

    # outputs.json
    file_streams = []
    objects = []
    for dom in domains:
        scan = scans[dom]
        for fs in scan["file_streams"]:
            file_streams.append({
                "src": fs["src"],
                "stream_start_token": fs["stream_start_token"],
                "tokens_with_eos": fs["tokens_with_eos"],
                "sha256": fs["sha256"],
            })
        for o in scan["objects"]:
            objects.append({
                "name": f"tokens/{dom}/{o['name']}",
                "source": dom,
                "stream_start_token": o["stream_start_token"],
                "tokens": o["tokens"],
                "bytes": o["bytes"],
                "sha256": o["sha256"],
            })

    total_tokens = sum(s["tokens_with_eos"] for s in file_streams)
    total_bytes = sum(o["bytes"] for o in objects)
    outputs = {
        "corpus_id": "olmo-127b-v1",
        "format": {"dtype": "uint32", "endianness": "little", "eos_token_id": int(sj["tokenizer"]["eos_token_id"])},
        "doc_windows": {"eos_per_window": EOS_PER_WINDOW},
        "layout": {
            "sources": domains,
            "shard_bytes": 1073741824,
            "val_fraction": 0.0015,
            "val_rule": "last floor(train_bytes*0.0015) bytes (rounded down to a multiple of 4) "
                        "of each source's concatenated stream, carved off the tail of the last "
                        "train shard(s), written to tokens/<source>/val-00000.u32le.bin",
        },
        "totals": {"tokens_with_eos": total_tokens, "bytes": total_bytes,
                    "files": len(file_streams), "objects": len(objects)},
        "file_streams": file_streams,
        "objects": objects,
    }
    (args.out_dir / "outputs.json").write_text(json.dumps(outputs, indent=1) + "\n", encoding="utf-8")

    print(f"wrote {doc_path.name}, {win_path.name}, outputs.json: "
          f"{len(files)} files, {len(objects)} objects, {total_tokens:,} tokens, "
          f"{total_bytes:,} bytes", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest-dir", type=Path, default=HERE)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("tokenize")
    t.add_argument("--src", type=int, required=True)
    t.add_argument("--cache", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.set_defaults(fn=cmd_tokenize)

    s = sub.add_parser("scan-domain")
    s.add_argument("--domain", required=True)
    s.add_argument("--stats-dir", type=Path, required=True)
    s.add_argument("--publish-dir", type=Path, required=True)
    s.add_argument("--objects-json", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.set_defaults(fn=cmd_scan_domain)

    a = sub.add_parser("assemble")
    a.add_argument("--stats-dir", type=Path, required=True)
    a.add_argument("--domain-scan-dir", type=Path, required=True)
    a.add_argument("--out-dir", type=Path, required=True)
    a.set_defaults(fn=cmd_assemble)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
