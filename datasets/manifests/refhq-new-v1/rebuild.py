#!/usr/bin/env python3
"""Rebuild the refhq-new-v1 tokenized files byte-for-byte from public Hugging Face data.

Inputs (all in this directory):
  sources.json            public files (repo, 40-char revision, path, size, sha256) + tokenizer
  outputs.json            the 46 trained-on token files: bytes, sha256, parts, v3 object layout
  documents-<src>.tsv.gz  one row per output document, in output order

Per document the rebuild does exactly what the original pipeline did, minus the
selection steps (metadata filters, fastText English filter, holdout draw), which
are recorded in the manifest instead of recomputed:

  row   = row `row` of source file `src` (native order of that file)
  text  = flatten_conversation(row)            # verbatim copy of the code that ran
  ids   = AutoTokenizer(dolma2)(text, add_special_tokens=False)["input_ids"]
  bytes = uint32 little-endian ids + [100257]  # EOS after every document

Documents are grouped into parts of 10,000 (the original per-shard `.parts/` files);
an output is the byte concatenation of its parts. No header (the `.npy` suffix is
historical: the files are raw memmaps).

Usage:
  rebuild.py --out DIR --cache DIR                       # everything, then verify
  rebuild.py --out DIR --cache DIR --only smoltalk/math/train
  rebuild.py --out DIR --cache DIR --download-only [--src I]
  rebuild.py --out DIR --cache DIR --list-tasks > tasks.txt
  rebuild.py --out DIR --cache DIR --task-file tasks.txt --task-index N   # one part
Set HF_TOKEN for the gated NousResearch/Hermes-3-Dataset.
Exit status is non-zero if any sha256 differs; the first mismatching document is printed.
"""

from __future__ import annotations

import argparse
import codecs
import gzip
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, Mapping

HERE = Path(__file__).resolve().parent
EOS_TOKEN_ID = 100257
BATCH_SIZE = 64

# --------------------------------------------------------------------------------------
# Rendering: verbatim copy of normalize_filter_source.py (refhq_new, as run on FarmShare
# 2026-08-03). Do not edit; the manifest was matched against exactly this code.
# --------------------------------------------------------------------------------------

ROLE_LABELS: dict[str, str] = {
    "system": "System",
    "user": "User",
    "human": "User",
    "assistant": "Assistant",
    "gpt": "Assistant",
    "bot": "Assistant",
    "model": "Assistant",
    "tool": "Tool",
    "function": "Tool",
}


def _message_content(message: Mapping[str, Any]) -> str:
    content = message.get("content")
    if content is None:
        content = message.get("value")
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping):
                text = item.get("text") or item.get("content") or ""
                if isinstance(text, str) and text.strip():
                    parts.append(text)
        return "\n".join(parts).strip()
    if isinstance(content, str):
        return content.strip()
    return ""


def flatten_conversation(row: Mapping[str, Any]) -> str:
    """Flatten messages/conversations to plain text for Dolma (one example = one doc)."""
    if isinstance(row.get("text"), str) and str(row["text"]).strip():
        # Prefer explicit text only when no chat turns exist.
        messages = row.get("messages") or row.get("conversations")
        if not messages:
            return str(row["text"]).strip()

    messages = row.get("messages") or row.get("conversations") or []
    if not isinstance(messages, list) or not messages:
        for key in ("prompt", "instruction", "input", "output", "response"):
            val = row.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        return ""

    parts: list[str] = []
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        role_raw = str(message.get("role") or message.get("from") or "user").strip().lower()
        label = ROLE_LABELS.get(role_raw, role_raw.capitalize() or "User")
        content = _message_content(message)
        if not content:
            continue
        parts.append(f"{label}: {content}")
    return "\n\n".join(parts).strip()


# Row fields flatten_conversation can read. Readers load only these.
RENDER_COLUMNS = ("text", "messages", "conversations", "prompt", "instruction", "input", "output", "response")

# --------------------------------------------------------------------------------------
# Source readers. `row` is the 0-based index in the file's native order:
#   parquet: row order of the file (row groups in order)
#   json:    element index of the top-level JSON array
#   jsonl:   index among non-blank lines (every non-blank line is one JSON object)
# --------------------------------------------------------------------------------------


def iter_json_array(path: Path, chunk_bytes: int = 1 << 24) -> Iterator[tuple[int, Any]]:
    """Stream the elements of a top-level JSON array without loading the whole file."""
    decoder = json.JSONDecoder()
    utf8 = codecs.getincrementaldecoder("utf-8")()
    with open(path, "rb") as handle:
        buf = ""
        pos = 0
        eof = False

        def fill() -> None:
            nonlocal buf, pos, eof
            data = handle.read(chunk_bytes)
            if not data:
                eof = True
                buf = buf[pos:] + utf8.decode(b"", final=True)
            else:
                buf = buf[pos:] + utf8.decode(data)
            pos = 0

        def skip(chars: str) -> None:
            nonlocal pos
            while True:
                while pos < len(buf) and buf[pos] in chars:
                    pos += 1
                if pos < len(buf) or eof:
                    return
                fill()

        fill()
        skip(" \t\r\n﻿")
        if pos >= len(buf) or buf[pos] != "[":
            raise ValueError(f"{path}: not a top-level JSON array")
        pos += 1
        index = 0
        while True:
            skip(" \t\r\n,")
            if pos >= len(buf):
                raise ValueError(f"{path}: unterminated JSON array")
            if buf[pos] == "]":
                return
            while True:
                try:
                    obj, end = decoder.raw_decode(buf, pos)
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise
                    fill()
            pos = end
            yield index, obj
            index += 1


def iter_jsonl(path: Path) -> Iterator[tuple[int, str]]:
    """Yield (row, raw_line) for non-blank lines, reading like normalize_filter_source.py."""
    index = 0
    with open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield index, line
            index += 1


class ParquetReader:
    def __init__(self, path: Path, columns: tuple[str, ...] | None = RENDER_COLUMNS) -> None:
        import pyarrow.parquet as pq

        self.pf = pq.ParquetFile(path)
        md = self.pf.metadata
        self.starts: list[int] = []
        total = 0
        for i in range(md.num_row_groups):
            self.starts.append(total)
            total += md.row_group(i).num_rows
        self.num_rows = total
        names = set(self.pf.schema_arrow.names)
        self.columns = None if columns is None else [c for c in columns if c in names]

    def _rg_of(self, row: int) -> int:
        import bisect

        return bisect.bisect_right(self.starts, row) - 1

    def get(self, rows: list[int]) -> dict[int, dict]:
        out: dict[int, dict] = {}
        by_rg: dict[int, list[int]] = defaultdict(list)
        for row in rows:
            if not (0 <= row < self.num_rows):
                raise IndexError(f"row {row} out of range (rows={self.num_rows})")
            by_rg[self._rg_of(row)].append(row)
        for rg in sorted(by_rg):
            wanted = by_rg[rg]
            table = self.pf.read_row_group(rg, columns=self.columns)
            local = [r - self.starts[rg] for r in wanted]
            out.update(zip(wanted, table.take(local).to_pylist()))
        return out

    def iter_range(self, start: int, stop: int) -> Iterator[tuple[int, dict]]:
        for rg in range(len(self.starts)):
            rg_start = self.starts[rg]
            rg_stop = rg_start + self.pf.metadata.row_group(rg).num_rows
            if rg_stop <= start or rg_start >= stop:
                continue
            table = self.pf.read_row_group(rg, columns=self.columns)
            lo = max(start, rg_start) - rg_start
            hi = min(stop, rg_stop) - rg_start
            for offset, rec in enumerate(table.slice(lo, hi - lo).to_pylist()):
                yield rg_start + lo + offset, rec


class SequentialReader:
    """Forward cursor over json/jsonl rows; restarts only if asked for an earlier row."""

    def __init__(self, path: Path, fmt: str) -> None:
        self.path = path
        self.fmt = fmt
        self._reset()

    def _reset(self) -> None:
        self.it = iter_json_array(self.path) if self.fmt == "json" else iter_jsonl(self.path)
        self.next_row = 0

    def get(self, rows: list[int]) -> dict[int, dict]:
        rows = sorted(set(rows))
        if rows and rows[0] < self.next_row:
            self._reset()
        want = set(rows)
        out: dict[int, dict] = {}
        last = rows[-1] if rows else -1
        while self.next_row <= last:
            try:
                row, obj = next(self.it)
            except StopIteration:
                raise IndexError(f"{self.path}: row {last} past end ({self.next_row} rows)") from None
            self.next_row = row + 1
            if row in want:
                if self.fmt == "jsonl":
                    obj = json.loads(obj)
                if not isinstance(obj, dict):
                    raise ValueError(f"{self.path}: row {row} is not a JSON object")
                out[row] = obj
        return out

    def iter_range(self, start: int, stop: int) -> Iterator[tuple[int, dict]]:
        it = iter_json_array(self.path) if self.fmt == "json" else iter_jsonl(self.path)
        for row, obj in it:
            if row >= stop:
                return
            if row < start:
                continue
            if self.fmt == "jsonl":
                obj = json.loads(obj)
            yield row, obj


def open_reader(path: Path, fmt: str, columns: tuple[str, ...] | None = RENDER_COLUMNS):
    if fmt == "parquet":
        return ParquetReader(path, columns)
    if fmt in ("json", "jsonl"):
        return SequentialReader(path, fmt)
    raise ValueError(f"unknown format {fmt!r}")


# --------------------------------------------------------------------------------------
# Download + integrity
# --------------------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1 << 24)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _verified(path: Path, entry: Mapping[str, Any], cache: Path) -> None:
    """Check size + sha256; remember a pass (keyed on size and mtime) under cache/verified/."""
    st = path.stat()
    if st.st_size != int(entry["size"]):
        raise SystemExit(f"size mismatch for {entry['path']}: {st.st_size} != {entry['size']}")
    marker = cache / "verified" / f"{entry['sha256']}.json"
    stamp = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "path": str(path.resolve())}
    if marker.is_file():
        try:
            if json.loads(marker.read_text()) == stamp:
                return
        except (OSError, ValueError):
            pass
    got = sha256_file(path)
    if got != entry["sha256"]:
        raise SystemExit(f"sha256 mismatch for {entry['repo_id']}/{entry['path']}: {got} != {entry['sha256']}")
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        pass  # concurrent array tasks + NFS attribute-cache lag: benign race, dir exists
    tmp = marker.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(stamp))
    tmp.replace(marker)


def fetch(entry: Mapping[str, Any], cache: Path) -> Path:
    from huggingface_hub import hf_hub_download

    delay = 30
    for attempt in range(8):
        try:
            local = hf_hub_download(
                repo_id=entry["repo_id"],
                filename=entry["path"],
                revision=entry["revision"],
                repo_type=entry["repo_type"],
                cache_dir=str(cache / "hub"),
            )
            break
        except Exception as exc:  # noqa: BLE001 - retry transient Hub errors (429, 5xx)
            msg = str(exc)
            if ("401" in msg or "403" in msg or "gated" in msg.lower()) and not os.environ.get("HF_TOKEN"):
                raise SystemExit(f"{entry['repo_id']} is gated: set HF_TOKEN ({exc})") from exc
            if attempt == 7:
                raise
            print(f"download retry {attempt + 1} for {entry['path']} in {delay}s: {exc!r}", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 600)
    path = Path(local)
    _verified(path, entry, cache)
    return path


def tokenizer_dir(sources: Mapping[str, Any], cache: Path) -> Path:
    tok = sources["tokenizer"]
    paths = [
        fetch({**f, "repo_id": tok["repo_id"], "repo_type": tok["repo_type"], "revision": tok["revision"]}, cache)
        for f in tok["files"]
    ]
    dirs = {p.parent for p in paths}
    if len(dirs) != 1:
        raise SystemExit(f"tokenizer files landed in several dirs: {dirs}")
    return dirs.pop()


# --------------------------------------------------------------------------------------
# Tokenization (same call as trim_and_tokenize_regmix._encode_batch, which ran)
# --------------------------------------------------------------------------------------

_TOKENIZER = None


def _worker_init(tok_dir: str) -> None:
    global _TOKENIZER
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer

    _TOKENIZER = AutoTokenizer.from_pretrained(tok_dir, use_fast=True)


def encode_batch(texts: list[str]) -> tuple[list[int], bytes]:
    """Return (content token counts, uint32le bytes of ids+EOS for each text)."""
    import numpy as np

    assert _TOKENIZER is not None
    enc = _TOKENIZER(texts, add_special_tokens=False, padding=False, truncation=False)
    lengths: list[int] = []
    chunks: list[bytes] = []
    for ids in enc["input_ids"]:
        lengths.append(len(ids))
        arr = np.empty(len(ids) + 1, dtype="<u4")
        arr[:-1] = ids
        arr[-1] = EOS_TOKEN_ID
        chunks.append(arr.tobytes())
    return lengths, b"".join(chunks)


class Encoder:
    def __init__(self, tok_dir: Path, workers: int) -> None:
        self.workers = max(1, workers)
        if self.workers == 1:
            _worker_init(str(tok_dir))
            self.pool = None
        else:
            import multiprocessing as mp

            self.pool = mp.get_context("fork").Pool(self.workers, initializer=_worker_init, initargs=(str(tok_dir),))

    def imap(self, batches: list[list[str]]):
        if self.pool is None:
            return map(encode_batch, batches)
        return self.pool.imap(encode_batch, batches, chunksize=1)

    def close(self) -> None:
        if self.pool is not None:
            self.pool.close()
            self.pool.join()


# --------------------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------------------


class Manifest:
    def __init__(self, mdir: Path) -> None:
        self.dir = mdir
        self.sources = json.loads((mdir / "sources.json").read_text(encoding="utf-8"))
        self.outputs_meta = json.loads((mdir / "outputs.json").read_text(encoding="utf-8"))
        self.files = {int(f["src"]): f for f in self.sources["files"]}
        self.outputs = {o["name"]: o for o in self.outputs_meta["outputs"]}
        self._docs: dict[str, list[tuple[int, int, int]]] = {}
        self._loaded_sources: set[str] = set()

    def output_names(self, only: list[str] | None) -> list[str]:
        names = [o["name"] for o in self.outputs_meta["outputs"]]
        if not only:
            return names
        picked = [n for n in names if any(n == s or n.startswith(s.rstrip("/") + "/") for s in only)]
        if not picked:
            raise SystemExit(f"--only {only} matches no output; names look like 'smoltalk/math/train'")
        return picked

    def docs(self, name: str) -> list[tuple[int, int, int]]:
        source = self.outputs[name]["source"]
        if source not in self._loaded_sources:
            self._load_source(source)
        return self._docs[name]

    def _load_source(self, source: str) -> None:
        path = self.dir / f"documents-{source}.tsv.gz"
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            header = handle.readline().rstrip("\n").split("\t")
            col = {c: i for i, c in enumerate(header)}
            for key in ("domain", "split", "src", "row", "ntok"):
                if key not in col:
                    raise SystemExit(f"{path}: missing column {key}")
            i_dom, i_split, i_src, i_row, i_ntok = (col[k] for k in ("domain", "split", "src", "row", "ntok"))
            for line in handle:
                f = line.rstrip("\n").split("\t")
                name = f"{source}/{f[i_dom]}/{f[i_split]}"
                self._docs.setdefault(name, []).append((int(f[i_src]), int(f[i_row]), int(f[i_ntok])))
        self._loaded_sources.add(source)
        for name, out in self.outputs.items():
            if out["source"] == source:
                n = len(self._docs.get(name, []))
                if n != int(out["docs"]):
                    raise SystemExit(f"{name}: manifest has {n} documents, outputs.json says {out['docs']}")


# --------------------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------------------


class Builder:
    def __init__(self, man: Manifest, out: Path, cache: Path, workers: int) -> None:
        self.man = man
        self.out = out
        self.cache = cache
        self.workers = workers
        self._encoder: Encoder | None = None
        self._readers: dict[int, Any] = {}
        self.first_bad_doc: dict | None = None
        self.results: list[dict] = []

    def encoder(self) -> Encoder:
        if self._encoder is None:
            self._encoder = Encoder(tokenizer_dir(self.man.sources, self.cache), self.workers)
        return self._encoder

    def reader(self, src: int):
        if src not in self._readers:
            entry = self.man.files[src]
            self._readers[src] = open_reader(fetch(entry, self.cache), entry["format"])
        return self._readers[src]

    def part_path(self, name: str, part: Mapping[str, Any]) -> Path:
        source, domain, split = name.split("/")
        return self.out / source / domain / f"{split}.parts" / part["name"]

    def build_part(self, name: str, index: int, reuse: bool = True) -> bool:
        out = self.man.outputs[name]
        part = out["parts"][index]
        path = self.part_path(name, part)
        if reuse and path.is_file() and path.stat().st_size == int(part["bytes"]) and sha256_file(path) == part["sha256"]:
            self.results.append({"output": name, "part": part["name"], "sha256": part["sha256"], "match": True, "reused": True})
            return True
        docs = self.man.docs(name)
        start = int(part["doc_start"])
        chunk = docs[start : start + int(part["docs"])]
        need: dict[int, list[int]] = defaultdict(list)
        for src, row, _ in chunk:
            need[src].append(row)
        records: dict[int, dict[int, dict]] = {}
        for src in sorted(need):
            records[src] = self.reader(src).get(sorted(set(need[src])))
        texts = [flatten_conversation(records[src][row]) for src, row, _ in chunk]
        batches = [texts[i : i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + f".tmp{os.getpid()}")
        h = hashlib.sha256()
        nbytes = 0
        pos = 0
        with open(tmp, "wb") as handle:
            for lengths, data in self.encoder().imap(batches):
                for n in lengths:
                    src, row, ntok = chunk[pos]
                    if n != ntok and self.first_bad_doc is None:
                        self._report_doc(name, start + pos, src, row, ntok, n, texts[pos])
                    pos += 1
                handle.write(data)
                h.update(data)
                nbytes += len(data)
        tmp.replace(path)
        got = h.hexdigest()
        ok = got == part["sha256"] and nbytes == int(part["bytes"])
        self.results.append({"output": name, "part": part["name"], "sha256": got, "expected": part["sha256"], "match": ok})
        if not ok:
            print(f"MISMATCH part {name} {part['name']}: sha256 {got[:12]} != {part['sha256'][:12]} "
                  f"(bytes {nbytes} vs {part['bytes']})", flush=True)
            if self.first_bad_doc is None:
                print(f"  every document in {name} {part['name']} has the expected token count; "
                      "the difference is in token values (compare against a reference copy)", flush=True)
        return ok

    def _report_doc(self, name, position, src, row, ntok, got, text) -> None:
        entry = self.man.files[src]
        self.first_bad_doc = {
            "output": name, "position": position, "src": src, "file": entry["path"],
            "repo_id": entry["repo_id"], "row": row, "expected_ntok": ntok, "rebuilt_ntok": got,
        }
        print("FIRST MISMATCHING DOCUMENT: " + json.dumps(self.first_bad_doc), flush=True)
        print("  rendered text (first 400 chars): " + json.dumps(text[:400]), flush=True)

    def assemble(self, name: str, v3_dir: Path | None) -> bool:
        out = self.man.outputs[name]
        source, domain, split = name.split("/")
        final = self.out / source / domain / f"{split}.npy"
        tmp = final.with_name(final.name + f".tmp{os.getpid()}")
        h = hashlib.sha256()
        v3 = out.get("v3_objects") or []
        v3_results = []
        v3_idx = 0
        v3_hash = hashlib.sha256() if v3 else None
        v3_filled = 0
        v3_handle = None
        nbytes = 0

        def open_v3():
            nonlocal v3_handle
            if v3_dir is not None and v3_idx < len(v3):
                p = v3_dir / v3[v3_idx]["key"]
                p.parent.mkdir(parents=True, exist_ok=True)
                v3_handle = open(p, "wb")

        open_v3()
        with open(tmp, "wb") as handle:
            for part in out["parts"]:
                with open(self.part_path(name, part), "rb") as src:
                    while True:
                        data = src.read(1 << 24)
                        if not data:
                            break
                        handle.write(data)
                        h.update(data)
                        nbytes += len(data)
                        while data and v3_hash is not None:
                            room = int(v3[v3_idx]["bytes"]) - v3_filled
                            piece, data = data[:room], data[room:]
                            v3_hash.update(piece)
                            if v3_handle is not None:
                                v3_handle.write(piece)
                            v3_filled += len(piece)
                            if v3_filled == int(v3[v3_idx]["bytes"]):
                                got = v3_hash.hexdigest()
                                v3_results.append({"key": v3[v3_idx]["key"], "sha256": got, "match": got == v3[v3_idx]["sha256"]})
                                if v3_handle is not None:
                                    v3_handle.close()
                                    v3_handle = None
                                v3_idx += 1
                                v3_filled = 0
                                if v3_idx < len(v3):
                                    v3_hash = hashlib.sha256()
                                    open_v3()
                                else:
                                    v3_hash = None
        tmp.replace(final)
        got = h.hexdigest()
        ok = got == out["sha256"] and nbytes == int(out["bytes"])
        v3_ok = len(v3_results) == len(v3) and all(r["match"] for r in v3_results)
        self.results.append({"output": name, "sha256": got, "expected": out["sha256"], "bytes": nbytes,
                             "match": ok, "v3_objects": v3_results, "v3_match": v3_ok})
        print(f"{'OK      ' if ok else 'MISMATCH'} {name:28s} expected {out['sha256'][:12]} rebuilt {got[:12]} "
              f"v3 {'ok' if v3_ok else 'MISMATCH'} ({len(v3_results)} objects)", flush=True)
        return ok and v3_ok

    def close(self) -> None:
        if self._encoder is not None:
            self._encoder.close()


def task_lines(man: Manifest, names: list[str]) -> list[str]:
    return [f"{name} {i}" for name in names for i in range(len(man.outputs[name]["parts"]))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="output dir (tokenized/<source>/<domain>/<split>.npy layout)")
    ap.add_argument("--cache", type=Path, required=True, help="download cache dir (HF hub layout under cache/hub)")
    ap.add_argument("--manifest-dir", type=Path, default=HERE)
    ap.add_argument("--only", action="append", default=None, help="output name or prefix, e.g. smoltalk/math/train or tulu-v2")
    ap.add_argument("--part", type=int, default=None, help="build only this part index of the single --only output")
    ap.add_argument("--task-file", type=Path, default=None, help="file from --list-tasks; with --task-index builds one part")
    ap.add_argument("--task-index", type=int, default=None)
    ap.add_argument("--list-tasks", action="store_true", help="print '<output> <part>' lines and exit")
    ap.add_argument("--download-only", action="store_true")
    ap.add_argument("--src", type=int, action="append", default=None, help="with --download-only: only these source files")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) or min(4, os.cpu_count() or 1))
    ap.add_argument("--v3-dir", type=Path, default=None, help="also write the published v3 objects (tokens/<source>/<domain>/<split>-NNNNN.u32le.bin)")
    ap.add_argument("--no-reuse", action="store_true", help="rebuild parts even if a verified copy exists")
    args = ap.parse_args()

    man = Manifest(args.manifest_dir)
    names = man.output_names(args.only)
    if args.list_tasks:
        print("\n".join(task_lines(man, names)))
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)

    if args.download_only:
        srcs = args.src if args.src else sorted(man.files)
        for src in srcs:
            p = fetch(man.files[src], args.cache)
            print(f"ok src={src} {man.files[src]['repo_id']}/{man.files[src]['path']} -> {p}", flush=True)
        if not args.src:
            print(f"ok tokenizer -> {tokenizer_dir(man.sources, args.cache)}", flush=True)
        return 0

    builder = Builder(man, args.out, args.cache, args.workers)
    try:
        if args.task_file is not None or args.part is not None:
            if args.task_file is not None:
                if args.task_index is None:
                    raise SystemExit("--task-file needs --task-index")
                lines = [ln.split() for ln in args.task_file.read_text().splitlines() if ln.strip()]
                name, index = lines[args.task_index][0], int(lines[args.task_index][1])
            else:
                if len(names) != 1:
                    raise SystemExit("--part needs exactly one --only output")
                name, index = names[0], args.part
            t0 = time.time()
            ok = builder.build_part(name, index, reuse=not args.no_reuse)
            r = builder.results[-1]
            print(f"{'OK' if ok else 'MISMATCH'} {name} part {index} ({r['part']}) sha256 {r['sha256'][:12]} "
                  f"in {time.time() - t0:.1f}s", flush=True)
            return 0 if ok else 1

        all_ok = True
        for name in names:
            for index in range(len(man.outputs[name]["parts"])):
                builder.build_part(name, index, reuse=not args.no_reuse)
            all_ok &= builder.assemble(name, args.v3_dir)
        report = {"outputs": [r for r in builder.results if "part" not in r],
                  "parts": [r for r in builder.results if "part" in r],
                  "first_mismatching_document": builder.first_bad_doc}
        (args.out / "rebuild_report.json").write_text(json.dumps(report, indent=1) + "\n")
        n_ok = sum(1 for r in report["outputs"] if r["match"] and r["v3_match"])
        print(f"\n{n_ok}/{len(report['outputs'])} outputs sha256-identical", flush=True)
        if builder.first_bad_doc:
            print("first mismatching document: " + json.dumps(builder.first_bad_doc), flush=True)
        return 0 if all_ok else 1
    finally:
        builder.close()


if __name__ == "__main__":
    sys.exit(main())
