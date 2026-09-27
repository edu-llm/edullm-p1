#!/usr/bin/env python3
"""Standalone byte-for-byte rebuild of refhq-regmix-5p5b-v1 from public Hugging Face data.

Usage:
    python rebuild.py --out DIR --cache DIR [--only DOMAIN]

For each domain (dclm, arxiv, starcoder, pes2o, open-web-math, algebraic-stack, wiki):
  1. Download the exact source files listed in sources.json (huggingface_hub, pinned
     revisions; anonymous except bigcode/starcoderdata, which needs HF_TOKEN and prior
     acceptance of its gated terms) into --cache, verifying each file's sha256.
  2. Replay documents-<domain>.tsv.gz (one row per output document, in output order) to
     reconstruct that domain's selected document-text stream EXACTLY as it was tokenized
     upstream. No filter, classifier, or random seed is re-run: the selection is data, not
     code, in this manifest.
  3. Tokenize the reconstructed text stream with allenai/dolma2-tokenizer at the pinned
     revision (add_special_tokens=False), appending EOS token id 100257 after every
     document, to build a headerless uint32 little-endian memmap matching
     tokenized/<domain>/<domain>.npy.
  4. Split into 1 GiB train shards plus a 0.15% tail validation shard per domain (the same
     rule scripts/publish_refhq_edullm_data.py used) and sha256 every object.
  5. Compare every sha256 (whole npy + every publish object) against outputs.json. Any
     mismatch: print the first mismatching document (index, id, expected vs. actual text
     head) and exit non-zero.

Only the requirements pinned in requirements.txt are needed. Only the source files actually
referenced by documents-<domain>.tsv.gz are downloaded -- nothing else in any of these
repos is fetched.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterator

HERE = Path(__file__).resolve().parent
EOS_TOKEN_ID = 100257
TARGET_TOKENIZER = "allenai/dolma2-tokenizer"
TARGET_TOKENIZER_REV = "5292e5d6c0f40b67cc765fe41bec991cf4345b5c"
SHARD_BYTES = 1_073_741_824
VAL_FRACTION = 0.0015

DOMAINS = ["dclm", "arxiv", "starcoder", "pes2o", "open-web-math", "algebraic-stack", "wiki"]


def log(*a: Any) -> None:
    print(*a, file=sys.stderr, flush=True)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def download_and_verify(repo_id: str, filename: str, revision: str, repo_type: str,
                         expected_sha256: str | None, cache_dir: Path, token: str | None = None) -> Path:
    from huggingface_hub import hf_hub_download

    local = Path(hf_hub_download(repo_id=repo_id, filename=filename, revision=revision,
                                  repo_type=repo_type, local_dir=str(cache_dir), token=token))
    if expected_sha256:
        got = sha256_file(local)
        if got != expected_sha256:
            raise SystemExit(
                f"sha256 mismatch for {repo_id}:{filename}@{revision}\n  expected {expected_sha256}\n  got      {got}"
            )
    return local


def read_tsv_gz(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


class DocStream:
    """Reconstructs one domain's selected-document text stream, in output order.

    Documents are shuffled relative to their source file's native row order, so a naive
    per-document "open file, seek to row" would re-read a jsonl.gz file from byte 0 for
    every single row (O(rows^2)) and re-read parquet row groups redundantly. Instead, for
    each referenced file we do ONE pass gathering every needed row's text into memory, then
    yield in the original (output) order.
    """

    def __init__(self, domain: str, sources: dict, cache_dir: Path, token: str | None):
        self.domain = domain
        self.sources = sources
        self.cache_dir = cache_dir
        self.token = token

    def _domain_spec(self) -> dict:
        return self.sources["domains"][self.domain]

    def _file_entry(self, rel_path: str) -> dict:
        spec = self._domain_spec()
        for f in spec["files"]:
            if f["path"] == rel_path:
                return f
        raise KeyError(f"{rel_path} not listed in sources.json domains.{self.domain}.files")

    def _local_path(self, rel_path: str) -> Path:
        spec = self._domain_spec()
        entry = self._file_entry(rel_path)
        return download_and_verify(spec["repo_id"], rel_path, spec["revision"], spec.get("repo_type", "dataset"),
                                    entry.get("sha256"), self.cache_dir / self.domain, self.token)

    def _bulk_parquet_texts(self, rel_path: str, rows_needed: set[int], column: str = "text") -> dict[int, str]:
        import pyarrow.parquet as pq

        local = self._local_path(rel_path)
        pf = pq.ParquetFile(str(local))
        out: dict[int, str] = {}
        base = 0
        for rg in range(pf.metadata.num_row_groups):
            n = pf.metadata.row_group(rg).num_rows
            if any(base <= r < base + n for r in rows_needed):
                tbl = pf.read_row_group(rg, columns=[column]).column(column)
                for r in rows_needed:
                    if base <= r < base + n:
                        out[r] = tbl[r - base].as_py()
            base += n
            if len(out) >= len(rows_needed):
                break
        return out

    def _bulk_jsonl_texts(self, rel_path: str, rows_needed: set[int]) -> dict[int, str]:
        local = self._local_path(rel_path)
        out: dict[int, str] = {}
        max_row = max(rows_needed)
        with gzip.open(local, "rt", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i in rows_needed:
                    out[i] = json.loads(line).get("text", "")
                if i >= max_row:
                    break
        return out

    def iter_docs(self, doc_rows: list[dict[str, str]]) -> Iterator[str]:
        domain = self.domain
        if domain == "dclm":
            yield from self._dclm_docs(doc_rows)
            return
        if domain == "starcoder":
            yield from self._starcoder_docs(doc_rows)
            return
        bulk_fn = self._bulk_parquet_texts if domain in ("wiki", "algebraic-stack", "open-web-math") \
            else self._bulk_jsonl_texts

        by_file: dict[str, set[int]] = {}
        for row in doc_rows:
            by_file.setdefault(row["file"], set()).add(int(row["row"]))
        text_by_file_row: dict[tuple[str, int], str] = {}
        for rel_path, rows_needed in by_file.items():
            for r, text in bulk_fn(rel_path, rows_needed).items():
                text_by_file_row[(rel_path, r)] = text

        for row in doc_rows:
            yield text_by_file_row.get((row["file"], int(row["row"])), "")

    # ---- starcoder: parquet `content` + recorded copyright/comment-block spans removed;
    # ---- a handful of documents are "templated" placeholders with no source row at all
    # ---- (see README's "StarCoder placeholder documents" section) ----
    def _starcoder_docs(self, doc_rows: list[dict[str, str]]) -> Iterator[str]:
        spec = self._domain_spec()
        repo_id = spec["repo_id"]

        def is_templated(row: dict[str, str]) -> bool:
            return (row.get("templated") or "") in ("1", "true", "True")

        by_file: dict[str, set[int]] = {}
        for row in doc_rows:
            if is_templated(row):
                continue
            by_file.setdefault(row["file"], set()).add(int(row["row"]))
        text_by_file_row: dict[tuple[str, int], str] = {}
        for rel_path, rows_needed in by_file.items():
            for r, text in self._bulk_parquet_texts(rel_path, rows_needed, column="content").items():
                text_by_file_row[(rel_path, r)] = text

        for row in doc_rows:
            if is_templated(row):
                # Fully determined by (id, repo) -- no source row exists to fetch (see
                # README). Matches the exact form observed in out/starcoder verbatim.
                yield f"{row['id']}\n{repo_id}\n"
                continue
            raw = text_by_file_row.get((row["file"], int(row["row"])), "")
            spans_json = row.get("spans") or ""
            if spans_json:
                spans = json.loads(spans_json)
                parts: list[str] = []
                prev = 0
                for s, e in spans:
                    parts.append(raw[prev:s])
                    prev = e
                parts.append(raw[prev:])
                raw = "".join(parts)
            yield raw

    # ---- dclm: decode/re-encode windows of a DataDecide token memmap ----
    def _dclm_docs(self, doc_rows: list[dict[str, str]]) -> Iterator[str]:
        import numpy as np
        from transformers import AutoTokenizer

        spec = self._domain_spec()
        src_spec = spec["source_tokenizer"]
        src_tok = AutoTokenizer.from_pretrained(src_spec["repo_id"], revision=src_spec["revision"], use_fast=True)
        try:
            tgt_tok = AutoTokenizer.from_pretrained(TARGET_TOKENIZER, revision=TARGET_TOKENIZER_REV, use_fast=True)
        except Exception:  # noqa: BLE001
            tgt_tok = AutoTokenizer.from_pretrained(TARGET_TOKENIZER, revision=TARGET_TOKENIZER_REV, use_fast=False)

        arrs: dict[str, "np.ndarray"] = {}

        def get_arr(rel: str):
            if rel not in arrs:
                local = self._local_path(rel)
                size = local.stat().st_size
                if size % 2 == 0:
                    mm = np.memmap(local, dtype=np.uint16, mode="r")
                    sample = np.asarray(mm[: min(4096, mm.size)])
                    if sample.size and int(sample.max()) < 100_000:
                        arrs[rel] = np.asarray(mm).reshape(-1)
                        return arrs[rel]
                if size % 4 == 0:
                    arrs[rel] = np.asarray(np.memmap(local, dtype=np.uint32, mode="r")).reshape(-1)
                else:
                    arrs[rel] = np.asarray(np.load(local, mmap_mode="r")).reshape(-1)
            return arrs[rel]

        for row in doc_rows:
            flat = get_arr(row["file"])
            s, length = int(row["start"]), int(row["length"])
            wrapped = row["wrapped"] in ("True", "true", "1")
            if not wrapped:
                chunk = np.asarray(flat[s: s + length], dtype=np.int64)
            else:
                first = np.asarray(flat[s:], dtype=np.int64)
                second = np.asarray(flat[: length - (flat.size - s)], dtype=np.int64)
                chunk = np.concatenate([first, second])
            # transformers >= 5 auto-detects BPE tokenizers and ignores a destructive
            # clean_up_tokenization_spaces=True default (see requirements.txt note); do not
            # pass clean_up_tokenization_spaces explicitly so that behavior applies here too.
            text = src_tok.decode(chunk.tolist(), skip_special_tokens=True)
            trimmed_to = row.get("trimmed_to") or ""
            if trimmed_to:
                tgt_ids = tgt_tok.encode(text, add_special_tokens=False)
                text = tgt_tok.decode(tgt_ids[: int(trimmed_to)], skip_special_tokens=True)
            yield text


class TokenWriter:
    def __init__(self, out_path: Path, eos_token_id: int):
        self.out_path = out_path
        self.eos = eos_token_id
        self.f = out_path.open("wb")
        self.total_tokens = 0

    def write_doc_ids(self, ids: list[int]) -> None:
        import numpy as np

        arr = np.asarray(ids + [self.eos], dtype=np.uint32)
        self.f.write(arr.tobytes())
        self.total_tokens += arr.size

    def finalize(self) -> int:
        self.f.close()
        return self.total_tokens


def tokenize_domain(domain: str, texts: Iterator[str], out_npy: Path) -> dict:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TARGET_TOKENIZER, revision=TARGET_TOKENIZER_REV, use_fast=True)
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    writer = TokenWriter(out_npy, EOS_TOKEN_ID)
    docs = 0
    content_tokens = 0
    for text in texts:
        ids = tok.encode(text, add_special_tokens=False)
        content_tokens += len(ids)
        writer.write_doc_ids(ids)
        docs += 1
        if docs % 100_000 == 0:
            log(f"  {domain}: tokenized {docs:,} docs, {content_tokens:,} content tokens")
    total = writer.finalize()
    return {"domain": domain, "docs": docs, "content_tokens": content_tokens, "stream_tokens_with_eos": total}


def split_npy_to_shards(npy_path: Path, out_dir: Path, shard_bytes: int) -> list[Path]:
    shard_bytes -= shard_bytes % 4
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    i = 0
    with npy_path.open("rb") as src:
        while True:
            chunk = src.read(shard_bytes)
            if not chunk:
                break
            p = out_dir / f"train-{i:05d}.u32le.bin"
            p.write_bytes(chunk)
            written.append(p)
            i += 1
    return written


def carve_val_holdout(out_dir: Path, fraction: float = VAL_FRACTION) -> Path:
    train_shards = sorted(out_dir.glob("train-*.u32le.bin"))
    total_bytes = sum(p.stat().st_size for p in train_shards)
    val_bytes = int(total_bytes * fraction)
    val_bytes -= val_bytes % 4
    remaining = val_bytes
    chunks = []
    for shard in reversed(train_shards):
        if remaining <= 0:
            break
        data = shard.read_bytes()
        take = min(remaining, len(data))
        take -= take % 4
        if take <= 0:
            continue
        if take >= len(data):
            chunks.append(data)
            shard.unlink()
        else:
            shard.write_bytes(data[:-take])
            chunks.append(data[-take:])
        remaining -= take
    val_path = out_dir / "val-00000.u32le.bin"
    val_path.write_bytes(b"".join(reversed(chunks)))
    return val_path


def check_outputs(domain: str, npy_path: Path, stage_dir: Path, outputs: dict) -> bool:
    ok = True
    expected_npy = outputs["tokenized_npy"][domain]
    got = sha256_file(npy_path)
    status = "OK" if got == expected_npy["sha256"] else "MISMATCH"
    if status != "OK":
        ok = False
    log(f"  [{status}] tokenized/{domain}/{domain}.npy  expected={expected_npy['sha256'][:12]} got={got[:12]}")

    for obj in outputs["publish_objects"]:
        if obj["domain"] != domain:
            continue
        name = obj["name"].split("/")[-1]
        p = stage_dir / name
        if not p.is_file():
            log(f"  [MISSING] {obj['name']}")
            ok = False
            continue
        got = sha256_file(p)
        status = "OK" if got == obj["sha256"] else "MISMATCH"
        if status != "OK":
            ok = False
        log(f"  [{status}] {obj['name']}  expected={obj['sha256'][:12]} got={got[:12]}")
    return ok


def diagnose_mismatch(domain: str, doc_rows: list[dict], ds: DocStream, real_texts_hint: str = "") -> None:
    log(f"  diagnosing first few {domain} documents (decoding both sides is the caller's job;")
    log("  this prints the manifest row so the offending source row can be inspected by hand):")
    for row in doc_rows[:3]:
        log("   ", {k: row[k] for k in row})


def run_domain(domain: str, manifests_dir: Path, sources: dict, outputs: dict, cache_dir: Path,
               out_dir: Path, token: str | None) -> str:
    """Returns 'ok', 'mismatch', 'aborted', or 'skipped'."""
    tsv = manifests_dir / f"documents-{domain}.tsv.gz"
    if not tsv.is_file():
        log(f"[{domain}] SKIPPED: no documents-{domain}.tsv.gz in this manifest "
            f"(see README.md -- this domain is not yet rebuildable)")
        return "skipped"
    doc_rows = read_tsv_gz(tsv)
    log(f"[{domain}] {len(doc_rows):,} documents in manifest")

    ds = DocStream(domain, sources, cache_dir, token)
    npy_path = out_dir / "tokenized" / domain / f"{domain}.npy"
    try:
        stats = tokenize_domain(domain, ds.iter_docs(doc_rows), npy_path)
    except SystemExit as exc:
        log(f"[{domain}] ABORTED: {exc}")
        return "aborted"
    log(f"[{domain}] tokenized: {stats}")

    stage_dir = out_dir / "tokens" / domain
    split_npy_to_shards(npy_path, stage_dir, SHARD_BYTES)
    carve_val_holdout(stage_dir, VAL_FRACTION)

    ok = check_outputs(domain, npy_path, stage_dir, outputs)
    if not ok:
        diagnose_mismatch(domain, doc_rows, ds)
        return "mismatch"
    return "ok"


def chunk_doc_rows(doc_rows: list[dict], chunk_index: int, chunk_count: int) -> list[dict]:
    """Contiguous, near-equal-size slice i of n. Concatenating slices 0..n-1 in order
    reproduces the exact same document sequence (and thus token stream) as processing
    the full list at once -- chunking is a parallel execution strategy, not a different
    selection."""
    n = len(doc_rows)
    base, rem = divmod(n, chunk_count)
    pos = 0
    for i in range(chunk_count):
        size = base + (1 if i < rem else 0)
        if i == chunk_index:
            return doc_rows[pos:pos + size]
        pos += size
    raise IndexError(f"chunk_index {chunk_index} out of range for chunk_count {chunk_count}")


def run_chunk(domain: str, manifests_dir: Path, sources: dict, cache_dir: Path, token: str | None,
              chunk_index: int, chunk_count: int, part_dir: Path) -> str:
    """Tokenize only this chunk's documents, writing a part file under --part-dir. Meant
    to be run once per chunk_index (e.g. one Slurm array task each), all sharing --cache
    (safe: huggingface_hub's local_dir download is idempotent/resumable) but each writing
    its own part file -- write that part file to node-local disk ($TMPDIR/tmp) yourself by
    pointing --part-dir there, then copy the finished file to shared scratch; this script
    does not do that copy for you, to keep it filesystem-agnostic."""
    tsv = manifests_dir / f"documents-{domain}.tsv.gz"
    if not tsv.is_file():
        log(f"[{domain}] SKIPPED: no documents-{domain}.tsv.gz in this manifest")
        return "skipped"
    doc_rows_full = read_tsv_gz(tsv)
    doc_rows = chunk_doc_rows(doc_rows_full, chunk_index, chunk_count)
    log(f"[{domain}] chunk {chunk_index}/{chunk_count}: {len(doc_rows):,} of {len(doc_rows_full):,} documents")

    ds = DocStream(domain, sources, cache_dir, token)
    out_part_dir = part_dir / domain
    out_part_dir.mkdir(parents=True, exist_ok=True)
    part_path = out_part_dir / f"part-{chunk_index:05d}-of-{chunk_count:05d}.u32le.bin"
    try:
        stats = tokenize_domain(f"{domain}[{chunk_index}/{chunk_count}]", ds.iter_docs(doc_rows), part_path)
    except SystemExit as exc:
        log(f"[{domain}] chunk {chunk_index}/{chunk_count} ABORTED: {exc}")
        return "aborted"
    log(f"[{domain}] chunk {chunk_index}/{chunk_count} done: {stats} -> {part_path}")
    return "ok"


def run_assemble(domain: str, manifests_dir: Path, outputs: dict, part_dir: Path, chunk_count: int,
                  out_dir: Path) -> str:
    """Concatenate this domain's chunk_count part files (in index order -- this is exactly
    equivalent to tokenizing the whole document list in one pass) into tokenized/<domain>/
    <domain>.npy, then run the normal shard + sha256-check path."""
    src_dir = part_dir / domain
    parts = [src_dir / f"part-{i:05d}-of-{chunk_count:05d}.u32le.bin" for i in range(chunk_count)]
    missing = [str(p) for p in parts if not p.is_file()]
    if missing:
        log(f"[{domain}] ASSEMBLE ABORTED: missing part file(s): {missing}")
        return "aborted"

    npy_path = out_dir / "tokenized" / domain / f"{domain}.npy"
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    with npy_path.open("wb") as out:
        for p in parts:
            with p.open("rb") as f:
                while True:
                    buf = f.read(1 << 24)
                    if not buf:
                        break
                    out.write(buf)
    log(f"[{domain}] assembled {npy_path} from {chunk_count} parts ({npy_path.stat().st_size:,} bytes)")

    stage_dir = out_dir / "tokens" / domain
    split_npy_to_shards(npy_path, stage_dir, SHARD_BYTES)
    carve_val_holdout(stage_dir, VAL_FRACTION)
    ok = check_outputs(domain, npy_path, stage_dir, outputs)
    if not ok:
        log(f"[{domain}] MISMATCH after assembling {chunk_count} chunks "
            f"(re-check each part's source chunk range for the culprit)")
        return "mismatch"
    return "ok"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--only", default=None, help="rebuild a single domain")
    ap.add_argument("--hf-token", default=None, help="only needed for bigcode/starcoderdata")
    ap.add_argument("--chunk-index", type=int, default=None,
                     help="0-based chunk to tokenize (requires --only, --chunk-count, --part-dir)")
    ap.add_argument("--chunk-count", type=int, default=None, help="total chunks for --only's domain")
    ap.add_argument("--part-dir", type=Path, default=None, help="where chunk part files live")
    ap.add_argument("--assemble", action="store_true",
                     help="concatenate --chunk-count parts from --part-dir and verify (requires --only)")
    args = ap.parse_args()

    manifests_dir = HERE
    sources = load_json(manifests_dir / "sources.json")
    outputs = load_json(manifests_dir / "outputs.json")

    args.out.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)

    if args.chunk_count is not None:
        if not args.only:
            raise SystemExit("--chunk-count requires --only DOMAIN")
        if args.assemble:
            status = run_assemble(args.only, manifests_dir, outputs, args.part_dir, args.chunk_count, args.out)
        else:
            if args.chunk_index is None or args.part_dir is None:
                raise SystemExit("chunked tokenize mode requires --chunk-index and --part-dir")
            status = run_chunk(args.only, manifests_dir, sources, args.cache, args.hf_token,
                                args.chunk_index, args.chunk_count, args.part_dir)
        log(f"[{args.only}] status: {status}")
        return 0 if status == "ok" else 1

    domains = [args.only] if args.only else DOMAINS
    results: dict[str, str] = {}
    for domain in domains:
        results[domain] = run_domain(domain, manifests_dir, sources, outputs, args.cache, args.out, args.hf_token)

    log("")
    log("=== summary ===")
    for domain, status in results.items():
        log(f"  {domain:16s} {status}")

    if any(s in ("mismatch", "aborted") for s in results.values()):
        log("REBUILD FAILED: one or more domains did not match outputs.json (see above)")
        return 1
    if any(s == "skipped" for s in results.values()):
        log("REBUILD PARTIAL: every rebuildable domain matched outputs.json, "
            "but at least one domain has no manifest yet (see README.md)")
        return 2
    log("REBUILD OK: every domain matched outputs.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
