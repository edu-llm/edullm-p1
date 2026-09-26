#!/usr/bin/env python3
"""Derive the refhq-new-v1 document manifest from surviving FarmShare artifacts.

Not needed to rebuild the corpus (see rebuild.py); this is the one-off script
that was run ON FARMSHARE to produce sources.json / documents-<source>.tsv.gz /
outputs.json from what actually survives:

  - the six public HF source repos at the pinned revisions (raw text was NOT
    kept; we re-download and re-render/re-tokenize every metadata-filter-kept
    candidate row),
  - the trained tokenized/<source>/<domain>/{train,val}.parts/*.npy (+.json)
    files under the read-only build dir /scratch/users/nzhao2/refhq-new-v1.

Why re-download candidates instead of just trusting the old docs/out/holdout
jsonl.gz shards: those were deleted after tokenization: only the final
tokenized/*.npy survive. The metadata filter (keep_row) and domain map
(map_domain) are cheap deterministic Python and are recomputed here exactly
(verified byte-identical, modulo line endings, against the code that ran on
FarmShare on 2026-08-03/04 -- see the diff notes in README.md). The one step
that is NOT recomputed is the Dolma English-language filter (needs fastText),
so which metadata-filter-survivors also passed English filtering is recovered
by CONTENT: tokenize every metadata-kept candidate, hash its token bytes, and
match those hashes against the token bytes actually found in the trained
.npy files.

Pipeline (see DATASET-DESIGN.md / experiments/token-selection/datasets/refhq_new
for the code that ran):
  1. normalize_filter_source.py: stream each source's native rows, apply
     keep_row (metadata filter) + map_domain, render with flatten_conversation,
     write to docs/<source>/<domain>/documents-NNNNN.jsonl.gz (10k docs/shard,
     in streaming order). This is exactly "candidates" below.
  2. dolma_english_filter.py: Dolma tag+mix keeps docs scoring >=0.5 English.
     Order-preserving (Dolma's mixer streams documents in file order). NOT
     recomputed (needs fastText); recovered by hash-matching instead.
  3. holdout_docs.py: for each (source, domain), seed = sha256-derived from
     (42, source, domain); shuffles range(n_docs) and takes the first
     round(n_docs*0.0015) (capped at n_docs-1) as val by INDEX, but writes
     both train and val out by ascending index -- i.e. holdout selects WHICH
     positions go to val, but neither split reorders the surviving documents.
     Recomputed exactly (see _pair_seed/holdout_counts below, copied verbatim).
  4. tokenize_source.py + merge_tokenized.py: tokenize each holdout shard with
     the dolma2 tokenizer, uint32-LE ids + EOS(100257) per doc, concatenate
     part files (sorted by shard filename) into <split>.npy.

Manifest recovery algorithm, per (source, domain):
  a. candidates[]  = ordered (file, row, ntok, hash) after keep_row, in the
     same order normalize_filter_source.py would have streamed them (source
     files enumerated in the same order datasets.load_dataset(streaming=True)
     would list them: sorted config name for smoltalk, ascending shard index).
     hash = sha256(uint32-LE(ids + [EOS])) -- the exact bytes that would land
     in a trained .npy for that document.
  b. trained_docs[split] = ordered (hash, ntok) recovered by EOS-splitting the
     trained train.parts/val.parts (part order = sorted filename, matching
     merge_tokenized.py's concatenation order). Cross-checked against each
     part's own .json "docs" count; if a literal "<|endoftext|>" tokenized to
     EOS mid-document caused an over-split, adjacent pieces are re-merged
     until they match a candidate hash (see split_part_into_docs / phase
     "match" reconciliation).
  c. n_docs = len(trained_docs['train']) + len(trained_docs['val']); replay
     the holdout RNG with that n_docs to get, for i in range(n_docs), whether
     position i (in the *pre-holdout* / post-English-filter order) is train
     or val. Walking i=0..n_docs-1 and pulling the next unconsumed doc from
     trained_docs['train'] or trained_docs['val'] accordingly reconstructs
     the single pre-holdout order exactly (both splits individually preserve
     relative order; only membership is randomized).
  d. Walk that combined, reconstructed pre-holdout sequence against
     candidates[] with a single left-to-right hash match, consuming the
     earliest not-yet-consumed candidate with a matching hash (this is what
     "preserving order among duplicates" means: ties are broken by candidate
     order, not by which split a document lands in). A miss means the
     candidate before it was dropped by the English filter; skip forward.
     A candidate hash with no consumer by the time we reach the *next*
     matched hash is an English-filter drop (recorded, not an error). If a
     trained document is never matched, that's a genuine reproduction gap.

Phases (each independently resumable/parallel; see submit_build_manifest.sh):
  files        --print-files                 list (source, file-index) pairs to fan out over
  candidates   --source S --file-index I      tokenize one source file's kept rows -> candidates-S-I.tsv.gz
  splitparts   --source S --domain D --split P  EOS-split one output's parts -> parts-S-D-P.json + doc hashes
  match        --source S --domain D          combine + holdout replay + hash match -> matched-S-D.tsv.gz
  reduce                                      assemble sources.json / documents-*.tsv.gz / outputs.json

Run with the ORIGINAL BUILD VENV (has pyarrow/transformers/datasets already
installed); never with the rebuild venv. Never uses fastText.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterator, Mapping

# ------------------------------------------------------------------------------------
# Pinned identity (must match README.md / sources.json)
# ------------------------------------------------------------------------------------

EOS_TOKEN_ID = 100257
TOKENIZER_REPO = "allenai/dolma2-tokenizer"
TOKENIZER_REVISION = "5292e5d6c0f40b67cc765fe41bec991cf4345b5c"
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
)
BASE_SEED = 42
HOLDOUT_FRACTION = 0.0015
DOCS_PER_SHARD = 10_000  # informational only; not needed for matching

SOURCES: tuple[str, ...] = ("tulu-v2", "openhermes-25", "tulu-3", "hermes-3", "smoltalk", "dolci")
DOMAINS: tuple[str, ...] = ("general", "math", "code", "science", "chat")
SPLITS: tuple[str, ...] = ("train", "val")

SOURCE_REPO: dict[str, tuple[str, str]] = {
    "tulu-v2": ("allenai/tulu-v2-sft-mixture", "6248b175d2ccb5ec7c4aeb22e6d8ee3b21b2c752"),
    "openhermes-25": ("teknium/OpenHermes-2.5", "b82037821055c377bed0d495e72e46de3bc72e84"),
    "tulu-3": ("allenai/tulu-3-sft-mixture", "b14afda60f1bbebe55d5d2fa1e4df5042f97f8be"),
    "hermes-3": ("NousResearch/Hermes-3-Dataset", "b1fddbdcae4e6714889365d1e6ce266a45289cc9"),
    "smoltalk": ("HuggingFaceTB/smoltalk", "5feaf2fd3ffca7c237fc38d1861bc30365d48ffa"),
    "dolci": ("allenai/Dolci-Instruct-SFT", "bd3c8f3a9b2cc5a9682e44b96ddd0bb2ff027221"),
}

# Physical files to read per source, in the exact order normalize_filter_source.py's
# _iter_source_rows would have streamed them (datasets.load_dataset(..., streaming=True)
# lists a repo's train-split parquet shards in ascending shard order; smoltalk iterates
# `sorted(configs)` and skips apigen-80k/smol-constraints per exclusion_rules.yaml).
# fmt: "parquet" | "json" (top-level array) | "jsonl"
SMOLTALK_SKIP_CONFIGS = {"apigen-80k", "smol-constraints"}
SMOLTALK_CONFIG_SHARDS: dict[str, int] = {
    "all": 9,
    "everyday-conversations": 1,
    "explore-instruct-rewriting": 1,
    "longalign": 1,
    "metamathqa-50k": 1,
    "numina-cot-100k": 1,
    "openhermes-100k": 1,
    "self-oss-instruct": 1,
    "smol-magpie-ultra": 6,
    "smol-rewrite": 1,
    "smol-summarize": 1,
    "systemchats-30k": 1,
}


def source_files(source: str) -> list[dict[str, Any]]:
    """Return [{"path", "fmt", "config"}] in candidate order for one source."""
    if source == "tulu-v2":
        return [{"path": f"data/train-0000{i}-of-00003-{h}.parquet", "fmt": "parquet", "config": None}
                for i, h in enumerate(["99ee8754042a69f6", "278198836de5994c", "a52de323599d586a"])]
    if source == "openhermes-25":
        return [{"path": "openhermes2_5.json", "fmt": "json", "config": None}]
    if source == "tulu-3":
        return [{"path": f"data/train-0000{i}-of-00006.parquet", "fmt": "parquet", "config": None} for i in range(6)]
    if source == "hermes-3":
        return [{"path": "hermes-3-dataset.jsonl", "fmt": "jsonl", "config": None}]
    if source == "dolci":
        return [{"path": f"data/train-{i:05d}-of-00015.parquet", "fmt": "parquet", "config": None} for i in range(15)]
    if source == "smoltalk":
        files = []
        for config in sorted(SMOLTALK_CONFIG_SHARDS):
            if config in SMOLTALK_SKIP_CONFIGS:
                continue
            n = SMOLTALK_CONFIG_SHARDS[config]
            for i in range(n):
                path = f"data/{config}/train-{i:05d}-of-{n:05d}.parquet"
                files.append({"path": path, "fmt": "parquet", "config": config})
        return files
    raise ValueError(source)


# ------------------------------------------------------------------------------------
# Metadata filter + domain map + rendering: verbatim copies, checked byte-identical
# (modulo CRLF/LF) against experiments/token-selection/datasets/refhq_new/{exclusion.py,
# domain_map.py} and .../scripts/normalize_filter_source.py as they exist in this repo,
# which in turn diffed byte-identical (modulo CRLF/LF) against the FarmShare copy that
# actually ran on 2026-08-03/04 (datasets/refhq_new/scripts/normalize_filter_source.py
# on FarmShare, mtime 2026-08-04 02:00 -- see README.md "Provenance check").
# ------------------------------------------------------------------------------------

EXCLUSION_RULES: dict[str, Any] = {
    "tulu-v2": {"drop_source_substrings": []},
    "openhermes-25": {
        "drop_source_substrings": [],
        "drop_non_english_language": True,
        "english_language_values": ["en", "eng", "english"],
    },
    "tulu-3": {
        "drop_source_substrings": [
            "wildguardmix", "wildjailbreak", "coconot",
            "tulu-3-sft-personas-instruction-following", "aya",
        ],
    },
    "hermes-3": {"drop_source_substrings": []},
    "smoltalk": {"skip_configs": ["apigen-80k", "smol-constraints"]},
    "dolci": {
        "drop_domains": ["Safety", "Precise IF"],
        "drop_source_dataset_substrings": ["coconot", "aya", "wildguard", "wildjailbreak", "tool use"],
        "drop_if_function_calls": True,
    },
}

ROLE_LABELS: dict[str, str] = {
    "system": "System", "user": "User", "human": "User", "assistant": "Assistant",
    "gpt": "Assistant", "bot": "Assistant", "model": "Assistant", "tool": "Tool", "function": "Tool",
}


def _haystack(*values: Any) -> str:
    parts = []
    for v in values:
        if v is None:
            continue
        t = str(v).strip()
        if t:
            parts.append(t.lower())
    return " ".join(parts)


def _contains_any(haystack: str, needles: list[str] | None) -> bool:
    if not needles or not haystack:
        return False
    return any(n.lower() in haystack for n in needles if n)


def _is_non_null_tool_field(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict)):
        return len(value) > 0
    return True


def keep_row(source: str, row: Mapping[str, Any], *, smoltalk_config: str | None = None) -> bool:
    cfg = EXCLUSION_RULES[source]
    if source == "smoltalk":
        return True  # file selection already excludes skip_configs
    if source == "openhermes-25" and cfg.get("drop_non_english_language"):
        language = row.get("language")
        if language is not None and str(language).strip():
            allowed = {str(v).strip().lower() for v in cfg.get("english_language_values", [])}
            if str(language).strip().lower() not in allowed:
                return False
    if source == "tulu-3":
        haystack = _haystack(row.get("source"), row.get("id"))
        return not _contains_any(haystack, cfg.get("drop_source_substrings"))
    if source == "dolci":
        domain = str(row.get("domain") or "").strip().lower()
        drop_domains = {str(d).strip().lower() for d in cfg.get("drop_domains", [])}
        if domain and domain in drop_domains:
            return False
        haystack = _haystack(row.get("source_dataset"), row.get("id"))
        if _contains_any(haystack, cfg.get("drop_source_dataset_substrings")):
            return False
        if cfg.get("drop_if_function_calls"):
            messages = row.get("messages") or row.get("conversations") or []
            if isinstance(messages, list):
                for m in messages:
                    if not isinstance(m, Mapping):
                        continue
                    if _is_non_null_tool_field(m.get("function_calls")) or _is_non_null_tool_field(m.get("functions")):
                        return False
        return True
    haystack = _haystack(row.get("source"), row.get("dataset"), row.get("source_dataset"), row.get("id"))
    return not _contains_any(haystack, cfg.get("drop_source_substrings"))


SMOLTALK_CONFIG_DOMAIN: dict[str, str] = {
    "numina-cot-100k": "math", "metamathqa-50k": "math", "self-oss-instruct": "code",
    "everyday-conversations": "chat", "systemchats-30k": "chat", "smol-magpie-ultra": "chat",
    "openhermes-100k": "general", "smol-summarize": "general", "smol-rewrite": "general",
    "explore-instruct-rewriting": "general", "longalign": "general",
}
_TULU3_SOURCE_DOMAIN: tuple[tuple[str, str], ...] = (
    ("personas-math", "math"), ("personas_math", "math"), ("math-grade", "math"),
    ("personas-algebra", "math"), ("personas_algebra", "math"), ("numina", "math"),
    ("personas-code", "code"), ("personas_code", "code"), ("evol-codealpaca", "code"),
    ("codealpaca", "code"), ("sciriff", "science"), ("wildchat", "chat"), ("no_robots", "chat"),
    ("no-robots", "chat"), ("oasst", "chat"), ("flan", "general"), ("table-gpt", "general"),
    ("tablegpt", "general"), ("hard-coded", "general"), ("hard_coded", "general"),
)
_TULU2_DATASET_DOMAIN: tuple[tuple[str, str], ...] = (
    ("code_alpaca", "code"), ("codealpaca", "code"), ("science", "science"), ("sharegpt", "chat"),
    ("hard_coded", "chat"), ("cot", "general"), ("flan", "general"), ("open_orca", "general"),
    ("gpt4_alpaca", "general"), ("wizardlm", "general"), ("lima", "general"),
)
_OPENHERMES_CATEGORY_DOMAIN: tuple[tuple[str, str], ...] = (
    ("math", "math"), ("code", "code"), ("coding", "code"), ("science", "science"),
    ("roleplay", "chat"), ("general", "general"),
)
_DOLCI_DOMAIN_MAP: dict[str, str] = {
    "math": "math", "science": "science", "coding": "code", "code": "code",
    "chat": "chat", "other": "general", "general": "general",
}


def _first_substring_domain(haystack: str, rules: tuple[tuple[str, str], ...]) -> str | None:
    text = haystack.lower()
    for needle, domain in rules:
        if needle in text:
            return domain
    return None


def _normalize_domain(value: str | None) -> str | None:
    if value is None:
        return None
    key = str(value).strip().lower()
    if not key:
        return None
    if key in DOMAINS:
        return key
    return _DOLCI_DOMAIN_MAP.get(key)


def map_domain(source: str, row: Mapping[str, Any], *, smoltalk_config: str | None = None) -> str:
    if source == "smoltalk":
        config = (smoltalk_config or str(row.get("source") or "")).strip().lower()
        return SMOLTALK_CONFIG_DOMAIN.get(config, "general")
    if source == "tulu-3":
        h = " ".join(str(row.get(k) or "") for k in ("source", "id")).lower()
        return _first_substring_domain(h, _TULU3_SOURCE_DOMAIN) or "general"
    if source == "tulu-v2":
        h = " ".join(str(row.get(k) or "") for k in ("dataset", "source", "id")).lower()
        return _first_substring_domain(h, _TULU2_DATASET_DOMAIN) or "general"
    if source == "dolci":
        mapped = _normalize_domain(str(row.get("domain") or "") or None)
        if mapped:
            return mapped
        h = str(row.get("source_dataset") or "").lower()
        if "math" in h:
            return "math"
        if "science" in h:
            return "science"
        if "code" in h or "python" in h or "coding" in h:
            return "code"
        if "wildchat" in h or "chat" in h:
            return "chat"
        return "general"
    if source == "openhermes-25":
        category = str(row.get("category") or row.get("source") or "").lower()
        return _first_substring_domain(category, _OPENHERMES_CATEGORY_DOMAIN) or "general"
    return "general"  # hermes-3


def _message_content(message: Mapping[str, Any]) -> str:
    content = message.get("content")
    if content is None:
        content = message.get("value")
    if isinstance(content, list):
        parts = []
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
    if isinstance(row.get("text"), str) and str(row["text"]).strip():
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
    parts = []
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


# ------------------------------------------------------------------------------------
# Holdout replay: verbatim logic of refhq_new_sources.holdout_counts + holdout_docs._pair_seed
# ------------------------------------------------------------------------------------


def pair_seed(base_seed: int, source: str, domain: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{source}:{domain}".encode("utf-8")).hexdigest()
    return (base_seed + int(digest[:8], 16)) % (2**31 - 1)


def holdout_counts(n_docs: int, fraction: float = HOLDOUT_FRACTION) -> tuple[int, int]:
    if n_docs <= 0:
        return 0, 0
    n_val = int(round(n_docs * fraction))
    if n_docs > 1:
        n_val = min(n_val, n_docs - 1)
    else:
        n_val = 0
    return n_docs - n_val, n_val


def replay_holdout_membership(n_docs: int, seed: int, n_val: int) -> list[bool]:
    """Return is_val[i] for i in range(n_docs), replaying holdout_docs.py exactly."""
    rng = random.Random(seed)
    order = list(range(n_docs))
    rng.shuffle(order)
    val_idx = set(order[:n_val])
    return [i in val_idx for i in range(n_docs)]


# ------------------------------------------------------------------------------------
# Row readers
# ------------------------------------------------------------------------------------


def iter_json_array(path: Path, chunk_bytes: int = 1 << 24) -> Iterator[tuple[int, Any]]:
    import codecs

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


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict]]:
    index = 0
    with open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield index, json.loads(line)
            index += 1


def iter_parquet(path: Path) -> Iterator[tuple[int, dict]]:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    row = 0
    for i in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(i)
        for rec in table.to_pylist():
            yield row, rec
            row += 1


def iter_rows(path: Path, fmt: str) -> Iterator[tuple[int, dict]]:
    if fmt == "parquet":
        return iter_parquet(path)
    if fmt == "json":
        return iter_json_array(path)
    if fmt == "jsonl":
        return iter_jsonl(path)
    raise ValueError(fmt)


# ------------------------------------------------------------------------------------
# Download helpers
# ------------------------------------------------------------------------------------


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download_source_file(source: str, rel_path: str, cache: Path) -> Path:
    from huggingface_hub import hf_hub_download

    repo_id, revision = SOURCE_REPO[source]
    delay = 20
    for attempt in range(6):
        try:
            local = hf_hub_download(
                repo_id=repo_id, repo_type="dataset", filename=rel_path, revision=revision,
                cache_dir=str(cache / "hub"),
            )
            return Path(local)
        except Exception as exc:  # noqa: BLE001
            if attempt == 5:
                raise
            print(f"retry download {source}/{rel_path}: {exc!r}", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 300)
    raise RuntimeError("unreachable")


def download_tokenizer(cache: Path) -> Path:
    from huggingface_hub import hf_hub_download

    dirs = set()
    last = None
    for fname in TOKENIZER_FILES:
        try:
            p = hf_hub_download(
                repo_id=TOKENIZER_REPO, repo_type="model", filename=fname, revision=TOKENIZER_REVISION,
                cache_dir=str(cache / "hub"),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"tokenizer file {fname} skipped: {exc!r}", flush=True)
            continue
        dirs.add(Path(p).parent)
        last = Path(p)
    if len(dirs) != 1 or last is None:
        raise SystemExit(f"tokenizer files landed inconsistently: {dirs}")
    return last.parent


# ------------------------------------------------------------------------------------
# Tokenization
# ------------------------------------------------------------------------------------

_TOKENIZER = None


def _worker_init(tok_dir: str) -> None:
    global _TOKENIZER
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer

    _TOKENIZER = AutoTokenizer.from_pretrained(tok_dir, use_fast=True)


def _encode_batch(texts: list[str]) -> list[tuple[int, str]]:
    """Return (ntok_content, sha256_hex_of_uint32le(ids+EOS)) per text."""
    import numpy as np

    assert _TOKENIZER is not None
    enc = _TOKENIZER(texts, add_special_tokens=False, padding=False, truncation=False)
    out = []
    for ids in enc["input_ids"]:
        arr = np.empty(len(ids) + 1, dtype="<u4")
        arr[:-1] = ids
        arr[-1] = EOS_TOKEN_ID
        h = hashlib.sha256(arr.tobytes()).hexdigest()
        out.append((len(ids), h))
    return out


# ------------------------------------------------------------------------------------
# Phase: candidates
# ------------------------------------------------------------------------------------


def phase_candidates(source: str, file_index: int, cache: Path, out_dir: Path, workers: int, batch_size: int) -> None:
    files = source_files(source)
    entry = files[file_index]
    rel_path = entry["path"]
    fmt = entry["fmt"]
    config = entry["config"]

    local_path = download_source_file(source, rel_path, cache)
    size = local_path.stat().st_size
    sha256 = sha256_file(local_path)
    tok_dir = download_tokenizer(cache)

    out_path = out_dir / f"candidates-{source}-{file_index:04d}.tsv.gz"
    meta_path = out_dir / f"candidates-{source}-{file_index:04d}.meta.json"

    import multiprocessing as mp

    workers = max(1, workers)
    pool = mp.get_context("fork").Pool(workers, initializer=_worker_init, initargs=(str(tok_dir),)) if workers > 1 else None
    if pool is None:
        _worker_init(str(tok_dir))

    n_seen = 0
    n_kept = 0
    domain_counts: dict[str, int] = defaultdict(int)
    t0 = time.time()
    superbatch_target = workers * batch_size * 8

    with gzip.open(out_path, "wt", encoding="utf-8") as out:
        out.write("row\tdomain\tntok\thash\n")
        pending: list[tuple[int, str]] = []  # (row, domain) per text, in `texts` order

        def flush(texts: list[str], pending_local: list[tuple[int, str]]) -> None:
            nonlocal n_kept
            if not texts:
                return
            chunks = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
            if pool is None:
                chunk_results = [_encode_batch(c) for c in chunks]
            else:
                chunk_results = pool.map(_encode_batch, chunks, chunksize=1)
            results = [r for chunk in chunk_results for r in chunk]
            for (row, domain), (ntok, h) in zip(pending_local, results):
                out.write(f"{row}\t{domain}\t{ntok}\t{h}\n")
                domain_counts[domain] += 1
                n_kept += 1

        texts: list[str] = []
        for row, rec in iter_rows(local_path, fmt):
            n_seen += 1
            if not keep_row(source, rec, smoltalk_config=config):
                continue
            text = flatten_conversation(rec)
            if not text:
                continue
            domain = map_domain(source, rec, smoltalk_config=config)
            texts.append(text)
            pending.append((row, domain))
            if len(texts) >= superbatch_target:
                flush(texts, pending)
                texts = []
                pending = []
            if n_seen % 200_000 == 0:
                print(f"{source}/{file_index} progress seen={n_seen:,} kept={n_kept:,} "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
        flush(texts, pending)
    if pool is not None:
        pool.close()
        pool.join()

    meta = {
        "source": source, "file_index": file_index, "path": rel_path, "fmt": fmt, "config": config,
        "repo_id": SOURCE_REPO[source][0], "revision": SOURCE_REPO[source][1],
        "size": size, "sha256": sha256, "n_seen": n_seen, "n_kept": n_kept,
        "domain_counts": dict(domain_counts), "elapsed_s": time.time() - t0,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)


# ------------------------------------------------------------------------------------
# Phase: splitparts (EOS-split one output's train/val parts; naive pass only --
# reconciliation of any over-split parts happens in phase "match", which has the
# candidate hashes needed to decide where the real boundaries are)
# ------------------------------------------------------------------------------------


def split_part_naive(data: bytes) -> list[tuple[int, int]]:
    """Return [(start, end)) uint32-index spans, end index inclusive of the EOS."""
    import numpy as np

    arr = np.frombuffer(data, dtype="<u4")
    eos_positions = np.nonzero(arr == EOS_TOKEN_ID)[0]
    spans = []
    start = 0
    for pos in eos_positions:
        spans.append((start, int(pos) + 1))
        start = int(pos) + 1
    if start != len(arr):
        raise SystemExit(f"trailing {len(arr) - start} tokens after last EOS (corrupt part?)")
    return spans


def phase_splitparts(build_root: Path, source: str, domain: str, split: str, out_dir: Path) -> None:
    parts_dir = build_root / "tokenized" / source / domain / f"{split}.parts"
    parts = sorted(parts_dir.glob("*.npy"))
    if not parts:
        raise SystemExit(f"no parts under {parts_dir}")

    out_path = out_dir / f"parts-{source}-{domain}-{split}.tsv.gz"
    part_stats = []
    doc_pos = 0
    with gzip.open(out_path, "wt", encoding="utf-8") as out:
        out.write("part\tdoc_in_part\tntok\thash\n")
        for part in parts:
            data = part.read_bytes()
            part_sha256 = hashlib.sha256(data).hexdigest()
            meta_path = part.with_suffix(".json")
            declared_docs = None
            if meta_path.is_file():
                declared_docs = json.loads(meta_path.read_text(encoding="utf-8")).get("docs")
            spans = split_part_naive(data)
            for k, (s, e) in enumerate(spans):
                seg = data[s * 4 : e * 4]
                h = hashlib.sha256(seg).hexdigest()
                ntok = (e - s) - 1  # exclude EOS
                out.write(f"{part.name}\t{k}\t{ntok}\t{h}\n")
            part_stats.append({
                "part": part.name, "bytes": len(data), "sha256": part_sha256,
                "naive_docs": len(spans), "declared_docs": declared_docs,
                "clean_split": declared_docs is None or len(spans) == declared_docs,
            })
            doc_pos += len(spans)
    meta = {
        "source": source, "domain": domain, "split": split,
        "parts": part_stats, "total_naive_docs": doc_pos,
        "output_bytes": sum(p["bytes"] for p in part_stats),
        "output_sha256_parts_in_order": [p["sha256"] for p in part_stats],
    }
    (out_dir / f"parts-{source}-{domain}-{split}.meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "output_sha256_parts_in_order"}, indent=2), flush=True)


# ------------------------------------------------------------------------------------
# Phase: match
# ------------------------------------------------------------------------------------


def load_candidates(work_dir: Path, source: str, domain: str, files: list[dict[str, Any]]) -> list[tuple[int, int, int, str]]:
    """Return ordered [(src_file_index, row, ntok, hash)] for this (source, domain)."""
    out: list[tuple[int, int, int, str]] = []
    for file_index, _entry in enumerate(files):
        path = work_dir / "candidates" / f"candidates-{source}-{file_index:04d}.tsv.gz"
        if not path.is_file():
            raise SystemExit(f"missing {path}; run phase candidates first")
        with gzip.open(path, "rt", encoding="utf-8") as f:
            header = f.readline()
            assert header.strip() == "row\tdomain\tntok\thash"
            for line in f:
                row_s, dom, ntok_s, h = line.rstrip("\n").split("\t")
                if dom != domain:
                    continue
                out.append((file_index, int(row_s), int(ntok_s), h))
    return out


def load_part_docs(work_dir: Path, source: str, domain: str, split: str) -> tuple[list[tuple[str, int, str, int]], dict]:
    """Return ordered [(hash, ntok, part_name, doc_in_part)] plus meta dict."""
    tsv_path = work_dir / "splitparts" / f"parts-{source}-{domain}-{split}.tsv.gz"
    meta_path = work_dir / "splitparts" / f"parts-{source}-{domain}-{split}.meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    docs = []
    with gzip.open(tsv_path, "rt", encoding="utf-8") as f:
        header = f.readline()
        assert header.strip() == "part\tdoc_in_part\tntok\thash"
        for line in f:
            part, doc_in_part, ntok_s, h = line.rstrip("\n").split("\t")
            docs.append((h, int(ntok_s), part, int(doc_in_part)))
    return docs, meta


def match_domain(build_root: Path, work_dir: Path, out_dir: Path, source: str, domain: str) -> dict:
    files = source_files(source)
    candidates = load_candidates(work_dir, source, domain, files)
    cand_bucket: dict[str, deque[int]] = defaultdict(deque)
    for idx, (_f, _r, _n, h) in enumerate(candidates):
        cand_bucket[h].append(idx)
    cand_hash_set = set(cand_bucket.keys())

    trained: dict[str, list[tuple[str, int, str, int]]] = {}
    part_metas: dict[str, dict] = {}
    for split in SPLITS:
        parts_dir = build_root / "tokenized" / source / domain / f"{split}.parts"
        if not parts_dir.is_dir():
            trained[split] = []
            continue
        docs, meta = load_part_docs(work_dir, source, domain, split)
        part_metas[split] = meta
        # Reconcile any over-split parts using raw bytes + candidate hash set.
        docs = _reconcile_with_bytes(parts_dir, meta, docs, cand_hash_set)
        trained[split] = docs

    n_docs = len(trained["train"]) + len(trained["val"])
    n_train_expected, n_val_expected = holdout_counts(n_docs, HOLDOUT_FRACTION)
    holdout_ok = (n_train_expected == len(trained["train"])) and (n_val_expected == len(trained["val"]))
    seed = pair_seed(BASE_SEED, source, domain)
    is_val = replay_holdout_membership(n_docs, seed, len(trained["val"])) if n_docs else []

    part_doc_counts: dict[str, list[list[Any]]] = {}
    for split in SPLITS:
        counts: list[list[Any]] = []
        for h, ntok, part, doc_in_part in trained[split]:
            if counts and counts[-1][0] == part:
                counts[-1][1] += 1
            else:
                counts.append([part, 1])
        part_doc_counts[split] = counts

    train_q = deque(trained["train"])
    val_q = deque(trained["val"])
    combined: list[tuple[str, int, str, int]] = []  # hash, ntok, part, doc_in_part
    for v in is_val:
        combined.append(val_q.popleft() if v else train_q.popleft())

    rows_out: dict[str, list[tuple[int, int, int]]] = {"train": [], "val": []}
    unmatched: list[dict] = []
    dropped_candidates = 0
    out_of_order = 0
    last_matched_cand_idx = -1
    for pos, (h, ntok, part, doc_in_part) in enumerate(combined):
        bucket = cand_bucket.get(h)
        cand_idx = bucket.popleft() if bucket else None
        if cand_idx is None:
            unmatched.append({"position": pos, "split": "val" if is_val[pos] else "train",
                               "part": part, "doc_in_part": doc_in_part, "hash": h, "ntok": ntok})
            continue
        if cand_idx <= last_matched_cand_idx:
            # Duplicate-hash candidate consumed out of true relative order: content is
            # still byte-identical (that's what the hash guarantees) but the specific
            # (src, row) attribution among duplicates may not reflect true provenance.
            out_of_order += 1
        else:
            dropped_candidates += cand_idx - (last_matched_cand_idx + 1)
            last_matched_cand_idx = cand_idx
        src_file_idx, row, cand_ntok, _h = candidates[cand_idx]
        split_name = "val" if is_val[pos] else "train"
        rows_out[split_name].append((src_file_idx, row, ntok))

    result = {
        "source": source, "domain": domain, "n_docs": n_docs,
        "n_train": len(trained["train"]), "n_val": len(trained["val"]),
        "holdout_seed": seed, "holdout_counts_match": holdout_ok,
        "n_candidates": len(candidates), "n_matched": sum(len(v) for v in rows_out.values()),
        "n_unmatched": len(unmatched), "n_english_filter_dropped_estimate": dropped_candidates,
        "n_out_of_order_duplicate_matches": out_of_order,
        "unmatched_sample": unmatched[:20],
        "part_doc_counts": part_doc_counts,
    }
    for split in SPLITS:
        out_path = out_dir / f"matched-{source}-{domain}-{split}.tsv.gz"
        with gzip.open(out_path, "wt", encoding="utf-8") as out:
            out.write("src\trow\tntok\n")
            for src_file_idx, row, ntok in rows_out[split]:
                out.write(f"{src_file_idx}\t{row}\t{ntok}\n")
    (out_dir / f"matched-{source}-{domain}.meta.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def _reconcile_with_bytes(parts_dir: Path, meta: dict, docs: list[tuple[str, int, str, int]],
                           cand_hash_set: set[str]) -> list[tuple[str, int, str, int]]:
    by_part: dict[str, dict] = {p["part"]: p for p in meta["parts"]}
    clean = all(by_part[p]["clean_split"] for p in by_part)
    if clean:
        return docs
    fixed: list[tuple[str, int, str, int]] = []
    docs_by_part: dict[str, list[tuple[str, int, str, int]]] = defaultdict(list)
    for d in docs:
        docs_by_part[d[2]].append(d)
    for part_name, part_docs in docs_by_part.items():
        info = by_part[part_name]
        if info["clean_split"]:
            fixed.extend(part_docs)
            continue
        data = (parts_dir / part_name).read_bytes()
        spans = split_part_naive(data)
        merged_spans: list[tuple[int, int]] = []
        i = 0
        while i < len(spans):
            start = spans[i][0]
            end = spans[i][1]
            seg_hash = hashlib.sha256(data[start * 4: end * 4]).hexdigest()
            j = i
            while seg_hash not in cand_hash_set and j + 1 < len(spans):
                j += 1
                end = spans[j][1]
                seg_hash = hashlib.sha256(data[start * 4: end * 4]).hexdigest()
            merged_spans.append((start, end))
            i = j + 1
        declared = info.get("declared_docs")
        if declared is not None and len(merged_spans) != declared:
            print(f"WARN {part_name}: reconciled to {len(merged_spans)} docs, declared {declared}", flush=True)
        for k, (s, e) in enumerate(merged_spans):
            seg = data[s * 4: e * 4]
            h = hashlib.sha256(seg).hexdigest()
            ntok = (e - s) - 1
            fixed.append((h, ntok, part_name, k))
    # Restore original part+doc_in_part ordering (ascending part name, then doc_in_part)
    fixed.sort(key=lambda d: (d[2], d[3]))
    return fixed


# ------------------------------------------------------------------------------------
# Phase: reduce -> sources.json / documents-<source>.tsv.gz / outputs.json
# ------------------------------------------------------------------------------------


# publish_refhq_new.py's split_npy_to_shards: 1 GiB (2**30) byte-aligned shards of the
# merged <split>.npy, named tokens/<source>/<domain>/{split}-NNNNN.u32le.bin. Byte-range
# split of an already-verified file, so the v3 layout is fully derived here (no S3 access,
# no finalize_upload run needed) -- profile "pretrain-tokens/v1" per PUBLISH_PROFILE.
V3_SHARD_BYTES = 1_073_741_824


def v3_objects_for_output(npy_path: Path, source: str, domain: str, split: str) -> list[dict[str, Any]]:
    objects = []
    with open(npy_path, "rb") as f:
        idx = 0
        while True:
            chunk = f.read(V3_SHARD_BYTES)
            if not chunk:
                break
            objects.append({
                "key": f"tokens/{source}/{domain}/{split}-{idx:05d}.u32le.bin",
                "bytes": len(chunk),
                "sha256": hashlib.sha256(chunk).hexdigest(),
            })
            idx += 1
    return objects


def tokenizer_file_entries(cache: Path) -> list[dict[str, Any]]:
    """size/sha256 per tokenizer file -- rebuild.py's fetch()/_verified() require both."""
    from huggingface_hub import hf_hub_download

    entries = []
    for fname in TOKENIZER_FILES:
        try:
            p = Path(hf_hub_download(
                repo_id=TOKENIZER_REPO, repo_type="model", filename=fname, revision=TOKENIZER_REVISION,
                cache_dir=str(cache / "hub"),
            ))
        except Exception as exc:  # noqa: BLE001
            print(f"tokenizer file {fname} unavailable: {exc!r}", flush=True)
            continue
        entries.append({"path": fname, "size": p.stat().st_size, "sha256": sha256_file(p)})
    if not entries:
        raise SystemExit("no tokenizer files downloaded; run phase candidates at least once first")
    return entries


# Independent cross-check: total metadata-filter-survivor rows the original FarmShare
# normalize_filter_source.py run reported per source (manifests/<source>-normalize.json
# "kept", copied from the build dir on 2026-09-25), BEFORE the (unrecomputed) Dolma
# English filter. If phase_candidates' recomputed keep_row/map_domain disagree with this,
# something about the source data or the pinned revision has drifted.
EXPECTED_KEPT_BY_SOURCE: dict[str, int] = {
    "tulu-v2": 326_154, "openhermes-25": 1_001_112, "tulu-3": 728_240,
    "hermes-3": 958_829, "smoltalk": 1_970_266, "dolci": 1_577_266,
}


def phase_reduce(build_root: Path, work_dir: Path, out_dir: Path, sources: list[str], cache: Path) -> None:
    src_files_flat: list[dict[str, Any]] = []
    src_index: dict[tuple[str, int], int] = {}
    kept_by_source: dict[str, int] = defaultdict(int)
    for source in sources:
        files = source_files(source)
        for file_index, entry in enumerate(files):
            meta_path = work_dir / "candidates" / f"candidates-{source}-{file_index:04d}.meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            kept_by_source[source] += int(meta["n_kept"])
            global_id = len(src_files_flat)
            src_index[(source, file_index)] = global_id
            src_files_flat.append({
                "src": global_id, "repo_id": meta["repo_id"], "repo_type": "dataset",
                "revision": meta["revision"], "path": meta["path"], "format": meta["fmt"],
                "size": meta["size"], "sha256": meta["sha256"],
                "source": source, "smoltalk_config": meta.get("config"),
            })

    for source in sources:
        expected = EXPECTED_KEPT_BY_SOURCE.get(source)
        got = kept_by_source.get(source, 0)
        status = "OK" if expected == got else "MISMATCH"
        print(f"{status} candidate kept-count cross-check {source}: recomputed={got} "
              f"expected(from build's normalize.json)={expected}", flush=True)

    sources_json = {
        "tokenizer": {
            "repo_id": TOKENIZER_REPO, "repo_type": "model", "revision": TOKENIZER_REVISION,
            "files": tokenizer_file_entries(cache),
        },
        "files": src_files_flat,
    }
    (out_dir / "sources.json").write_text(json.dumps(sources_json, indent=1) + "\n", encoding="utf-8")

    outputs_meta = {"eos_token_id": EOS_TOKEN_ID, "tokenizer": TOKENIZER_REPO, "outputs": []}
    all_stats = []
    for source in sources:
        doc_rows: list[tuple[str, str, int, int, int]] = []  # domain, split, src, row, ntok
        for domain in DOMAINS:
            match_meta_path = work_dir / "match" / f"matched-{source}-{domain}.meta.json"
            if not match_meta_path.is_file():
                continue
            match_meta = json.loads(match_meta_path.read_text(encoding="utf-8"))
            all_stats.append(match_meta)
            for split in SPLITS:
                tsv_path = work_dir / "match" / f"matched-{source}-{domain}-{split}.tsv.gz"
                rows = []
                with gzip.open(tsv_path, "rt", encoding="utf-8") as f:
                    header = f.readline()
                    assert header.strip() == "src\trow\tntok"
                    for line in f:
                        s, r, n = line.rstrip("\n").split("\t")
                        rows.append((int(s), int(r), int(n)))
                if not rows:
                    continue
                for s, r, n in rows:
                    doc_rows.append((domain, split, src_index[(source, s)], r, n))

                # outputs.json entry from splitparts meta (bytes/sha256 per part + whole).
                # doc_start/docs per part come from match_meta["part_doc_counts"][split],
                # which reflects the RECONCILED (post EOS-oversplit-merge) doc counts in
                # true output order -- required by rebuild.py's Builder.build_part slicing.
                parts_meta_path = work_dir / "splitparts" / f"parts-{source}-{domain}-{split}.meta.json"
                parts_meta = json.loads(parts_meta_path.read_text(encoding="utf-8"))
                parts_by_name = {p["part"]: p for p in parts_meta["parts"]}
                doc_start = 0
                parts_list = []
                for part_name, count in match_meta["part_doc_counts"][split]:
                    p = parts_by_name[part_name]
                    parts_list.append({
                        "name": p["part"], "bytes": p["bytes"], "sha256": p["sha256"],
                        "doc_start": doc_start, "docs": count,
                    })
                    doc_start += count
                if doc_start != len(rows):
                    print(f"WARN {source}/{domain}/{split}: part doc_start sum {doc_start} "
                          f"!= matched rows {len(rows)}", flush=True)
                final_npy = build_root / "tokenized" / source / domain / f"{split}.npy"
                final_bytes = final_npy.stat().st_size
                final_sha256 = sha256_file(final_npy)
                outputs_meta["outputs"].append({
                    "name": f"{source}/{domain}/{split}",
                    "path": f"tokenized/{source}/{domain}/{split}.npy",
                    "bytes": final_bytes, "sha256": final_sha256, "docs": len(rows),
                    "stream_tokens_with_eos": final_bytes // 4,
                    "source": source,
                    "parts": parts_list,
                    "v3_objects": v3_objects_for_output(final_npy, source, domain, split),
                })

        doc_path = out_dir / f"documents-{source}.tsv.gz"
        with gzip.open(doc_path, "wt", encoding="utf-8") as f:
            f.write("domain\tsplit\tsrc\trow\tntok\n")
            for domain, split, s, r, n in doc_rows:
                f.write(f"{domain}\t{split}\t{s}\t{r}\t{n}\n")

    (out_dir / "outputs.json").write_text(json.dumps(outputs_meta, indent=1) + "\n", encoding="utf-8")
    (out_dir / "match_stats.json").write_text(json.dumps(all_stats, indent=1) + "\n", encoding="utf-8")
    print(f"wrote sources.json ({len(src_files_flat)} files), outputs.json "
          f"({len(outputs_meta['outputs'])} outputs), documents-<source>.tsv.gz for {len(sources)} sources",
          flush=True)


# ------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-root", type=Path, default=Path("/scratch/users/nzhao2/refhq-new-v1"))
    ap.add_argument("--work-dir", type=Path, required=True, help="agent-runs work dir (writable)")
    ap.add_argument("--out-dir", type=Path, default=None, help="reduce phase output (manifest dir); default work-dir/manifest_out")
    sub = ap.add_subparsers(dest="phase", required=True)

    p = sub.add_parser("print-files")
    p.add_argument("--source", default=None)

    p = sub.add_parser("candidates")
    p.add_argument("--source", required=True)
    p.add_argument("--file-index", type=int, required=True)
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    p.add_argument("--batch-size", type=int, default=64)

    p = sub.add_parser("splitparts")
    p.add_argument("--source", required=True)
    p.add_argument("--domain", required=True)
    p.add_argument("--split", required=True)

    p = sub.add_parser("match")
    p.add_argument("--source", required=True)
    p.add_argument("--domain", required=True)

    p = sub.add_parser("reduce")
    p.add_argument("--sources", nargs="*", default=list(SOURCES))

    args = ap.parse_args()
    work = args.work_dir
    (work / "candidates").mkdir(parents=True, exist_ok=True)
    (work / "splitparts").mkdir(parents=True, exist_ok=True)
    (work / "match").mkdir(parents=True, exist_ok=True)
    cache = work / "hf-cache"
    cache.mkdir(parents=True, exist_ok=True)

    if args.phase == "print-files":
        srcs = [args.source] if args.source else list(SOURCES)
        for s in srcs:
            for i, entry in enumerate(source_files(s)):
                print(f"{s} {i} {entry['path']}")
        return 0

    if args.phase == "candidates":
        phase_candidates(args.source, args.file_index, cache, work / "candidates", args.workers, args.batch_size)
        return 0

    if args.phase == "splitparts":
        phase_splitparts(args.build_root, args.source, args.domain, args.split, work / "splitparts")
        return 0

    if args.phase == "match":
        match_domain(args.build_root, work, work / "match", args.source, args.domain)
        return 0

    if args.phase == "reduce":
        out_dir = args.out_dir or (work / "manifest_out")
        out_dir.mkdir(parents=True, exist_ok=True)
        phase_reduce(args.build_root, work, out_dir, args.sources, cache)
        return 0

    raise SystemExit(f"unknown phase {args.phase}")


if __name__ == "__main__":
    sys.exit(main())
