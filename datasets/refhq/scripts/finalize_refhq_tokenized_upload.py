#!/usr/bin/env python3
"""Check dolma2 tokenized .npy memmaps and write the local tokenized manifest."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    scratch_root = Path(plan["scratch_root"])
    tokenized_root = scratch_root / "tokenized"

    reports: dict[str, dict] = {}
    failures: list[str] = []
    total_stream_tokens = 0

    for domain in plan["domains"]:
        meta_path = tokenized_root / domain / f"{domain}.json"
        npy_path = tokenized_root / domain / f"{domain}.npy"
        if not meta_path.is_file() or not npy_path.is_file():
            failures.append(f"{domain}: missing {npy_path} or {meta_path}")
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        stream_tokens = int(meta.get("stream_tokens_with_eos") or 0)
        reports[domain] = {
            **meta,
            "bytes": npy_path.stat().st_size,
            "within_expected": stream_tokens > 0,
        }
        total_stream_tokens += stream_tokens

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tokenizer_id": "allenai/dolma2-tokenizer",
        "eos_token_id": 100257,
        "total_stream_tokens_with_eos": total_stream_tokens,
        "domains": reports,
        "failures": failures,
        "accepted": not failures,
    }
    manifest_path = scratch_root / "manifests" / "tokenized_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)

    if failures:
        print(f"ACCEPTANCE FAILED: {len(failures)} domain(s)", flush=True)
        return 1

    print(f"accepted -> {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
