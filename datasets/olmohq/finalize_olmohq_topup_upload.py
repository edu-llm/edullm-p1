#!/usr/bin/env python3
"""Merge top-up shards into the local olmohq manifests without touching regmix-10b.

Local raw + tokenized top-up shards are already on scratch (no upload step);
this only merges plan manifests to include both old and new shards, backing up
the manifests it overwrites. Existing entries are left in place (append-only).
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    run_dir = args.run_dir

    topup_rows = [
        json.loads(l)
        for l in (run_dir / "plan/topup_manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    tok_index = [
        json.loads(l)
        for l in (run_dir / "plan/topup_tokenize_index.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]

    # Confirm the raw and tokenized top-up shards are already local.
    for row in topup_rows:
        local = run_dir / "data" / row["path"]
        if not local.is_file():
            raise SystemExit(f"missing local shard {local}")

    shard_reports = []
    for row in tok_index:
        npy = run_dir / "tokenized" / row["npy"]
        meta = npy.with_suffix(".json")
        if not npy.is_file():
            raise SystemExit(f"missing tokenized {npy}")
        if meta.is_file():
            tokens = int(json.loads(meta.read_text(encoding="utf-8")).get("tokens") or 0)
        else:
            tokens = npy.stat().st_size // 4
            meta.write_text(
                json.dumps({"tokens": tokens, "path": row["npy"]}, indent=2) + "\n",
                encoding="utf-8",
            )
        shard_reports.append(
            {
                "path": row["npy"],
                "manifest_path": row["manifest_path"],
                "domain": row["domain"],
                "tokens": tokens,
                "bytes": npy.stat().st_size,
                "topup": True,
            }
        )

    # Merge manifests.
    old_manifest = [
        json.loads(l)
        for l in (run_dir / "plan/manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    existing_paths = {r["path"] for r in old_manifest}
    merged_manifest = list(old_manifest)
    for row in topup_rows:
        if row["path"] in existing_paths:
            continue
        merged_manifest.append(
            {
                "path": row["path"],
                "size": row["size"],
                "domain": row["domain"],
                "est_tokens": row.get("est_tokens"),
                "topup": True,
            }
        )

    old_tok = json.loads((run_dir / "plan/tokenized_manifest.json").read_text(encoding="utf-8"))
    old_paths = {s["path"] for s in old_tok["shards"]}
    new_shards = list(old_tok["shards"])
    for s in shard_reports:
        if s["path"] not in old_paths:
            new_shards.append(s)
    total_tokens = sum(int(s.get("tokens") or 0) for s in new_shards)
    new_tok = dict(old_tok)
    new_tok["shards"] = new_shards
    new_tok["total_content_tokens"] = total_tokens
    new_tok["topup_appended_at"] = datetime.now(timezone.utc).isoformat()
    new_tok["topup_domains"] = sorted({s["domain"] for s in shard_reports})

    by_domain: dict[str, int] = {}
    for s in new_shards:
        by_domain[s["domain"]] = by_domain.get(s["domain"], 0) + int(s.get("tokens") or 0)

    (run_dir / "plan/manifest_merged.jsonl").write_text(
        "\n".join(json.dumps(r) for r in merged_manifest) + "\n", encoding="utf-8"
    )
    (run_dir / "plan/tokenized_manifest_merged.json").write_text(
        json.dumps(new_tok, indent=2) + "\n", encoding="utf-8"
    )
    availability = {
        "updated_at": new_tok["topup_appended_at"],
        "measured_tokens_by_domain": by_domain,
        "planned_available": {
            "dclm": 28_600_000_000,
            "arxiv": 20_800_000_000,
            "starcoder": 20_300_000_000,
            "pes2o": 26_300_000_000,
            "open-web-math": 12_200_000_000,
            "algebraic-stack": 11_800_000_000,
            "wiki": 3_660_000_000,
        },
    }
    for d, planned in availability["planned_available"].items():
        meas = by_domain.get(d, 0)
        err = abs(planned - meas) / meas if meas else None
        availability.setdefault("rel_err", {})[d] = err
        availability.setdefault("within_10pct", {})[d] = bool(err is not None and err <= 0.10)
    (run_dir / "plan/availability_after_topup.json").write_text(
        json.dumps(availability, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(availability, indent=2))

    if args.dry_run:
        print("dry-run: skipping manifest overwrite")
        return 0

    # Back up the manifests we're about to overwrite, then promote the merged ones.
    shutil.copy2(run_dir / "plan/manifest.jsonl", run_dir / "plan/manifest.pre_topup.jsonl")
    shutil.copy2(
        run_dir / "plan/tokenized_manifest.json",
        run_dir / "plan/tokenized_manifest.pre_topup.json",
    )
    shutil.copy2(run_dir / "plan/manifest_merged.jsonl", run_dir / "plan/manifest.jsonl")
    shutil.copy2(
        run_dir / "plan/tokenized_manifest_merged.json", run_dir / "plan/tokenized_manifest.json"
    )
    print("olmohq top-up merged locally (regmix-10b untouched)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
