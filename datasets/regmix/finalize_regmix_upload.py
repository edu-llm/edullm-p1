#!/usr/bin/env python3
"""Finalize RegMix 10B mix: build the final manifest and a local staging layout."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    run = args.run_dir
    summary = json.loads((run / "plan" / "summary.json").read_text(encoding="utf-8"))
    domain_order = list(summary["domain_targets"].keys())

    new_manifest: list[dict] = []
    trim_meta: list[dict] = []
    tokenized_files: list[dict] = []
    for domain in domain_order:
        result_path = run / "trim" / domain / "trim_result.json"
        if not result_path.exists():
            raise SystemExit(f"missing trim result for {domain}: {result_path}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        local = Path(result["output_shard"])
        if not local.exists():
            raise SystemExit(f"missing trimmed shard {local}")
        npy = Path(result.get("tokenized_npy") or run / "tokenized" / domain / f"{domain}.npy")
        npy_meta = Path(result.get("tokenized_meta") or run / "tokenized" / domain / f"{domain}.json")
        if not npy.exists():
            raise SystemExit(f"missing tokenized npy for {domain}: {npy}")
        if not npy_meta.exists():
            raise SystemExit(f"missing tokenized meta for {domain}: {npy_meta}")
        rel = f"data/{domain}/{domain}-regmix.json.gz"
        tok_rel = f"tokenized/{domain}/{domain}.npy"
        tok_meta_rel = f"tokenized/{domain}/{domain}.json"
        new_manifest.append(
            {
                "path": rel,
                "size": local.stat().st_size,
                "domain": domain,
                "est_tokens": result["tokens_after"],
                "measured_tokens": result["tokens_after"],
                "tokens_with_eos": result.get("tokens_with_eos"),
                "target_tokens": result["target_tokens"],
                "relative_error": result.get("relative_error"),
                "docs_after": result["docs_after"],
                "tokenizer": result["tokenizer"],
                "tokenized_npy": tok_rel,
                "local_path": str(local),
                "local_npy": str(npy),
                "local_npy_meta": str(npy_meta),
            }
        )
        tokenized_files.append(
            {"path": tok_rel, "local": str(npy), "size": npy.stat().st_size}
        )
        tokenized_files.append(
            {"path": tok_meta_rel, "local": str(npy_meta), "size": npy_meta.stat().st_size}
        )
        summary["domains"][domain]["measured_tokens"] = result["tokens_after"]
        summary["domains"][domain]["tokens_with_eos"] = result.get("tokens_with_eos")
        summary["domains"][domain]["docs_after"] = result["docs_after"]
        summary["domains"][domain]["relative_error"] = result.get("relative_error")
        summary["domains"][domain]["output_shard"] = rel
        summary["domains"][domain]["tokenized_npy"] = tok_rel
        summary["domains"][domain]["bytes_final"] = local.stat().st_size
        summary["domains"][domain]["tokenized_bytes"] = npy.stat().st_size
        trim_meta.append(result)

    new_manifest.sort(key=lambda x: x["path"])
    for i, row in enumerate(new_manifest):
        row["index"] = i

    total_tokens = sum(m["measured_tokens"] for m in new_manifest)
    summary["kind"] = "regmix-optimized-10b-final"
    summary["tokenizer"] = "allenai/dolma2-tokenizer"
    summary["eos_token_id"] = 100257
    summary["files_final"] = len(new_manifest)
    summary["bytes_final"] = sum(m["size"] for m in new_manifest)
    summary["tokenized_bytes"] = sum(
        f["size"] for f in tokenized_files if f["path"].endswith(".npy")
    )
    summary["measured_tokens_total"] = total_tokens
    summary["trim"] = {
        "method": "document-shuffle-then-keep-to-target",
        "seed": 42,
        "tokenizer": "allenai/dolma2-tokenizer",
        "results": trim_meta,
    }

    plan_dir = run / "plan"
    final_manifest = plan_dir / "manifest_final.jsonl"
    with final_manifest.open("w", encoding="utf-8") as fh:
        for row in new_manifest:
            out = {
                k: v
                for k, v in row.items()
                if not k.startswith("local_")
            }
            fh.write(json.dumps(out) + "\n")
    (plan_dir / "summary_final.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (plan_dir / "trim_results.json").write_text(
        json.dumps(trim_meta, indent=2), encoding="utf-8"
    )

    readme = f"""# RegMix-optimized OLMo-mix 10B

- Source: {summary.get('source', 'the local olmohq run')}
- Method: random whole-shard sample from source, then document-shuffle trim to RegMix-mapped budgets
- Seed: {summary.get('seed', 42)}
- Tokenizer: allenai/dolma2-tokenizer (OLMo-2 canonical; EOS=100257)
- Text shards: data/<domain>/<domain>-regmix.json.gz
- Tokenized: tokenized/<domain>/<domain>.npy (uint32 memmap) + .json metadata
- Measured total content tokens: {total_tokens:,.0f}

## Domain budgets

| Domain | Weight | Target | Measured |
|--------|--------|--------|----------|
"""
    for domain in domain_order:
        d = summary["domains"][domain]
        readme += (
            f"| {domain} | {d['weight']:.4f} | {d['target_tokens']:,} | "
            f"{d.get('measured_tokens', 'n/a'):,} |\n"
        )
    readme += "\nSee plan/summary_final.json for full accounting.\n"
    (run / "README.md").write_text(readme, encoding="utf-8")

    staging = run / "staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    shutil.copy2(run / "README.md", staging / "README.md")
    (staging / "plan").mkdir()
    shutil.copy2(final_manifest, staging / "plan" / "manifest.jsonl")
    shutil.copy2(plan_dir / "summary_final.json", staging / "plan" / "summary.json")
    shutil.copy2(plan_dir / "trim_results.json", staging / "plan" / "trim_results.json")
    shutil.copy2(plan_dir / "domain_targets.json", staging / "plan" / "domain_targets.json")

    for row in new_manifest:
        dest = staging / row["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row["local_path"], dest)
        npy_dest = staging / row["tokenized_npy"]
        npy_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row["local_npy"], npy_dest)
        meta_dest = staging / Path(row["tokenized_npy"]).with_suffix(".json")
        shutil.copy2(row["local_npy_meta"], meta_dest)

    # Convenience paths list for training loaders.
    paths_txt = staging / "tokenized" / "paths.txt"
    paths_txt.parent.mkdir(parents=True, exist_ok=True)
    paths_txt.write_text(
        "\n".join(f["path"] for f in tokenized_files if f["path"].endswith(".npy")) + "\n",
        encoding="utf-8",
    )

    print(f"staging ready at {staging}", flush=True)
    print(json.dumps({"measured_tokens_total": total_tokens, "bytes_final": summary["bytes_final"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
