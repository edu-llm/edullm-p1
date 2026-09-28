#!/usr/bin/env python3
"""Build the local, no-S3 input manifest every token-selection arm reads from.

Two modes, run in order:

  --mode corpora     Bind the three pinned FarmShare corpus directories
                      (verified byte-identical to this repository's
                      datasets/manifests/*/outputs.json on 2026-09-25) into
                      ready.json. No copying: these directories are already
                      on the same shared filesystem every training job runs
                      on, so ready.json just points at them directly.

  --mode references   Materialize the Instruct reference checkpoint (trained
                      by the instruct-reference arm in this same study) into
                      a flat .pt file and add it to ready.json under its
                      symbolic reference contract name. Run this only after
                      that arm's training has finished.

Every path here is local; there is no boto3, no S3 URI, and no edullm_data
import in this file or anywhere else in this tree.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FARMSHARE_DIR = Path(__file__).resolve().parent
EDULLM_DIR = FARMSHARE_DIR.parent
if str(EDULLM_DIR) not in sys.path:
    sys.path.insert(0, str(EDULLM_DIR))

from token_selection_370m.arms import (  # noqa: E402
    INSTRUCT_REFERENCE_CONTRACT,
    REFHQ_INSTRUCT,
    REGMIX,
)

# Pinned FarmShare source directories. See the plan's "Pinned training data"
# table: every file here was verified byte-identical to
# datasets/manifests/<corpus>/outputs.json on 2026-09-25, and these
# directories were made read-only before any training started.
CORPUS_SOURCES: dict[str, dict] = {
    REGMIX: {
        "version": "v1",
        "root": "/scratch/users/nzhao2/agent-runs/regmix-10b-20260725-124810/tokenized",
        "files": [
            "dclm/dclm.npy",
            "arxiv/arxiv.npy",
            "starcoder/starcoder.npy",
            "pes2o/pes2o.npy",
            "open-web-math/open-web-math.npy",
            "algebraic-stack/algebraic-stack.npy",
            "wiki/wiki.npy",
        ],
    },
    REFHQ_INSTRUCT: {
        "version": "v3",
        "root": "/scratch/users/nzhao2/refhq-new-v1/tokenized",
        # Two levels deep: <source>/<domain>/{train,val}.npy. Only the train
        # split is trained on; val.npy files are deliberately excluded.
        "glob": "*/*/train.npy",
    },
}

MANIFEST_SCHEMA_VERSION = 2


def _empty_manifest() -> dict:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "family": "token-selection",
        "corpora": {},
        "references": {},
    }


def _load_manifest(path: Path) -> dict:
    if not path.is_file():
        return _empty_manifest()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise SystemExit(f"{path} has an unexpected schema_version; refusing to extend it")
    return payload


def _write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def stage_corpus(dataset_id: str, cfg: dict) -> dict:
    root = Path(cfg["root"])
    if "files" in cfg:
        paths = [root / name for name in cfg["files"]]
    else:
        paths = sorted(root.glob(cfg["glob"]))
    if not paths:
        raise RuntimeError(f"no source files found for {dataset_id} under {root}")
    objects = []
    total_bytes = 0
    for path in paths:
        if not path.is_file():
            raise RuntimeError(f"pinned source file for {dataset_id} is missing: {path}")
        size = path.stat().st_size
        objects.append({"path": str(path), "size": size})
        total_bytes += size
    if total_bytes % 4 != 0:
        raise RuntimeError(f"{dataset_id}'s total byte count is not uint32-aligned: {total_bytes}")
    return {
        "version": cfg["version"],
        "dtype": "uint32",
        "rows": total_bytes // 4,
        "objects": objects,
    }


def stage_corpora(manifest_path: Path) -> None:
    payload = _load_manifest(manifest_path)
    for dataset_id, cfg in CORPUS_SOURCES.items():
        record = stage_corpus(dataset_id, cfg)
        payload["corpora"][dataset_id] = record
        print(
            f"staged {dataset_id}/{record['version']}: {len(record['objects'])} files, "
            f"{record['rows']} tokens"
        )
    _write_manifest(manifest_path, payload)
    print(f"wrote {manifest_path}")


def stage_references(manifest_path: Path, run_root: Path) -> None:
    from eval_task_loss_olmo_core import materialize_model_eval

    payload = _load_manifest(manifest_path)

    instruct_checkpoint = run_root / "instruct-reference" / "checkpoints" / "step940"
    if not instruct_checkpoint.is_dir():
        raise RuntimeError(
            f"instruct-reference has not produced step940 yet: {instruct_checkpoint}"
        )
    instruct_flat = materialize_model_eval(instruct_checkpoint)
    payload["references"][INSTRUCT_REFERENCE_CONTRACT] = str(instruct_flat)
    print(f"materialized instruct reference: {instruct_flat}")

    _write_manifest(manifest_path, payload)
    print(f"wrote {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("corpora", "references"), required=True)
    parser.add_argument(
        "--run-root",
        type=Path,
        help="Root of runs/<arm>/checkpoints; required for --mode references",
    )
    args = parser.parse_args()
    manifest_path = args.stage_root / "ready.json"
    if args.mode == "corpora":
        stage_corpora(manifest_path)
    else:
        if args.run_root is None:
            raise SystemExit("--mode references requires --run-root")
        stage_references(manifest_path, args.run_root)


if __name__ == "__main__":
    main()
