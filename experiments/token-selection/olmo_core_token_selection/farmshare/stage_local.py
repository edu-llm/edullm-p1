#!/usr/bin/env python3
"""Build the local, no-S3 input manifest every token-selection arm reads from.

Two modes, run in order:

  --mode corpora     Bind the three pinned FarmShare corpus directories
                      (verified byte-identical to this repository's
                      datasets/manifests/*/outputs.json on 2026-09-25) into
                      ready.json. No copying: these directories are already
                      on the same shared filesystem every training job runs
                      on, so ready.json just points at them directly.

  --mode references   Materialize the HQ and Instruct reference checkpoints
                      (trained by the hq-reference/instruct-reference arms
                      in this same study) into flat .pt files, average the
                      HQ ones, and add them to ready.json under their
                      symbolic reference contract names. Run this only after
                      those two arms' training has finished.

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
    HQ_REFERENCE_AVERAGED_STEPS,
    HQ_REFERENCE_CONTRACT,
    INSTRUCT_REFERENCE_CONTRACT,
    REFHQ_5P5B,
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
    REFHQ_5P5B: {
        "version": "v1",
        "root": "/scratch/users/nzhao2/refhq-regmix-5p5b-v1/tokenized",
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


def _average_checkpoints(materialized: list[Path], output: Path) -> None:
    """Float32-average several materialized .pt checkpoints (HQ steps 1000/1125/1315)."""
    import torch

    def model_state(path: Path) -> dict:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload.get("model") if isinstance(payload, dict) else None
        if not isinstance(state, dict) or not state:
            raise RuntimeError(f"materialized checkpoint has no model state: {path}")
        return state

    states = [model_state(path) for path in materialized]
    keys = set(states[0])
    if any(set(state) != keys for state in states[1:]):
        raise RuntimeError("HQ reference checkpoints have different parameter keys")
    averaged = {}
    for key, first in states[0].items():
        if first.is_floating_point():
            accumulator = first.detach().to(torch.float32).clone()
            for state in states[1:]:
                other = state[key]
                if other.shape != first.shape:
                    raise RuntimeError(f"HQ reference shape mismatch for {key}")
                accumulator.add_(other.detach().to(torch.float32))
            averaged[key] = (accumulator / len(states)).to(first.dtype)
        else:
            averaged[key] = first.detach().clone()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".pt.partial")
    torch.save({"model": averaged, "steps": list(HQ_REFERENCE_AVERAGED_STEPS)}, temporary)
    temporary.replace(output)


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

    hq_materialized = []
    for step in HQ_REFERENCE_AVERAGED_STEPS:
        checkpoint = run_root / "hq-reference" / "checkpoints" / f"step{step}"
        if not checkpoint.is_dir():
            raise RuntimeError(f"hq-reference has not produced step{step} yet: {checkpoint}")
        hq_materialized.append(materialize_model_eval(checkpoint))
    hq_averaged = manifest_path.parent / "references" / "hq_reference_average_1000_1125_1315.pt"
    _average_checkpoints(hq_materialized, hq_averaged)
    payload["references"][HQ_REFERENCE_CONTRACT] = str(hq_averaged)
    print(f"averaged HQ reference: {hq_averaged}")

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
