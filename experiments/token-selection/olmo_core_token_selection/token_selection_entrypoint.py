#!/usr/bin/env python3
"""The one entrypoint for every token-selection arm, on FarmShare 4xL40S.

Corpora are read from a local manifest (``ready.json``) staged onto FarmShare
scratch by ``farmshare/stage_local.py`` from the local corpus directories
pinned in ``datasets/manifests/<corpus>/outputs.json``. There is no RunPod
path, no S3, and no ``edullm_data`` anywhere in this tree.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

EDULLM_DIR = Path(__file__).resolve().parent
if str(EDULLM_DIR) not in sys.path:
    sys.path.insert(0, str(EDULLM_DIR))

from production_contract.checkpoint import assert_resume_fingerprint  # noqa: E402
from production_contract.wandb_artifacts import restore_checkpoint_artifact  # noqa: E402
from token_selection_370m.arms import ARM_SPECS, REFHQ_INSTRUCT, get_arm  # noqa: E402
from token_selection_370m.recipe import (  # noqa: E402
    build_trainer,
    immutable_corpus_binding,
    reference_digest,
    scientific_identity,
    total_steps,
    write_identity,
)


@dataclass(frozen=True)
class Corpus:
    dataset_id: str
    version: str
    paths: Sequence[str]
    dtype: "object"
    tokenizer: "object"
    rows: Optional[int]


def _manifest_path() -> Path:
    return Path(
        os.environ.get(
            "EDULLM_INPUT_MANIFEST",
            "/tmp/edullm-inputs/token-selection/ready.json",
        )
    )


def _manifest() -> dict:
    payload = json.loads(_manifest_path().read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2 or payload.get("family") != "token-selection":
        raise RuntimeError("invalid token-selection local input manifest")
    return payload


def resolve_corpus(*, dataset_id: str, version: str, tokenizer_id: str) -> Corpus:
    """Resolve one corpus from the staged local manifest.

    Every object's local path and size are checked; the manifest itself was
    built by ``stage_local.py`` directly from ``datasets/manifests/*/outputs.json``,
    so the per-file sha256 pinned there is the corpus's real reproducibility
    record. Re-hashing multi-gigabyte files on every launch would be slow for
    no benefit; ``stage_local.py`` verifies sha256 once, when it builds the
    staged copy.
    """
    from olmo_core.data import NumpyDatasetDType, TokenizerConfig

    if tokenizer_id != "tokenizer/dolma2-bpe":
        raise RuntimeError(f"unsupported tokenizer: {tokenizer_id}")
    record = _manifest()["corpora"].get(dataset_id)
    if record is None:
        raise RuntimeError(f"local manifest has no staged corpus for {dataset_id}")
    if version not in ("", "latest", record["version"]):
        raise RuntimeError(f"requested {dataset_id}/{version}, staged version is {record['version']}")
    paths = []
    for obj in record["objects"]:
        path = Path(obj["path"])
        if not path.is_file() or path.stat().st_size != int(obj["size"]):
            raise RuntimeError(f"staged object is missing or changed: {path}")
        paths.append(str(path))
    return Corpus(
        dataset_id=dataset_id,
        version=str(record["version"]),
        paths=paths,
        dtype=NumpyDatasetDType(record["dtype"]),
        tokenizer=TokenizerConfig.dolma2(),
        rows=int(record["rows"]) if record.get("rows") is not None else None,
    )


def resolve_reference_path(contract: Optional[str]) -> Optional[str]:
    """Resolve a symbolic reference contract to its materialized local .pt file."""
    if contract is None:
        return None
    record = _manifest().get("references", {}).get(contract)
    if record is None:
        raise RuntimeError(f"local manifest has no materialized reference for {contract!r}")
    path = Path(record)
    if not path.is_file():
        raise RuntimeError(f"materialized reference is missing: {path}")
    return str(path)


def resolve_reference_scores(contract: Optional[str]) -> tuple[Optional[str], Optional[dict]]:
    """Resolve a symbolic reference-scores contract to its table directory and manifest.

    Unlike ``resolve_reference_path`` (one flat .pt file), this points at a
    directory (see ``token_selection_370m/reference_scores.py``): a memmap
    plus a manifest binding it to the exact reference checkpoint and corpus it
    was scored from. The manifest is returned too, so the caller can fold its
    corpus binding and completeness into identity checks without a second
    resolve.
    """
    if contract is None:
        return None, None
    record = _manifest().get("reference_scores", {}).get(contract)
    if record is None:
        raise RuntimeError(
            f"local manifest has no materialized reference-score table for {contract!r}"
        )
    from token_selection_370m.reference_scores import read_manifest

    root = Path(record)
    manifest = read_manifest(root)
    if not manifest.get("complete"):
        raise RuntimeError(f"reference-score table for {contract!r} is not complete: {root}")
    return str(root), manifest


def runtime_environment() -> dict[str, object]:
    """Facts worth recording for the record but not part of the resume-blocking identity.

    A torch/CUDA/driver bump between a run and its resume shouldn't refuse the
    resume the way a change to the scientific identity itself does -- see
    ``recipe.write_identity``.
    """
    import subprocess
    import sys

    import torch

    info: dict[str, object] = {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "python_version": sys.version.split()[0],
    }
    try:
        info["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        info["gpu_name"] = None
    try:
        info["nvidia_driver_version"] = (
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
            .decode()
            .strip()
            .splitlines()[0]
        )
    except Exception:
        info["nvidia_driver_version"] = None
    try:
        freeze = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], stderr=subprocess.DEVNULL, timeout=60
        )
        info["pip_freeze_sha256"] = __import__("hashlib").sha256(freeze).hexdigest()
    except Exception:
        info["pip_freeze_sha256"] = None
    return info


def git_commit() -> Optional[str]:
    marker = EDULLM_DIR / "GIT_COMMIT"
    if marker.is_file():
        return marker.read_text(encoding="utf-8").strip() or None
    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(EDULLM_DIR), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def _path(name: str, default: str = "") -> str | None:
    value = os.environ.get(name, default).strip()
    return value or None


def _latest_resume_checkpoint(save_folder: Path) -> tuple[Path, bool]:
    """Return the newest normal or BLADE sync-boundary checkpoint.

    The boolean indicates a sync-boundary checkpoint, whose immutable run
    fingerprint lives at the save-folder root rather than inside the checkpoint.
    """

    explicit = os.environ.get("EDULLM_RESUME_CHECKPOINT", "").strip()
    if explicit:
        path = Path(explicit)
        if not (path / "model_and_optim" / ".metadata").is_file():
            raise SystemExit(
                f"EDULLM_RESUME_CHECKPOINT is not materialized: {path}"
            )
        return path, path.parent != save_folder

    candidates: list[tuple[int, int, Path, bool]] = []
    for path in save_folder.glob("step*"):
        if (
            not path.is_dir()
            or not path.name.removeprefix("step").isdigit()
            or not (path / "model_and_optim" / ".metadata").is_file()
        ):
            continue
        candidates.append((int(path.name.removeprefix("step")), 2, path, False))

    sync_root = save_folder / "sync_checkpoints"
    for path in (sync_root.glob("step*-*") if sync_root.is_dir() else ()):
        name = path.name.removeprefix("step")
        step_text, separator, phase = name.partition("-")
        if (
            path.is_dir()
            and separator
            and step_text.isdigit()
            and phase in {"pre", "post"}
            and (path / "model_and_optim" / ".metadata").is_file()
        ):
            # A completed normal step wins a tie. Otherwise post-sync wins pre-sync.
            phase_priority = 1 if phase == "post" else 0
            candidates.append((int(step_text), phase_priority, path, True))

    if not candidates:
        raise SystemExit("--resume found no local or restored checkpoint")
    _, _, path, is_sync_boundary = max(candidates)
    return path, is_sync_boundary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--arm", choices=tuple(ARM_SPECS), required=True)
    result.add_argument("--resume", action="store_true")
    result.add_argument("--local", action="store_true", help="Allow offline/local W&B behavior")
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--save-folder", type=Path, default=None)
    result.add_argument("--work-dir", type=Path, default=Path("/tmp/edullm-token-selection"))
    result.add_argument("--progress-dir", type=Path, default=None)
    result.add_argument("--task-loss-script", type=Path, default=None)
    return result


def assert_production_runtime(expected_world_size: int) -> None:
    """Fail before model construction unless torchrun supplied the locked GPU topology."""
    import torch

    from olmo_core.distributed.utils import get_world_size

    world_size = get_world_size()
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
    if world_size != expected_world_size or local_world_size != expected_world_size:
        raise RuntimeError(
            "production token-selection runs require one 4-GPU torchrun node "
            f"(WORLD_SIZE={world_size}, LOCAL_WORLD_SIZE={local_world_size})"
        )
    visible_devices = torch.cuda.device_count()
    if visible_devices < local_world_size:
        raise RuntimeError(
            f"torchrun started {local_world_size} local ranks but only "
            f"{visible_devices} CUDA devices are visible"
        )


def main() -> None:
    from token_selection_370m.recipe import PRODUCTION_WORLD_SIZE

    args = parser().parse_args()
    # torchrun's own argparse (parse_args, not parse_known_args) scans the
    # entire argv for abbreviation matches against its own flags regardless of
    # position, and "--local" ambiguously matches its --local-addr /
    # --local-ranks-filter, so torchrun refuses to start if it's passed on the
    # launcher command line. Callers that need --local under torchrun (e.g. a
    # smoke test at a GPU count other than the locked production topology)
    # should set EDULLM_LOCAL=1 instead.
    if os.environ.get("EDULLM_LOCAL") == "1":
        args.local = True
    arm = get_arm(args.arm)
    save_folder = args.save_folder or Path(
        os.environ.get("EDULLM_CHECKPOINT_DIR", f"/tmp/checkpoints/{arm.run_id}")
    )
    progress_dir = args.progress_dir or Path(
        os.environ.get("EDULLM_PROGRESS_DIR", f"/tmp/progress/{arm.run_id}")
    )
    task_loss_script = args.task_loss_script or Path(os.environ.get("TASK_LOSS_EVAL_SCRIPT", ""))
    if not args.local and not task_loss_script.is_file():
        raise SystemExit(
            "production run requires TASK_LOSS_EVAL_SCRIPT for synchronous 20-label evaluation"
        )

    version = os.environ.get("EDULLM_DATASET_VERSION", "latest")
    corpus = resolve_corpus(
        dataset_id=arm.dataset_id,
        version=version,
        tokenizer_id="tokenizer/dolma2-bpe",
    )
    max_tokens = int(arm.max_tokens if arm.max_tokens is not None else (corpus.rows or 0))
    if max_tokens <= 0:
        raise SystemExit("reference corpus manifest must declare a positive row/token count")
    reference_path = resolve_reference_path(arm.reference_contract)
    reference_scores_path, reference_scores_manifest = resolve_reference_scores(
        arm.reference_scores_contract
    )

    refhq_corpus = None
    if arm.requires_refhq_stream:
        refhq_version = os.environ.get("EDULLM_REFHQ_DATASET_VERSION", "latest")
        refhq_corpus = resolve_corpus(
            dataset_id=REFHQ_INSTRUCT,
            version=refhq_version,
            tokenizer_id="tokenizer/dolma2-bpe",
        )
    commit = git_commit()
    identity = scientific_identity(
        arm,
        dataset_binding=immutable_corpus_binding(arm.dataset_id, corpus),
        refhq_binding=(
            immutable_corpus_binding(REFHQ_INSTRUCT, refhq_corpus)
            if refhq_corpus is not None
            else None
        ),
        max_tokens=max_tokens,
        reference_path=reference_path,
        reference_scores_manifest=reference_scores_manifest,
        git_commit=commit,
    )
    print(
        json.dumps(
            {
                "arm": arm.name,
                "method": arm.method,
                "dataset": f"{arm.dataset_id}/{corpus.version}",
                "run_id": arm.run_id,
                "wandb_project": arm.wandb_project,
                "max_tokens": max_tokens,
                "total_steps": total_steps(max_tokens),
                "git_commit": commit,
            },
            indent=2,
        ),
        flush=True,
    )
    if args.dry_run:
        return

    save_folder.mkdir(parents=True, exist_ok=True)
    resume_checkpoint: Path | None = None
    if args.resume:
        artifact = os.environ.get("WANDB_RESUME_ARTIFACT", "").strip()
        if artifact and not any(save_folder.glob("step*")):
            restore_checkpoint_artifact(artifact, save_folder)
        resume_checkpoint, is_sync_boundary = _latest_resume_checkpoint(save_folder)
        assert_resume_fingerprint(
            save_folder if is_sync_boundary else resume_checkpoint,
            identity,
        )
        print(f"Resuming from checkpoint: {resume_checkpoint}", flush=True)
    elif any(save_folder.iterdir()):
        raise SystemExit(f"fresh run refuses non-empty save folder: {save_folder}")

    from olmo_core.train import prepare_training_environment, teardown_training_environment
    from olmo_core.utils import seed_all

    prepare_training_environment(seed=6198)
    try:
        if not args.local:
            assert_production_runtime(PRODUCTION_WORLD_SIZE)
        torch_imported = __import__("torch")
        torch_imported.set_float32_matmul_precision("high")
        seed_all(6198)
        environment = runtime_environment()
        trainer = build_trainer(
            arm,
            corpus,
            refhq_corpus=refhq_corpus,
            max_tokens=max_tokens,
            save_folder=save_folder,
            work_dir=args.work_dir / arm.name,
            progress_dir=progress_dir,
            task_loss_script=task_loss_script,
            reference_path=reference_path,
            reference_scores_path=reference_scores_path,
            reference_scores_reference_sha256=reference_digest(reference_path),
            reference_scores_corpus_binding=identity["dataset_binding"],
            resume=args.resume,
            production=not args.local,
            git_commit=commit,
            environment=environment,
        )
        write_identity(save_folder, progress_dir, identity, environment=environment)
        if resume_checkpoint is not None:
            trainer.load_checkpoint(
                resume_checkpoint,
                load_trainer_state=True,
                load_optim_state=True,
            )
        trainer.fit()
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    main()
