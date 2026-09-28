"""The single OLMo2-370M recipe; arm selection changes loaders and loss policy only."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from production_contract.checkpoint import (
    checkpointer_kwargs_for_ladder,
    make_run_fingerprint,
    write_run_fingerprint,
)

from .arms import ArmSpec

SEQUENCE_LENGTH = 2048
GLOBAL_BATCH_TOKENS = 4_194_304
PEAK_LR = 4e-4
WARMUP_STEPS = 24
ALPHA_F = 0.1
Z_LOSS = 1e-5
MAX_GRAD_NORM = 1.0
# Every reported arm runs on one FarmShare 4xL40S node; there is no other
# hardware contract and no RunPod path in this tree.
PRODUCTION_WORLD_SIZE = 4


def total_steps(max_tokens: int) -> int:
    return int(max_tokens) // GLOBAL_BATCH_TOKENS


def reference_digest(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"materialized reference does not exist: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def immutable_corpus_binding(dataset_id: str, corpus: Any) -> dict[str, Any]:
    """Compactly bind a resolved, seal-verified corpus into checkpoint identity."""
    paths = [str(path) for path in corpus.paths]
    version = str(corpus.version)
    if not version or version == "latest":
        raise ValueError(f"{dataset_id} did not resolve to an immutable version")
    dtype = getattr(corpus.dtype, "value", corpus.dtype)
    return {
        "dataset_id": dataset_id,
        "version": version,
        "path_count": len(paths),
        "paths_sha256": hashlib.sha256(
            json.dumps(paths, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        ).hexdigest(),
        "dtype": str(dtype),
        "rows": None if corpus.rows is None else int(corpus.rows),
    }


_reference_digest = reference_digest  # backward-compat alias; prefer the public name


def _manifest_digest(payload: Optional[Mapping[str, Any]]) -> Optional[str]:
    """sha256 of a canonical (sorted-key) JSON encoding of a small manifest dict.

    Used for the reference-score table: hashing the whole 40 GB table on every
    launch would be slow for no benefit, but the manifest already contains a
    sha256 per written chunk, so hashing just the manifest still detects any
    change to the underlying data.
    """
    if payload is None:
        return None
    return hashlib.sha256(
        json.dumps(dict(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def scientific_identity(
    arm: ArmSpec,
    *,
    dataset_binding: Mapping[str, Any],
    refhq_binding: Optional[Mapping[str, Any]],
    max_tokens: int,
    reference_path: Optional[str],
    reference_scores_manifest: Optional[Mapping[str, Any]] = None,
    git_commit: Optional[str] = None,
) -> dict[str, Any]:
    main_binding = dict(dataset_binding)
    if main_binding.get("dataset_id") != arm.dataset_id:
        raise ValueError("resolved main corpus does not match the selected arm")
    if arm.requires_refhq_stream != (refhq_binding is not None):
        raise ValueError("BLADE RefHQ binding is missing or attached to a non-BLADE arm")
    if (arm.reference_scores_contract is not None) != (reference_scores_manifest is not None):
        raise ValueError(
            "reference-scores manifest is missing or attached to an arm that doesn't use one"
        )
    return {
        "arm": arm.name,
        "run_id": arm.run_id,
        "method": arm.method,
        "dataset_id": arm.dataset_id,
        "dataset_version": main_binding["version"],
        "dataset_binding": main_binding,
        "refhq_binding": dict(refhq_binding) if refhq_binding is not None else None,
        "tokenizer": "tokenizer/dolma2-bpe",
        "model": "TransformerConfig.olmo2_370M",
        "init_seed": arm.init_seed,
        "data_seed": arm.data_seed,
        "sequence_length": SEQUENCE_LENGTH,
        "global_batch_tokens": GLOBAL_BATCH_TOKENS,
        "rank_microbatch_tokens": arm.rank_microbatch_tokens,
        "max_tokens": int(max_tokens),
        "total_steps": total_steps(max_tokens),
        "peak_lr": PEAK_LR,
        "warmup_steps": WARMUP_STEPS,
        "alpha_f": ALPHA_F,
        "z_loss_multiplier": Z_LOSS,
        "keep_fraction": arm.keep_fraction,
        "ema_seed": arm.ema_seed,
        "ema_tau": arm.ema_tau,
        "reference_contract": arm.reference_contract,
        "reference_sha256": reference_digest(reference_path),
        "reference_scores_contract": arm.reference_scores_contract,
        "reference_scores_sha256": _manifest_digest(reference_scores_manifest),
        "wandb_project": arm.wandb_project,
        "checkpoint_contract": "schema-v3-unified-farmshare-4xl40s",
        "git_commit": git_commit,
    }


def _loader(
    corpus,
    *,
    work_dir: Path,
    seed: int,
    process_group=None,
):
    from olmo_core.data import NumpyDataLoaderConfig, NumpyFSLDatasetConfig

    dataset = NumpyFSLDatasetConfig(
        paths=list(corpus.paths),
        sequence_length=SEQUENCE_LENGTH,
        tokenizer=corpus.tokenizer,
        dtype=corpus.dtype,
        work_dir=str(work_dir),
    ).build()
    return NumpyDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_TOKENS,
        seed=seed,
        num_workers=int(os.environ.get("EDULLM_NUM_WORKERS", "8")),
        num_threads=int(os.environ.get("EDULLM_NUM_THREADS", "8")),
        prefetch_factor=int(os.environ.get("EDULLM_PREFETCH_FACTOR", "4")),
    ).build(dataset, dp_process_group=process_group)


def _train_module_config(rank_microbatch_tokens: int):
    from olmo_core.config import DType
    from olmo_core.distributed.parallel import DataParallelType
    from olmo_core.distributed.utils import is_distributed
    from olmo_core.optim import CosWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
    from olmo_core.train.train_module import (
        TransformerDataParallelConfig,
        TransformerTrainModuleConfig,
    )

    dp_config = (
        TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
        )
        if is_distributed()
        else None
    )
    return TransformerTrainModuleConfig(
        rank_microbatch_size=rank_microbatch_tokens,
        max_sequence_length=SEQUENCE_LENGTH,
        optim=SkipStepAdamWConfig(
            lr=PEAK_LR,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts={"weight_decay": 0.0})
            ],
        ),
        scheduler=CosWithWarmup(warmup=WARMUP_STEPS, alpha_f=ALPHA_F),
        compile_model=True,
        dp_config=dp_config,
        z_loss_multiplier=Z_LOSS,
        max_grad_norm=MAX_GRAD_NORM,
        state_dict_save_opts={
            "full_state_dict": False,
            "cpu_offload": True,
            "flatten_optimizer_state_dict": True,
        },
        state_dict_load_opts={
            "full_state_dict": False,
            "strict": True,
            "flatten_optimizer_state_dict": True,
        },
    )


def _custom_module(model, config, selection):
    from torch.distributed.checkpoint.state_dict import StateDictOptions

    from .train_module import TokenWeightedTrainModule

    return TokenWeightedTrainModule(
        model=model,
        optim=config.optim,
        rank_microbatch_size=config.rank_microbatch_size,
        max_sequence_length=config.max_sequence_length,
        compile_model=config.compile_model,
        dp_config=config.dp_config,
        z_loss_multiplier=config.z_loss_multiplier,
        max_grad_norm=config.max_grad_norm,
        scheduler=config.scheduler,
        label_ignore_index=config.label_ignore_index,
        state_dict_save_opts=StateDictOptions(**(config.state_dict_save_opts or {})),
        state_dict_load_opts=StateDictOptions(**(config.state_dict_load_opts or {})),
        selection_config=selection,
    )


def build_trainer(
    arm: ArmSpec,
    corpus,
    *,
    refhq_corpus=None,
    max_tokens: int,
    save_folder: Path,
    work_dir: Path,
    progress_dir: Path,
    task_loss_script: Path,
    reference_path: Optional[str] = None,
    reference_scores_path: Optional[str] = None,
    reference_scores_reference_sha256: Optional[str] = None,
    reference_scores_corpus_binding: Optional[Mapping[str, Any]] = None,
    resume: bool = False,
    production: bool = True,
    git_commit: Optional[str] = None,
    environment: Optional[Mapping[str, Any]] = None,
):
    from olmo_core.nn.transformer import TransformerConfig
    from olmo_core.train import Duration, LoadStrategy, TrainerConfig
    from olmo_core.train.callbacks import (
        CheckpointerCallback,
        ConfigSaverCallback,
        GPUMemoryMonitorCallback,
        WandBCallback,
    )

    from production_contract.task_loss import TaskLossEvalCallback

    from .blade import BladeCallback, ResumableBatchStream
    from .train_module import TokenSelectionConfig, TokenSelectionStateCallback

    steps = total_steps(max_tokens)
    model_config = TransformerConfig.olmo2_370M(
        vocab_size=corpus.tokenizer.padded_vocab_size(),
        init_seed=arm.init_seed,
    )
    model = model_config.build(init_device="meta")
    module_config = _train_module_config(arm.rank_microbatch_tokens)
    selection_config = TokenSelectionConfig(
        method=arm.method,
        keep_fraction=arm.keep_fraction,
        total_steps=steps,
        seed=arm.data_seed,
        reference_scores_path=reference_scores_path,
        reference_scores_reference_sha256=reference_scores_reference_sha256,
        reference_scores_corpus_binding=reference_scores_corpus_binding,
        ema_seed=arm.ema_seed,
        ema_tau=arm.ema_tau,
    )
    # Every arm, including full-loss and BLADE, runs through the same custom
    # train module. BLADE's per-token weights come from batch["token_weight"]
    # (set by BladeCallback.pre_step), read before selection_weights() is ever
    # called; "full"'s weights are just every valid token.
    train_module = _custom_module(model, module_config, selection_config)

    main_loader = _loader(
        corpus,
        work_dir=work_dir / "main",
        seed=arm.data_seed,
        process_group=train_module.dp_process_group,
    )
    if main_loader.total_batches is not None and steps > main_loader.total_batches:
        raise ValueError(
            f"{arm.name} would need {steps} steps but its corpus epoch is only "
            f"{main_loader.total_batches} batches; max_tokens must not exceed one epoch"
        )
    checkpoint_kwargs = checkpointer_kwargs_for_ladder(steps, 125, save_async=False)
    checkpoint_kwargs["pre_train_checkpoint"] = not resume
    # TaskLossEvalCallback finalizes in post_step, before CheckpointerCallback's
    # post_train fallback. Save the true final step in post_train_batch so the
    # synchronous evaluator never waits for a checkpoint that cannot exist yet.
    checkpoint_kwargs["fixed_steps"] = [
        *checkpoint_kwargs["fixed_steps"],
        steps,
    ]
    trainer_config = (
        TrainerConfig(
            save_folder=str(save_folder),
            # TrainerConfig.work_dir defaults to save_folder itself when left
            # unset, which lands WandBCallback's local run dir
            # (work_dir/"wandb") inside the checkpoint tree. W&B's local
            # file-watcher then sweeps up the actual optimizer-state shard
            # files as run files to sync/cache, ballooning local disk usage
            # by tens of GB per checkpoint. Keep it in the separate work_dir
            # this recipe already threads through to the data loader.
            work_dir=str(work_dir),
            load_strategy=LoadStrategy.if_available if resume else LoadStrategy.never,
            load_trainer_state=resume,
            load_optim_state=resume,
            # Duration.tokens(max_tokens) would run ceil(max_tokens / batch)
            # steps (2361, not 2360, for the 9.9e9-token budget) -- one step
            # past every ladder/eval/BLADE-schedule/FLOP computation in this
            # file, which all use total_steps()'s floor. Duration.steps make
            # the trainer stop at exactly the step every other computation
            # already assumes is the true final one.
            max_duration=Duration.steps(steps),
        )
        .with_callback("checkpointer", CheckpointerCallback(**checkpoint_kwargs))
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "wandb",
            WandBCallback(
                name=arm.run_id,
                project=arm.wandb_project,
                group=arm.name,
                tags=["token-selection", arm.name, arm.method],
                enabled=production or bool(os.environ.get("WANDB_API_KEY")),
                config={
                    "arm": arm.name,
                    "method": arm.method,
                    "dataset_id": arm.dataset_id,
                    "max_tokens": max_tokens,
                    "git_commit": git_commit,
                    "environment": dict(environment) if environment is not None else None,
                },
            ),
        )
    )
    if production or task_loss_script.is_file():
        trainer_config = trainer_config.with_callback(
            "task_loss",
            TaskLossEvalCallback(
                total_steps=steps,
                save_folder=save_folder,
                run_name=arm.run_id,
                results_dir=progress_dir / "task_loss",
                eval_script=task_loss_script,
                arm=arm.name,
                progress_dir=progress_dir,
                method=arm.method,
                task_loss_nproc=PRODUCTION_WORLD_SIZE if production else None,
                production=production,
                wandb_mode=os.environ.get("WANDB_MODE", "online"),
                # Every reported run keeps its full checkpoint ladder (roughly
                # 1.3 TB total across all nine runs; scratch has room), so a
                # checkpoint step can be re-evaluated later (a new eval suite,
                # per-label analysis, a clean-subset contamination check)
                # without rerunning training to reach it.
                keep_all_checkpoints=True,
            ),
        )
    # TokenSelectionStateCallback (priority 3) must advance the EMA/step
    # counter before BladeCallback's own sync bookkeeping (priority 4) runs at
    # the same post_train_batch boundary, so it is always attached: BLADE's
    # arm doesn't use EMA, but this ordering only matters relative to BLADE's
    # own callback, and attaching it unconditionally keeps every arm's
    # TokenWeightedTrainModule.selection_state advancing the same way.
    trainer_config = trainer_config.with_callback(
        "token_selection_state", TokenSelectionStateCallback()
    )
    if arm.method == "blade":
        from olmo_core.optim import CosWithWarmup

        if refhq_corpus is None:
            raise ValueError("BLADE requires the immutable RefHQ corpus as its second stream")
        reference_train_loader = _loader(
            corpus,
            work_dir=work_dir / "blade-reference-train",
            seed=arm.data_seed + 17,
            process_group=train_module.dp_process_group,
        )
        refhq_loader = _loader(
            refhq_corpus,
            work_dir=work_dir / "blade-refhq",
            seed=arm.data_seed + 101,
            process_group=train_module.dp_process_group,
        )

        def reference_factory():
            reference = model_config.build(init_device=str(train_module.device))
            reference.eval()
            for parameter in reference.parameters():
                parameter.requires_grad_(False)
            return reference

        trainer_config = trainer_config.with_callback(
            "blade_state",
            BladeCallback(
                total_steps=steps,
                reference_factory=reference_factory,
                reference_train_stream=ResumableBatchStream(reference_train_loader),
                refhq_stream=ResumableBatchStream(refhq_loader),
                reference_scheduler=CosWithWarmup(warmup=WARMUP_STEPS, alpha_f=ALPHA_F),
                reference_initial_lr=PEAK_LR,
                reference_warmup_steps=WARMUP_STEPS,
                data_seed=arm.data_seed,
                reference_microbatch_tokens=arm.rank_microbatch_tokens,
                selection_microbatch_tokens=arm.rank_microbatch_tokens,
            ),
        )

    trainer = trainer_config.build(train_module, main_loader)
    trainer.callbacks["config_saver"].config = {
        "recipe": "unified-olmo2-370m-farmshare-4xl40s-v1",
        "arm": arm.name,
        "method": arm.method,
        "git_commit": git_commit,
        "scientific_constants": {
            "sequence_length": SEQUENCE_LENGTH,
            "global_batch_tokens": GLOBAL_BATCH_TOKENS,
            "rank_microbatch_tokens": arm.rank_microbatch_tokens,
            "peak_lr": PEAK_LR,
            "warmup_steps": WARMUP_STEPS,
            "alpha_f": ALPHA_F,
            "z_loss_multiplier": Z_LOSS,
            "max_grad_norm": MAX_GRAD_NORM,
        },
    }
    return trainer


def write_identity(
    save_folder: Path,
    progress_dir: Path,
    identity: dict[str, Any],
    *,
    environment: Optional[Mapping[str, Any]] = None,
) -> None:
    """Write the resume-fingerprint identity, plus a record copy for the record.

    ``environment`` (torch/CUDA/driver/GPU/pip-freeze facts, see
    ``token_selection_entrypoint.runtime_environment``) is recorded in the
    progress-dir copy only, *not* folded into ``identity`` or the
    resume-blocking fingerprint at ``save_folder`` -- a driver or dependency
    bump between a run and its resume shouldn't refuse the resume the way a
    change to the scientific identity itself does.
    """
    fingerprint = write_run_fingerprint(save_folder, identity)
    progress_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = make_run_fingerprint(identity)
    if environment is not None:
        payload = {**payload, "environment": dict(environment)}
    (progress_dir / "run_identity.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if fingerprint.name != "run_fingerprint.json":
        raise RuntimeError("unexpected fingerprint path")
