#!/usr/bin/env python3
"""Score the whole RegMix-10B corpus once against the frozen Instruct reference.

Standalone, single-process, single-GPU (the FarmShare ``qos=normal`` 1-GPU
lane -- NOT the 4-GPU ``qos=gpu`` lane production training uses). This is the
producer for the offline per-instance reference-loss table described in
``token_selection_370m/reference_scores.py`` (read that module first; this
script must write files it can read back without any changes to it).

Why this exists: RHO-1 and Perplexity both score every training token against
the same frozen ``instruct-reference`` checkpoint (step 940). Doing that
online means keeping a second copy of the reference resident in GPU memory
and re-scoring it every step. Since selection is per 2048-token instance and
instance content depends only on corpus paths/dtype/sequence-length -- never
on world size, rank, or microbatch composition -- the reference's per-token
CE can instead be computed exactly once, offline, and looked up later by
``batch["index"]`` (the global instance id every training batch carries).

This script does NOT import ``token_selection_370m.train_module`` (or
``token_selection_370m.recipe.build_trainer`` / ``_custom_module``, which
import it internally). That module is being actively rewritten by another
engineer in parallel with this script and, at the time this script was
written, its module-level ``from .selection import (..., WeightShadow, ...)``
does not import cleanly because ``WeightShadow`` does not exist yet in
``selection.py``. Depending on it here would make this script fail to import
through no fault of its own. Instead, this script builds a plain
``TransformerTrainModule`` directly from ``recipe.py``'s ``_train_module_config``
(a pure function with no such dependency) via ``TransformerTrainModuleConfig.build()``.
This is sufficient: ``TokenWeightedTrainModule`` (the class ``build_trainer``
would otherwise construct) never overrides ``model_forward``, only
``train_batch``/``_score_many``/etc, so a plain ``TransformerTrainModule``'s
``model_forward`` is byte-for-byte the same forward path -- same HSDP wrapping,
same bf16 params / fp32 reduce, same ``model_forward(..., loss_reduction="none",
return_logits=False)`` call and ``output.ce_loss`` read -- that online scoring
already used.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint.state_dict as dist_cp_sd

FARMSHARE_DIR = Path(__file__).resolve().parent
EDULLM_DIR = FARMSHARE_DIR.parent
for _extra_path in (str(EDULLM_DIR), str(FARMSHARE_DIR)):
    if _extra_path not in sys.path:
        sys.path.insert(0, _extra_path)

# Reuse the shared schema module as-is (do not edit it): table_file,
# manifest_file, progress_file, write_manifest, write_progress, read_progress,
# read_manifest, chunk_sha256, open_table, CHUNK_ROWS, ALGORITHM,
# MANIFEST_SCHEMA_VERSION.
from token_selection_370m import arms  # noqa: E402
from token_selection_370m import reference_scores  # noqa: E402
from token_selection_370m.recipe import (  # noqa: E402
    SEQUENCE_LENGTH,
    _reference_digest,
    _train_module_config,
    immutable_corpus_binding,
)
from token_selection_370m.selection import per_row_middle  # noqa: E402

# Reuse the one entrypoint's corpus/reference resolution -- do not reimplement
# it. Importing this module is safe even though train_module.py is currently
# broken: it only imports the *name* `build_trainer` from recipe.py at module
# level (which does not itself import train_module.py), and never calls it.
import token_selection_entrypoint as entrypoint  # noqa: E402

# stage_local.py lives next to this script; reuse its manifest read/write
# helpers rather than duplicating ready.json's schema-2 conventions. These are
# underscore-prefixed but that only affects `from stage_local import *`, not
# an explicit import like this.
from stage_local import _load_manifest, _write_manifest  # noqa: E402

from olmo_core.data import NumpyFSLDatasetConfig  # noqa: E402
from olmo_core.data.utils import get_labels  # noqa: E402
from olmo_core.distributed.utils import get_local_tensor  # noqa: E402
from olmo_core.nn.lm_head import LMOutputWithLoss  # noqa: E402
from olmo_core.nn.transformer import TransformerConfig  # noqa: E402

LOG = logging.getLogger("score_reference")

# The mask-agreement parity check reuses the same keep_fraction the real
# middle_ppl arm ("perplexity") uses; the exact value doesn't matter for a
# parity check (any fixed fraction would do), it just has to be applied
# identically to both the batched-write scores and the microbatch-1 rescore.
PARITY_KEEP_FRACTION = 0.6


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Defaults to <stage-root>/reference-scores/<REFERENCE_SCORES_CONTRACT>. "
            "Pass an explicit path for a truncated --max-instances validation run so it "
            "never collides with the real, full-corpus table."
        ),
    )
    parser.add_argument(
        "--dataset-version",
        default=None,
        help="Same semantics as token_selection_entrypoint.py's EDULLM_DATASET_VERSION: "
        "'', 'latest', or the exact staged version. Defaults to $EDULLM_DATASET_VERSION "
        "or 'latest'.",
    )
    parser.add_argument("--microbatch-instances", type=int, default=8)
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Score only the first N instances (corpus order), for a quick validation run "
        "before instruct-reference exists at full scale. The manifest's 'rows' then "
        "reflects the truncated count, not the full corpus, and is marked accordingly.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Scratch dir for the dataset's cached offsets. Defaults to "
        "<stage-root>/work/score-reference.",
    )
    parser.add_argument("--parity-sample-size", type=int, default=256)
    parser.add_argument("--log-interval-seconds", type=float, default=30.0)
    return parser.parse_args()


# --------------------------------------------------------------------------
# Single-rank process group + model construction
# --------------------------------------------------------------------------


def init_single_rank_process_group() -> None:
    """Init a real world_size=1 process group so is_distributed() is True.

    recipe.py's `_train_module_config` only attaches the HSDP
    (bf16 params / fp32 reduce) `dp_config` when `is_distributed()` is True;
    otherwise it scores in fp32, which would NOT match the online per-microbatch
    scoring path this table replaces. So this process must genuinely join a
    process group, not just set env vars.

    Follows eval_task_loss_olmo_core.py's world_size=1 precedent for *how* to
    rendezvous (a file:// init_method with rank=0, world_size=1, which avoids
    the bind-probe-release-rebind hang that an env:// TCPStore hits on
    FarmShare) but NOT its backend choice. That script stays on gloo because
    it never builds a CUDA device mesh (its model is a plain, unwrapped
    `.build(init_device="cuda")`). This script must build a genuine "cuda"
    DeviceMesh for HSDP so the model is wrapped exactly as production wraps
    it, and `init_device_mesh("cuda", ...)` needs a backend that actually
    supports CUDA collectives -- so this uses NCCL. With world_size=1 there is
    no cross-rank traffic, but NCCL's init still probes IB/network topology
    (the same risk called out in eval_task_loss_olmo_core.py's own comment)
    and can hang for minutes on a FarmShare allocation; the env vars below are
    the standard workaround (skip IB probing, force loopback, disable P2P,
    all irrelevant with a single GPU anyway).
    """
    if dist.is_initialized():
        return
    if not torch.cuda.is_available():
        raise SystemExit("score_reference.py requires a CUDA GPU (the qos=normal 1-GPU lane)")
    torch.cuda.set_device(0)
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("LOCAL_WORLD_SIZE", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    rendezvous_dir = Path(tempfile.mkdtemp(prefix="score-reference-rendezvous-"))
    rendezvous_file = rendezvous_dir / "store"
    LOG.info("init_process_group(backend=nccl, world_size=1, file://%s)...", rendezvous_file)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{rendezvous_file}",
        rank=0,
        world_size=1,
        timeout=timedelta(minutes=30),
    )
    LOG.info("process group initialized")


def build_scoring_module(microbatch_instances: int, vocab_size: int, device: torch.device):
    """Build a plain TransformerTrainModule wrapped exactly as training wraps it.

    Reuses `_train_module_config` from recipe.py so the HSDP dp_config
    (bf16 params, fp32 reduce) is identical to what `build_trainer` would
    build for these arms; only `compile_model` is overridden to False, since
    a one-shot forward-only inference pass doesn't need torch.compile's
    warm-up cost and compiling doesn't change shapes/dtypes/numerics.
    """
    model_config = TransformerConfig.olmo2_370M(vocab_size=vocab_size, init_seed=0)
    model = model_config.build(init_device="meta")
    module_config = _train_module_config(microbatch_instances * SEQUENCE_LENGTH)
    module_config.compile_model = False
    train_module = module_config.build(model, device=device)
    train_module.model.eval()
    for parameter in train_module.model.parameters():
        parameter.requires_grad_(False)
    return train_module


def load_reference_weights(model: torch.nn.Module, reference_path: str) -> int:
    """Load the materialized {"step": N, "model": {...}} reference checkpoint.

    Follows eval_task_loss_olmo_core.py's materialize_model_eval/load_model
    payload convention (a flat dict of real tensors under "model"), but loads
    with strict=True per this task's explicit instruction -- eval_task_loss's
    own `load_model` actually tolerates up to 5% missing keys via
    `strict=False`, but a reference-scoring table should fail loudly rather
    than silently score against a partially-initialized reference.

    Uses `torch.distributed.checkpoint.state_dict.set_model_state_dict` with
    `full_state_dict=True` rather than a plain `model.load_state_dict(...)`:
    the model's parameters are DTensors (it's HSDP-wrapped, matching
    production), and this is the documented way to load a full/unsharded
    state dict into a (possibly trivially, at world_size=1) sharded model.
    """
    payload = torch.load(reference_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise RuntimeError(f"invalid reference checkpoint payload: {reference_path}")
    raw_state_dict = payload["model"]
    state_dict = {str(name): value for name, value in raw_state_dict.items() if torch.is_tensor(value)}
    if len(state_dict) != len(raw_state_dict):
        raise RuntimeError(f"reference checkpoint contains non-tensor values: {reference_path}")
    options = dist_cp_sd.StateDictOptions(
        full_state_dict=True, broadcast_from_rank0=True, strict=True
    )
    dist_cp_sd.set_model_state_dict(model, state_dict, options=options)
    return int(payload.get("step", -1))


def build_dataset(corpus, work_dir: Path):
    work_dir.mkdir(parents=True, exist_ok=True)
    dataset = NumpyFSLDatasetConfig(
        paths=list(corpus.paths),
        sequence_length=SEQUENCE_LENGTH,
        tokenizer=corpus.tokenizer,
        dtype=corpus.dtype,
        work_dir=str(work_dir),
    ).build()
    dataset.prepare()
    return dataset


@torch.no_grad()
def score_microbatch(train_module, ids_cpu: torch.Tensor) -> np.ndarray:
    """Score one microbatch exactly as the (soon-to-be-removed) online
    `_score_many` did: `model_forward(ids, labels=labels, ignore_index=...,
    loss_reduction="none", return_logits=False)`, reading `output.ce_loss`
    (already detached; 0.0 at ignored/last-column positions).
    """
    labels_cpu = get_labels({"input_ids": ids_cpu}, label_ignore_index=train_module.label_ignore_index)
    ids = ids_cpu.to(train_module.device, non_blocking=True)
    labels = labels_cpu.to(train_module.device, non_blocking=True)
    output = train_module.model_forward(
        ids,
        labels=labels,
        ignore_index=train_module.label_ignore_index,
        loss_reduction="none",
        return_logits=False,
    )
    if not isinstance(output, LMOutputWithLoss):
        raise RuntimeError("OLMo did not return per-token scoring loss")
    ce = get_local_tensor(output.ce_loss).reshape_as(labels).detach()
    return ce.to("cpu", dtype=torch.float32).numpy()


# --------------------------------------------------------------------------
# Progress / chunk bookkeeping
# --------------------------------------------------------------------------


def _flush_completed_chunks(
    output_root: Path,
    memmap: np.memmap,
    rows: int,
    next_row: int,
    chunk_hashes: List[str],
    ce_sum: float,
    ce_count: int,
    corpus_binding: Dict[str, Any],
    reference_sha256: Optional[str],
) -> Tuple[float, int]:
    """Hash+checkpoint every chunk that `next_row` has now fully covered.

    A chunk boundary is [chunk_index*CHUNK_ROWS, min((chunk_index+1)*CHUNK_ROWS, rows)).
    Handles a `--microbatch-instances` that doesn't evenly divide CHUNK_ROWS
    (chunks complete independently of microbatch boundaries) and the final,
    possibly-shorter last chunk.
    """
    while True:
        chunk_index = len(chunk_hashes)
        chunk_start = chunk_index * reference_scores.CHUNK_ROWS
        if chunk_start >= rows:
            break
        chunk_end = min(chunk_start + reference_scores.CHUNK_ROWS, rows)
        if next_row < chunk_end:
            break
        chunk = np.asarray(memmap[chunk_start:chunk_end])
        if not np.isfinite(chunk).all():
            bad_local = np.argwhere(~np.isfinite(chunk))[0]
            raise RuntimeError(
                f"non-finite reference CE in rows [{chunk_start},{chunk_end}); "
                f"first bad (row_offset, col)={tuple(int(x) for x in bad_local)}"
            )
        chunk_hashes.append(reference_scores.chunk_sha256(chunk))
        # Corpus-wide mean CE is over all valid (non-final-column) positions;
        # this dataset never uses label_mask_paths, so every row's validity
        # mask is exactly "every column except the last".
        ce_sum += float(chunk[:, :-1].sum(dtype=np.float64))
        ce_count += int(chunk.shape[0] * (chunk.shape[1] - 1))
        memmap.flush()
        reference_scores.write_progress(
            output_root,
            {
                "corpus_binding": corpus_binding,
                "reference_sha256": reference_sha256,
                "next_row": next_row,
                "chunk_sha256": chunk_hashes,
                "rows": rows,
                "ce_sum": ce_sum,
                "ce_count": ce_count,
            },
        )
        LOG.info(
            "checkpointed chunk %d: rows [%d,%d) sha256=%s",
            chunk_index,
            chunk_start,
            chunk_end,
            chunk_hashes[-1][:12],
        )
    return ce_sum, ce_count


# --------------------------------------------------------------------------
# Parity / sanity checks
# --------------------------------------------------------------------------


def run_parity_checks(
    train_module,
    dataset,
    memmap: np.memmap,
    rows: int,
    *,
    sample_size: int,
) -> Tuple[float, float, int]:
    """Re-score `sample_size` random instances at microbatch size 1 and
    compare against the batched-write table. Returns
    (max_abs_diff, mask_agreement_fraction, actual_sample_size).
    """
    sample_size = max(0, min(sample_size, rows))
    if sample_size == 0:
        return 0.0, 1.0, 0
    rng = np.random.default_rng(0)
    sample_rows = rng.choice(rows, size=sample_size, replace=False)
    original = np.array(memmap[sample_rows], dtype=np.float32, copy=True)
    rescored = np.empty_like(original)
    for i, row in enumerate(sample_rows.tolist()):
        ids_cpu = dataset[int(row)]["input_ids"].unsqueeze(0)
        rescored[i] = score_microbatch(train_module, ids_cpu)[0]

    max_abs_diff = float(np.max(np.abs(rescored - original)))

    valid = torch.ones((sample_size, SEQUENCE_LENGTH), dtype=torch.bool)
    valid[:, -1] = False
    mask_original = per_row_middle(torch.from_numpy(original), PARITY_KEEP_FRACTION, valid)
    mask_rescored = per_row_middle(torch.from_numpy(rescored), PARITY_KEEP_FRACTION, valid)
    agree = (mask_original == mask_rescored)[:, :-1]
    mask_agreement = float(agree.float().mean().item())
    return max_abs_diff, mask_agreement, sample_size


# --------------------------------------------------------------------------
# ready.json wiring
# --------------------------------------------------------------------------


def update_ready_manifest(stage_root: Path, output_root: Path) -> None:
    manifest_path = stage_root / "ready.json"
    payload = _load_manifest(manifest_path)
    payload.setdefault("reference_scores", {})[arms.REFERENCE_SCORES_CONTRACT] = str(output_root)
    _write_manifest(manifest_path, payload)
    LOG.info("wrote %s: reference_scores[%r] = %s", manifest_path, arms.REFERENCE_SCORES_CONTRACT, output_root)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    stage_root = args.stage_root
    # Resolve inputs from the same local manifest every other script uses:
    # point token_selection_entrypoint.py's `_manifest()` at this run's
    # ready.json, exactly as farmshare/launch.sh's `EDULLM_INPUT_MANIFEST`
    # export does for the training entrypoint.
    os.environ["EDULLM_INPUT_MANIFEST"] = str(stage_root / "ready.json")
    manifest = entrypoint._manifest()
    LOG.info(
        "using local manifest %s (corpora=%s references=%s)",
        stage_root / "ready.json",
        sorted(manifest.get("corpora", {})),
        sorted(manifest.get("references", {})),
    )

    dataset_version = args.dataset_version
    if dataset_version is None:
        dataset_version = os.environ.get("EDULLM_DATASET_VERSION", "latest")

    corpus = entrypoint.resolve_corpus(
        dataset_id=arms.REGMIX,
        version=dataset_version,
        tokenizer_id="tokenizer/dolma2-bpe",
    )
    reference_path = entrypoint.resolve_reference_path(arms.INSTRUCT_REFERENCE_CONTRACT)
    if reference_path is None:
        raise SystemExit(f"no materialized reference for {arms.INSTRUCT_REFERENCE_CONTRACT!r}")

    corpus_binding = immutable_corpus_binding(arms.REGMIX, corpus)
    LOG.info("corpus binding: %s", corpus_binding)
    reference_sha256 = _reference_digest(reference_path)
    LOG.info("reference %s sha256=%s", reference_path, reference_sha256)

    output_root: Path = args.output_root or (
        stage_root / "reference-scores" / arms.REFERENCE_SCORES_CONTRACT
    )
    if args.max_instances is not None and args.output_root is None:
        LOG.warning(
            "--max-instances is set but --output-root was not overridden; this will write a "
            "TRUNCATED validation table to the production output path %s. Pass --output-root "
            "explicitly to avoid confusing it with the full-corpus table.",
            output_root,
        )

    work_dir = args.work_dir or (stage_root / "work" / "score-reference")
    dataset = build_dataset(corpus, work_dir)
    full_rows = len(dataset)
    rows = full_rows if args.max_instances is None else min(int(args.max_instances), full_rows)
    if rows <= 0:
        raise SystemExit(f"nothing to score: corpus has {full_rows} instances, --max-instances={args.max_instances}")
    LOG.info("corpus has %d instances; scoring %d", full_rows, rows)

    # Already-complete fast path: an idempotent rerun of this script (e.g. a
    # retried sbatch submission) should not silently re-score or overwrite a
    # finished table computed against a different reference/corpus.
    if reference_scores.manifest_file(output_root).is_file():
        existing_manifest = reference_scores.read_manifest(output_root)
        if (
            existing_manifest.get("complete")
            and existing_manifest.get("reference_sha256") == reference_sha256
            and existing_manifest.get("corpus_binding") == corpus_binding
            and int(existing_manifest.get("rows", -1)) == rows
        ):
            LOG.info("table at %s is already complete and matches this run; nothing to do", output_root)
            if args.max_instances is None:
                update_ready_manifest(stage_root, output_root)
            return
        raise SystemExit(
            f"{reference_scores.manifest_file(output_root)} already exists and does not match "
            "this run's reference/corpus/rows; refusing to overwrite. Use a different "
            "--output-root or remove the existing table if this is intentional."
        )

    existing_progress = reference_scores.read_progress(output_root)
    scores_path = reference_scores.table_file(output_root)
    if existing_progress is not None:
        if existing_progress.get("corpus_binding") != corpus_binding:
            raise SystemExit(
                f"cannot resume {output_root}: progress was recorded for a different corpus binding"
            )
        if existing_progress.get("reference_sha256") != reference_sha256:
            raise SystemExit(
                f"cannot resume {output_root}: progress was recorded against a different reference checkpoint"
            )
        if int(existing_progress.get("rows", rows)) != rows:
            raise SystemExit(
                f"cannot resume {output_root}: progress was recorded for rows="
                f"{existing_progress.get('rows')}, this run targets rows={rows}"
            )
        if not scores_path.is_file():
            raise SystemExit(f"progress.json exists at {output_root} but {scores_path} is missing")
        next_row = int(existing_progress.get("next_row", 0))
        chunk_hashes = list(existing_progress.get("chunk_sha256", []))
        ce_sum = float(existing_progress.get("ce_sum", 0.0))
        ce_count = int(existing_progress.get("ce_count", 0))
        LOG.info("resuming from progress.json: next_row=%d (%d chunks already checkpointed)", next_row, len(chunk_hashes))
        memmap = np.memmap(scores_path, mode="r+", dtype=np.float32, shape=(rows, SEQUENCE_LENGTH))
    else:
        if scores_path.is_file():
            raise SystemExit(
                f"{scores_path} already exists but no progress.json was found at {output_root}; "
                "refusing to overwrite -- remove it manually if this is intentional"
            )
        output_root.mkdir(parents=True, exist_ok=True)
        next_row = 0
        chunk_hashes = []
        ce_sum = 0.0
        ce_count = 0
        memmap = np.memmap(scores_path, mode="w+", dtype=np.float32, shape=(rows, SEQUENCE_LENGTH))

    init_single_rank_process_group()
    device = torch.device("cuda", 0)
    vocab_size = corpus.tokenizer.padded_vocab_size()
    train_module = build_scoring_module(args.microbatch_instances, vocab_size, device)
    step = load_reference_weights(train_module.model, reference_path)
    LOG.info("loaded reference checkpoint (step=%d) into a %s-wrapped model on %s", step, type(train_module.model).__name__, device)

    t_start = time.monotonic()
    last_log = t_start
    rows_at_start = next_row
    microbatch_instances = max(1, int(args.microbatch_instances))
    for start in range(next_row, rows, microbatch_instances):
        stop = min(start + microbatch_instances, rows)
        ids_cpu = torch.stack([dataset[i]["input_ids"] for i in range(start, stop)], dim=0)
        ce = score_microbatch(train_module, ids_cpu)
        memmap[start:stop] = ce
        next_row = stop
        ce_sum, ce_count = _flush_completed_chunks(
            output_root, memmap, rows, next_row, chunk_hashes, ce_sum, ce_count, corpus_binding, reference_sha256
        )
        now = time.monotonic()
        if now - last_log >= args.log_interval_seconds or next_row == rows:
            elapsed = max(now - t_start, 1e-9)
            done = next_row - rows_at_start
            rate = done / elapsed
            remaining = rows - next_row
            eta_min = (remaining / rate / 60.0) if rate > 0 else float("inf")
            LOG.info(
                "scored %d/%d rows (%.1f rows/s, %.0f tok/s, ETA %.1f min)",
                next_row,
                rows,
                rate,
                rate * SEQUENCE_LENGTH,
                eta_min,
            )
            last_log = now
    memmap.flush()

    LOG.info("running parity/sanity checks (sample_size=%d)...", args.parity_sample_size)
    max_abs_diff, mask_agreement, parity_sample_size = run_parity_checks(
        train_module, dataset, memmap, rows, sample_size=args.parity_sample_size
    )
    LOG.info("parity check: max_abs_diff=%.6e mask_agreement=%.6f (n=%d)", max_abs_diff, mask_agreement, parity_sample_size)

    ok = True
    if max_abs_diff > 1e-3:
        LOG.error(
            "!!! PARITY CHECK FAILED !!! max_abs_diff=%.6e exceeds 1e-3: a fresh microbatch-1 "
            "rescore of %d random rows does not match what the batched write already wrote for "
            "those same rows. Same weights, same forward, only batching differs -- this should "
            "be ~0, so a value this large is a real bug (batching-dependent numerics, an indexing "
            "mistake, or a non-deterministic forward path). Refusing to mark the table complete.",
            max_abs_diff,
            parity_sample_size,
        )
        ok = False
    if mask_agreement < 0.999:
        LOG.error(
            "!!! PARITY CHECK FAILED !!! middle_ppl mask agreement=%.6f is below the required "
            "0.999 between the batched-write scores and the microbatch-1 rescore. Refusing to "
            "mark the table complete.",
            mask_agreement,
        )
        ok = False
    if not ok:
        raise SystemExit(1)

    mean_ce = (ce_sum / ce_count) if ce_count > 0 else float("nan")
    LOG.info("corpus-wide mean CE over valid positions: %.6f", mean_ce)

    git_commit = entrypoint.git_commit()
    manifest_payload: Dict[str, Any] = {
        "schema_version": reference_scores.MANIFEST_SCHEMA_VERSION,
        "algorithm": reference_scores.ALGORITHM,
        "complete": True,
        "reference_sha256": reference_sha256,
        "corpus_binding": corpus_binding,
        "sequence_length": SEQUENCE_LENGTH,
        "rows": rows,
        "dtype": "float32",
        "chunk_rows": reference_scores.CHUNK_ROWS,
        "chunk_sha256": chunk_hashes,
        "microbatch_instances": microbatch_instances,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "git_commit": git_commit,
        "compiled": False,
        "dataset_version": corpus.version,
        "mean_ce": mean_ce,
        "parity_max_abs_diff": max_abs_diff,
        "parity_mask_agreement": mask_agreement,
        "parity_sample_size": parity_sample_size,
    }
    if args.max_instances is not None:
        manifest_payload["max_instances_arg"] = int(args.max_instances)
        manifest_payload["truncated_validation_run"] = True
    reference_scores.write_manifest(output_root, manifest_payload)
    LOG.info("wrote %s (complete=true)", reference_scores.manifest_file(output_root))

    if args.max_instances is None:
        update_ready_manifest(stage_root, output_root)
    else:
        LOG.info(
            "skipping ready.json update for a truncated (--max-instances) validation run; "
            "rerun without --max-instances (and without --output-root) to register the real table"
        )


if __name__ == "__main__":
    main()
