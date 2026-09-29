#!/usr/bin/env python3
"""Phase 1b.5: sanity-check the ``attention_topk`` arm's scoring before a production run.

The ``attention_topk`` selection method scores each token by how much causal
attention it receives from later positions, aligns that (log) score to the
target token it gates, standardizes it against an empirical per-position
baseline of the same model's attention (``AttentionPositionBaseline``: the
mean and std of the aligned log mass at that position over recently seen
rows), and keeps the top ``keep_fraction`` per row (see
``token_selection_370m/selection.py``). At initialization the model's
attention is close to uniform, so this score only carries signal once
attention has differentiated from uniform through training -- this diagnostic
therefore requires an *already-trained* checkpoint, not a fresh init.

In training, a step is scored against the baseline committed from the
*immediately preceding* step's global batch (unblended -- no smoothing
constant to justify, no lag beyond that one unavoidable step). To mirror
that without a training history, this script first builds the baseline from
``--calibration-batches`` batches of real corpus text that are disjoint from
the scored ones (one ``observe`` per batch, then a single ``commit``, as one
training step would), and only then scores ``--num-batches`` further batches
with it. The scored batches never feed the baseline, just as a training
step's own batch never feeds the baseline it is scored against.

Scoring exactly mirrors the ``attention_topk`` branch of
``TokenWeightedTrainModule.train_batch`` (``capture_last_attention`` around the
forward pass, then ``aligned_log_attention_mass``, then
``AttentionPositionBaseline.score``, then ``per_row_topk``) but on a plain,
non-distributed, non-compiled, eval-mode model with no gradients, on a single
GPU.

It then bins every token's *position* (not its selection score) into sixteen
128-position bins and reports each bin's keep rate, pooled across every row
and batch. The acceptance rule is fixed, not tunable from the command line:
every bin's keep rate must fall in [0.3, 0.9]. A bin outside that range means
the attention score is behaving like a disguised positional heuristic (e.g.
always keeping/dropping a whole position band) rather than a real per-token
signal, and the arm should not be run at production scale until that is
understood. This script only reports PASS/FAIL; it never adjusts
``keep_fraction`` or anything else to make a failing bin pass.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List

FARMSHARE_DIR = Path(__file__).resolve().parent
EDULLM_DIR = FARMSHARE_DIR.parent
if str(EDULLM_DIR) not in sys.path:
    sys.path.insert(0, str(EDULLM_DIR))

import torch  # noqa: E402

from eval_task_loss_olmo_core import (  # noqa: E402
    build_model,
    load_model,
    materialize_model_eval,
)
from token_selection_370m.arms import REGMIX, get_arm  # noqa: E402
from token_selection_370m.recipe import SEQUENCE_LENGTH  # noqa: E402
from token_selection_370m.selection import (  # noqa: E402
    AttentionPositionBaseline,
    aligned_log_attention_mass,
    capture_last_attention,
    per_row_topk,
    uniform_prior_log_mass,
)
from token_selection_entrypoint import resolve_corpus  # noqa: E402

# Fixed acceptance gate -- not a CLI argument, not tuned per run.
NUM_BINS = 16
KEEP_RATE_LOW = 0.3
KEEP_RATE_HIGH = 0.9
LABEL_IGNORE_INDEX = -100
TOKENIZER_ID = "tokenizer/dolma2-bpe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help=(
            "Checkpoint directory in the same materialized flat-weights format "
            "eval_task_loss_olmo_core.py's load_model/materialize_model_eval use "
            "(a model_eval.pt file, materialized from model_and_optim/ if missing)."
        ),
    )
    parser.add_argument(
        "--stage-root",
        type=Path,
        required=True,
        help="FarmShare stage root containing ready.json, as written by farmshare/stage_local.py.",
    )
    parser.add_argument("--num-batches", type=int, default=8)
    parser.add_argument(
        "--calibration-batches",
        type=int,
        default=16,
        help=(
            "Batches (disjoint from the scored ones) used to build the empirical "
            "position baseline before scoring. Training builds it from every "
            "row of every previous global batch; this only needs enough rows for "
            "a stable per-position mean/std."
        ),
    )
    parser.add_argument(
        "--batch-size-tokens",
        type=int,
        default=16_384,
        help="Must be a multiple of the recipe's sequence length (2048).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Seed for picking which corpus instances to sample (matches "
            "random-control's data_seed; does not need to match training's real "
            "shuffle, just needs to give a reasonable, reproducible sample)."
        ),
    )
    return parser.parse_args()


def load_checkpoint_model(checkpoint: Path) -> torch.nn.Module:
    """Load a trained checkpoint the same way eval_task_loss_olmo_core.py does.

    Reuses that script's build_model/load_model/materialize_model_eval so this
    diagnostic's model construction, attention backend, and checkpoint format
    never drift from what the production evaluator already does.
    """
    evaluator_state = checkpoint / "model_eval.pt"
    if not evaluator_state.is_file():
        print(f"materializing {evaluator_state} from {checkpoint / 'model_and_optim'} ...", flush=True)
        materialize_model_eval(checkpoint)
    model = build_model()
    step = load_model(checkpoint, model)
    print(f"loaded checkpoint {checkpoint} at step {step}", flush=True)
    return model


def sample_batches(
    *, stage_root: Path, num_batches: int, rows_per_batch: int, seed: int
) -> List[torch.Tensor]:
    """Sample real RegMix corpus instances, not synthetic tokens.

    Resolves the corpus the same way every other script does (via
    token_selection_entrypoint.resolve_corpus / the local ready.json
    manifest), then builds the same NumpyFSLDatasetConfig recipe.py's
    _loader() uses (sequence_length=2048) and grabs a reproducible random
    sample of instances directly from the dataset -- no data loader,
    distributed process group, or real training shuffle order required.
    """
    import os

    from olmo_core.data import NumpyFSLDatasetConfig

    os.environ["EDULLM_INPUT_MANIFEST"] = str(stage_root / "ready.json")
    corpus = resolve_corpus(dataset_id=REGMIX, version="latest", tokenizer_id=TOKENIZER_ID)

    work_dir = Path(tempfile.mkdtemp(prefix="attention-diagnostic-"))
    try:
        dataset = NumpyFSLDatasetConfig(
            paths=list(corpus.paths),
            sequence_length=SEQUENCE_LENGTH,
            tokenizer=corpus.tokenizer,
            dtype=corpus.dtype,
            work_dir=str(work_dir),
        ).build()
        dataset.prepare()

        total_rows = num_batches * rows_per_batch
        if len(dataset) < total_rows:
            raise SystemExit(
                f"corpus {REGMIX} has only {len(dataset)} instances of length "
                f"{SEQUENCE_LENGTH}; need {total_rows} for {num_batches} batches of "
                f"{rows_per_batch} rows each"
            )
        indices = random.Random(seed).sample(range(len(dataset)), total_rows)

        batches = []
        for batch_index in range(num_batches):
            picked = indices[batch_index * rows_per_batch : (batch_index + 1) * rows_per_batch]
            rows = [dataset[i]["input_ids"] for i in picked]
            batches.append(torch.stack(rows, dim=0))
        return batches
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def main() -> None:
    args = parse_args()

    if args.batch_size_tokens <= 0 or args.batch_size_tokens % SEQUENCE_LENGTH != 0:
        raise SystemExit(
            f"--batch-size-tokens ({args.batch_size_tokens}) must be a positive multiple "
            f"of the recipe sequence length ({SEQUENCE_LENGTH})"
        )
    rows_per_batch = args.batch_size_tokens // SEQUENCE_LENGTH
    if args.num_batches <= 0:
        raise SystemExit("--num-batches must be positive")
    if args.calibration_batches <= 0:
        raise SystemExit("--calibration-batches must be positive")
    if SEQUENCE_LENGTH % NUM_BINS != 0:
        raise SystemExit(f"sequence length {SEQUENCE_LENGTH} does not divide evenly into {NUM_BINS} bins")
    bin_size = SEQUENCE_LENGTH // NUM_BINS

    if not torch.cuda.is_available():
        raise SystemExit("this diagnostic requires a CUDA device")

    # Imported, not hardcoded, in case keep_fraction ever changes.
    keep_fraction = get_arm("attention").keep_fraction

    print(f"loading checkpoint from {args.checkpoint} ...", flush=True)
    model = load_checkpoint_model(args.checkpoint)
    device = next(model.parameters()).device

    total_batches = args.calibration_batches + args.num_batches
    print(
        f"sampling {total_batches} batches x {rows_per_batch} rows "
        f"({SEQUENCE_LENGTH} tokens/row) from {REGMIX} (seed={args.seed}): "
        f"{args.calibration_batches} to calibrate the position baseline, "
        f"{args.num_batches} (disjoint) to score ...",
        flush=True,
    )
    all_batches = sample_batches(
        stage_root=args.stage_root,
        num_batches=total_batches,
        rows_per_batch=rows_per_batch,
        seed=args.seed,
    )
    calibration_batches = all_batches[: args.calibration_batches]
    batches = all_batches[args.calibration_batches :]

    # get_labels is the same left-shift-by-one used for a plain LM batch with
    # no special masks in train_module.py's train_batch (batch has only
    # "input_ids"; no label_mask/attention_mask/instance_mask here).
    from olmo_core.data.utils import get_labels

    def log_mass_and_valid(input_ids: torch.Tensor):
        input_ids = input_ids.to(device, non_blocking=True)
        labels = get_labels({"input_ids": input_ids}, label_ignore_index=LABEL_IGNORE_INDEX)
        with capture_last_attention(model) as captured:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(
                    input_ids,
                    labels=labels,
                    ignore_index=LABEL_IGNORE_INDEX,
                    loss_reduction="none",
                    return_logits=False,
                )
        model.reset_auxiliary_metrics()
        return aligned_log_attention_mass(captured), labels != LABEL_IGNORE_INDEX

    baseline = AttentionPositionBaseline()
    with torch.no_grad():
        for batch_index, input_ids in enumerate(calibration_batches):
            log_mass, valid = log_mass_and_valid(input_ids)
            baseline.observe(log_mass, valid)
            print(f"  calibration batch {batch_index + 1}/{len(calibration_batches)}", flush=True)
    baseline.commit()

    # Informational only (not part of the acceptance rule): the measured
    # position effect next to the uniform-attention prior it replaces, as
    # per-bin averages of the aligned log mass's per-position mean/std.
    mean, std = baseline.position_statistics(SEQUENCE_LENGTH, device)
    prior = uniform_prior_log_mass(SEQUENCE_LENGTH, device)
    print()
    print("calibrated position baseline (aligned log attention mass; last column has no label):")
    print(f"{'bin':>3s}  {'positions':>13s}  {'mean':>9s}  {'std':>9s}  {'uniform prior':>13s}")
    for bin_index in range(NUM_BINS):
        start = bin_index * bin_size
        stop = min(start + bin_size, SEQUENCE_LENGTH - 1)
        print(
            f"{bin_index:>3d}  [{start:>5d},{stop - 1:>5d}]  {float(mean[start:stop].mean()):>9.4f}  "
            f"{float(std[start:stop].mean()):>9.4f}  {float(prior[start:stop].mean()):>13.4f}"
        )
    print()

    valid_per_bin = torch.zeros(NUM_BINS, dtype=torch.long)
    kept_per_bin = torch.zeros(NUM_BINS, dtype=torch.long)

    with torch.no_grad():
        for batch_index, input_ids in enumerate(batches):
            log_mass, valid = log_mass_and_valid(input_ids)
            attention = baseline.score(log_mass)
            mask = per_row_topk(attention, keep_fraction, valid, tiebreak=None)

            for bin_index in range(NUM_BINS):
                start = bin_index * bin_size
                stop = start + bin_size
                valid_per_bin[bin_index] += valid[:, start:stop].sum().to("cpu")
                kept_per_bin[bin_index] += mask[:, start:stop].sum().to("cpu")

            print(f"  scored batch {batch_index + 1}/{len(batches)}", flush=True)

    print()
    print(f"attention_topk keep_fraction = {keep_fraction} (from arms.py's 'attention' arm)")
    print(f"acceptance rule (fixed, not tunable): every bin's keep rate must be in [{KEEP_RATE_LOW}, {KEEP_RATE_HIGH}]")
    print()
    header = f"{'bin':>3s}  {'positions':>13s}  {'valid':>8s}  {'kept':>8s}  {'keep_rate':>9s}  result"
    print(header)
    print("-" * len(header))

    all_pass = True
    for bin_index in range(NUM_BINS):
        start = bin_index * bin_size
        stop = start + bin_size - 1
        valid_count = int(valid_per_bin[bin_index].item())
        kept_count = int(kept_per_bin[bin_index].item())
        if valid_count == 0:
            keep_rate = float("nan")
            bin_pass = False
        else:
            keep_rate = kept_count / valid_count
            bin_pass = KEEP_RATE_LOW <= keep_rate <= KEEP_RATE_HIGH
        all_pass = all_pass and bin_pass
        result = "PASS" if bin_pass else "FAIL"
        print(
            f"{bin_index:>3d}  [{start:>5d},{stop:>5d}]  {valid_count:>8d}  {kept_count:>8d}  "
            f"{keep_rate:>9.4f}  {result}"
        )

    overall_valid = int(valid_per_bin.sum().item())
    overall_kept = int(kept_per_bin.sum().item())
    overall_keep_rate = overall_kept / overall_valid if overall_valid else float("nan")
    print()
    print(
        f"corpus-wide keep rate: {overall_keep_rate:.4f} "
        f"(sanity check -- should be close to keep_fraction={keep_fraction} by construction)"
    )
    print()
    print(f"OVERALL: {'PASS' if all_pass else 'FAIL'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
