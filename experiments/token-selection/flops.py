"""Analytic FLOPs for every token-selection arm (Table 2).

The model: a forward pass costs ``2N + 4*L*T*d`` FLOPs per token (``N`` matmul
parameters including the untied output head, ``L`` layers, sequence length ``T``,
width ``d``; the second term is the attention score/value matmuls). A full
forward+backward pass costs 3x a forward pass. Everything below is counted in
"batch forwards" (BF): one no-grad forward over one 4,194,304-token global batch.

Two columns, matching Table 2 of the paper:

  * ``in_run``: training plus every forward pass done inside the training run to
    score tokens (REL-EMA's history forward; BLADE's per-step proxy and reference
    forwards, its pre-scoring forwards and its parity checks; Attention's
    last-block QK^T recompute).
  * ``total``: ``in_run`` plus reference-model training and offline scoring
    (RHO-1 and Perplexity read reference losses from one offline pass over the
    whole RegMix corpus by the frozen Instruct reference, and are each charged
    the full pass and the full reference pretraining, although both are shared;
    BLADE's reference update steps).

Usage:  python flops.py            # print the table
        python flops.py --json     # print machine-readable numbers
"""

from __future__ import annotations

import argparse
import json

N = 371_195_904
L = 16
T = 2048
D = 1024
FWD_PER_TOKEN = 2 * N + 4 * L * T * D
GLOBAL_BATCH_TOKENS = 4_194_304
BF = GLOBAL_BATCH_TOKENS * FWD_PER_TOKEN  # one batch forward

TRAIN_STEPS = 2360
REFERENCE_STEPS = 940
CORPUS_INSTANCES = 4_885_156  # whole RegMix corpus, scored once offline
BLADE_SYNC_STEPS = (0, 400, 800, 1200, 1600, 2000)
BLADE_K = 75

# Attention's extra cost: recompute Q and K for the last block and form QK^T,
# per token, relative to a forward pass of the whole model.
ATTENTION_EXTRA_PER_TOKEN = 2 * (2 * D * D) + 2 * T * D  # Q and K projections + QK^T (one layer)


def _e18(batch_forwards: float) -> float:
    return batch_forwards * BF / 1e18


def table() -> dict[str, dict[str, float]]:
    train = 3 * TRAIN_STEPS  # fwd+bwd, in BF
    reference_pretrain = 3 * REFERENCE_STEPS
    offline_pass = CORPUS_INSTANCES * T / GLOBAL_BATCH_TOKENS  # in BF
    n_sync = len(BLADE_SYNC_STEPS)
    blade_step_scoring = 2 * TRAIN_STEPS  # proxy + reference forward every step
    blade_prescoring = (n_sync - 1) * BLADE_K * 2  # 2 forwards per K batch, not at the first sync
    blade_parity = n_sync * 2
    blade_kupdates = n_sync * BLADE_K * 2 * 3  # 2 batches per K step, fwd+bwd
    attention = TRAIN_STEPS * GLOBAL_BATCH_TOKENS * ATTENTION_EXTRA_PER_TOKEN / BF

    rows = {
        "Full-loss control": (train, train),
        "Random control": (train, train),
        "Attention": (train + attention, train + attention),
        "REL-EMA": (train + TRAIN_STEPS, train + TRAIN_STEPS),
        "RHO-1": (train, train + offline_pass + reference_pretrain),
        "Perplexity": (train, train + offline_pass + reference_pretrain),
        "BLADE": (
            train + blade_step_scoring + blade_prescoring + blade_parity,
            train + blade_step_scoring + blade_prescoring + blade_parity + blade_kupdates,
        ),
    }
    control = _e18(train)
    out: dict[str, dict[str, float]] = {}
    for name, (in_run, total) in rows.items():
        out[name] = {
            "in_run_1e18": _e18(in_run),
            "total_1e18": _e18(total),
            "in_run_rel_control": _e18(in_run) / control,
            "total_rel_control": _e18(total) / control,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="print machine-readable numbers")
    args = parser.parse_args()
    data = table()
    if args.json:
        print(json.dumps({"forward_flops_per_token": FWD_PER_TOKEN, "arms": data}, indent=1))
        return
    print(f"forward FLOPs/token = 2N + 4LTd = {FWD_PER_TOKEN:,}")
    print(f"{'Arm':<20}{'in-run (e18)':>14}{'total (e18)':>14}{'in-run x':>10}{'total x':>10}")
    for name, row in data.items():
        print(
            f"{name:<20}{row['in_run_1e18']:>14.2f}{row['total_1e18']:>14.2f}"
            f"{row['in_run_rel_control']:>10.3f}{row['total_rel_control']:>10.3f}"
        )


if __name__ == "__main__":
    main()
