#!/usr/bin/env python3
"""MMLU STEM on the validation and test splits together, for the static arms.

The paper's per-task paragraph quotes the MMLU STEM shift of the static mixtures. This
averages the two MMLU STEM labels (val and test splits) at each eval step, fits and
bootstraps like ``fit_and_bootstrap_370m.py`` (own stream per arm), and prints each
static arm's fitted final, 95% CI and difference from the control average, plus the
seed spread. Needs W&B access (per-label curves come from the arms' runs).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pull_370m_curves as P  # noqa: E402
from fit_and_bootstrap_370m import ci, diff_p, fit_and_bootstrap, fmt_p  # noqa: E402

LABELS = ["mmlu_stem_val_rc_5shot_bpb", "mmlu_stem_test_rc_5shot_bpb"]
KEYS = {"s42": "olmo-mix-1124-s42", "s69": "olmo-mix-1124-s69", "dml": "data-mixing-laws-paper",
        "mixlaw": "mixlaw-fit", "lgbm": "lightgbm-l40s"}
N_BOOT = 200_000


def main() -> None:
    import wandb

    api = wandb.Api()
    arms = {a: P.pull_arm(api, k) for a, k in KEYS.items()}
    streams = np.random.SeedSequence(0).spawn(len(KEYS))
    fin, fit = {}, {}
    for (a, _), stream in zip(KEYS.items(), streams):
        v = [np.mean([arms[a]["_per_label"][s][lab] for lab in LABELS]) for s in P.EVAL_LADDER]
        fit[a], fin[a] = fit_and_bootstrap(P.EVAL_LADDER, v, final_step=2384, n_boot=N_BOOT, seed=stream)

    ctrl_d = 0.5 * (fin["s42"] + fin["s69"])
    ctrl_f = 0.5 * (fit["s42"] + fit["s69"])
    lo, hi = ci(ctrl_d)
    print(f"MMLU STEM (val+test mean) control average fitted final {ctrl_f:.4f} [{lo:.4f}, {hi:.4f}]")
    for a, name in (("dml", "DML"), ("mixlaw", "MixLaw"), ("lgbm", "LightGBM")):
        lo, hi = ci(fin[a])
        dlo, dhi = ci(fin[a] - ctrl_d)
        print(f"  {name:9s} fitted {fit[a]:.4f} [{lo:.4f}, {hi:.4f}] | delta {fit[a] - ctrl_f:+.4f} "
              f"[{dlo:+.4f}, {dhi:+.4f}] {fmt_p(diff_p(fin[a], ctrl_d), N_BOOT)}")
    slo, shi = ci(fin["s69"] - fin["s42"])
    print(f"  seed spread (s69 - s42) {fit['s69'] - fit['s42']:+.4f} [{slo:+.4f}, {shi:+.4f}] "
          f"{fmt_p(diff_p(fin['s69'], fin['s42']), N_BOOT)}")


if __name__ == "__main__":
    main()
