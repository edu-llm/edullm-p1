#!/usr/bin/env python3
"""Fetch token-selection task-loss curves from W&B and compute bootstrap CIs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import wandb

ROOT = Path(__file__).resolve().parent
MIXLAW_ROOT = ROOT.parent / "skill-dag" / "mixlaw"
if str(MIXLAW_ROOT) not in sys.path:
    sys.path.insert(0, str(MIXLAW_ROOT))

from mixlaw_power_law import residual_bootstrap_ci  # noqa: E402

METRIC = "eval/macro_bpb"
WANDB_ENTITY = "eduLLM"
WANDB_PROJECT = "token-selection"
# The control lives in the token-selection project now. It used to point at
# eduLLM/mixlaw-1/1e9df6cc... (run "mix01"), which is NOT the reported control and
# does not match the cached artifacts; fixed 2026-09-12 to the reported run.

ARMS = [
    {
        "key": "control",
        "label": "Control",
        "wandb_path": f"{WANDB_ENTITY}/{WANDB_PROJECT}/hh19uatg",
        "run_name": "full-loss-control-regmix10b-v2",
        "final_step": 2384,
    },
    {
        "key": "rho_1",
        "label": "RHO-1",
        "wandb_path": f"{WANDB_ENTITY}/{WANDB_PROJECT}/ebf1fa33048b3459f768cd471c2a8917",
        "run_name": "rho-1-regmix10b-v1",
        "final_step": 2360,
    },
    {
        "key": "attention",
        "label": "Attention top-k",
        "wandb_path": f"{WANDB_ENTITY}/{WANDB_PROJECT}/01e18e7141fdbf9b988f17c32bb0c084",
        "run_name": "attention-topk-10b-scratch-v1",
        "final_step": 2360,
    },
    {
        "key": "blade",
        "label": "BLADE",
        "wandb_path": f"{WANDB_ENTITY}/{WANDB_PROJECT}/005xjces",
        "run_name": "blade-regmix10b-refhq-instruct-v3-v1",
        "final_step": 2360,
    },
    {
        "key": "middle_ppl",
        "label": "Middle-PPL",
        "wandb_path": f"{WANDB_ENTITY}/{WANDB_PROJECT}/2bbd4ec49b531d37115a44f73a0512e2",
        "run_name": "middle-ppl-token-10b-v2",
        "final_step": 2360,
    },
    {
        "key": "rel_ema",
        "label": "REL-EMA",
        # 89db0d5b... is the OLD inverted-polarity A100 run; the reported arm is the
        # 2026-09-05 polarity-corrected re-run. Fixed 2026-09-12.
        "wandb_path": f"{WANDB_ENTITY}/{WANDB_PROJECT}/cc52d5537a03ad8e57cc87a025668b2e",
        "run_name": "rel-ema-exp-10b-scratch-v1",
        "final_step": 2360,
    },
    {
        "key": "random_control",
        "label": "Random control",
        "wandb_path": f"{WANDB_ENTITY}/{WANDB_PROJECT}/fa841187ff07e9164da282efd353c217",
        "run_name": "random-control-regmix10b-v1",
        "final_step": 2360,
    },
]

CURVES_PATH = ROOT / "token_selection_370m_wandb_curves.json"
BOOTSTRAP_PATH = ROOT / "token_selection_370m_bootstrap_results.json"


def fetch_curve(api: wandb.Api, arm: dict) -> tuple[list[int], list[float]]:
    run = api.run(arm["wandb_path"])
    hist = run.history(samples=500, keys=["_step", METRIC])
    hist = hist.dropna(subset=[METRIC]).sort_values("_step")
    steps = hist["_step"].astype(int).tolist()
    losses = hist[METRIC].astype(float).tolist()
    if not steps:
        raise RuntimeError(f"No {METRIC} history for {arm['wandb_path']}")
    return steps, losses


def main() -> None:
    api = wandb.Api()
    curves_payload: dict = {
        "source": f"wandb {WANDB_ENTITY}/{WANDB_PROJECT}",
        "metric": METRIC,
        "arms": {},
    }
    bootstrap_payload: dict = {
        "source": curves_payload["source"],
        "metric": "macro task-loss bits per byte (lower is better)",
        "method": (
            "Fit y = a + b/step^alpha on steps >= 1000; fitted final at arm final_step; "
            "95% CI from residual bootstrap (10k draws, alpha fixed from initial fit). "
            "NOTE: the reported Table 1 intervals come from fit_and_plot.py, which "
            "re-estimates alpha on every bootstrap draw over a wider grid "
            "np.linspace(0.05, 6.0, 1192) and therefore yields wider intervals. "
            "Prefer fit_and_plot.py for published numbers."
        ),
        "arms": [],
    }

    for arm in ARMS:
        steps, losses = fetch_curve(api, arm)
        curves_payload["arms"][arm["key"]] = {
            "label": arm["label"],
            "wandb_path": arm["wandb_path"],
            "run_name": arm["run_name"],
            "final_step": arm["final_step"],
            "steps": steps,
            "curve": losses,
        }
        fitted, observed, ci_lo, ci_hi = residual_bootstrap_ci(
            steps,
            losses,
            min_step=1000,
            final_step=arm["final_step"],
        )
        bootstrap_payload["arms"].append(
            {
                "key": arm["key"],
                "label": arm["label"],
                "wandb_path": arm["wandb_path"],
                "final_step": arm["final_step"],
                "fitted_final": round(fitted, 4),
                "observed": round(observed, 4),
                "ci_lo": round(ci_lo, 4),
                "ci_hi": round(ci_hi, 4),
            }
        )
        print(
            f"{arm['label']:18} fitted={fitted:.4f} observed={observed:.4f} "
            f"CI=[{ci_lo:.4f}, {ci_hi:.4f}]"
        )

    CURVES_PATH.write_text(json.dumps(curves_payload, indent=2) + "\n", encoding="utf-8")
    BOOTSTRAP_PATH.write_text(json.dumps(bootstrap_payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {CURVES_PATH}")
    print(f"Wrote {BOOTSTRAP_PATH}")


if __name__ == "__main__":
    main()
