#!/usr/bin/env python3
"""Targeted / never-targeted / per-label analysis for the two Skill-It arms.

Same power-law fit and alpha-free residual bootstrap as ``fit_and_bootstrap_370m.py``
and the same per-arm random streams (``SeedSequence(0).spawn(8)[bootstrap_stream]``).
For each label subset (all 20 labels, the 12 targeted labels, the 8 never-targeted
labels) and for each single label, the subset's labels are averaged at every eval step,
fitted, and bootstrapped. Each dynamic arm (and the LightGBM static run) is compared
with the control average of the two Olmo-mix-1124 seeds, and the dynamic arms are also
compared with the LightGBM static run they start from.

The p-value is one-sided ("the arm beats the reference") and is ``None`` when the
arm's fitted difference is not negative. The seed spread (s69 - s42) is reported
separately; its p-value in the paper is two-sided.

Per-label curves come from the arms' W&B runs (``pull_370m_curves.py``), so this needs
W&B access; the all-20 row reproduces ``skill_dag_370m_bootstrap_results.json``.

usage:
    python dynamic_arm_labels.py                 # compute, write dynamic_arm_labels_results.json
    python dynamic_arm_labels.py --report-only   # print the tables from the written file
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pull_370m_curves as P  # noqa: E402
from fit_and_bootstrap_370m import ci, fit_and_bootstrap  # noqa: E402

KEYS = ["olmo-mix-1124-s42", "olmo-mix-1124-s69", "lightgbm-l40s",
        "skillit-probe", "skillit-derivative"]
N_BOOT, FINAL = 200_000, 2384
DEFAULT_OUT = HERE / "dynamic_arm_labels_results.json"

COLUMNS = [("skillit-probe|ctrl", "Pr-ctl"), ("skillit-derivative|ctrl", "Dv-ctl"),
           ("skillit-probe|lightgbm-l40s", "Pr-LGB"), ("skillit-derivative|lightgbm-l40s", "Dv-LGB"),
           ("lightgbm-l40s|ctrl", "LGB-ctl")]


def one_sided(d: np.ndarray) -> float:
    return max(float((d >= 0).mean()), 1.0 / d.size)


def run_subset(arms: dict, labels: list[str], streams) -> dict:
    fin, fit = {}, {}
    for k in KEYS:
        v = [np.mean([arms[k]["_per_label"][s][lab] for lab in labels]) for s in P.EVAL_LADDER]
        fit[k], fin[k] = fit_and_bootstrap(P.EVAL_LADDER, v, final_step=FINAL, n_boot=N_BOOT,
                                           seed=streams[P.RUNS[k][4]])
    fin["ctrl"] = 0.5 * (fin["olmo-mix-1124-s42"] + fin["olmo-mix-1124-s69"])
    fit["ctrl"] = 0.5 * (fit["olmo-mix-1124-s42"] + fit["olmo-mix-1124-s69"])
    out = {"ctrl_fit": fit["ctrl"], "lgbm_fit": fit["lightgbm-l40s"]}
    for arm in ("skillit-probe", "skillit-derivative", "lightgbm-l40s"):
        for ref in ("ctrl", "lightgbm-l40s"):
            if arm == ref:
                continue
            d = fin[arm] - fin[ref]
            lo, hi = ci(d)
            delta = fit[arm] - fit[ref]
            out[f"{arm}|{ref}"] = (delta, lo, hi, one_sided(d) if delta < 0 else None)
    sd = fin["olmo-mix-1124-s69"] - fin["olmo-mix-1124-s42"]
    lo, hi = ci(sd)
    out["seed"] = (fit["olmo-mix-1124-s69"] - fit["olmo-mix-1124-s42"], lo, hi)
    return out


def compute(out_path: Path) -> dict:
    import wandb

    streams = np.random.SeedSequence(0).spawn(P.BOOTSTRAP_STREAM_COUNT)
    api = wandb.Api()
    arms = {k: P.pull_arm(api, k) for k in KEYS}
    results = {"subsets": {}, "labels": {}}
    for name, labels in (("all20", P.ALL_LABELS), ("targeted12", P.TARGETED_LABELS),
                         ("nevertargeted8", P.NEVER_TARGETED_LABELS)):
        results["subsets"][name] = run_subset(arms, labels, streams)
        print("subset", name, "done", flush=True)
    for lab in P.ALL_LABELS:
        results["labels"][lab] = run_subset(arms, [lab], streams)
        print("label", lab, "done", flush=True)
    out_path.write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")
    print("wrote", out_path)
    return results


def cell(c) -> str:
    d, lo, hi, p = c
    sig = "*" if (hi < 0 or lo > 0) else " "
    ps = "  -   " if p is None else (f"{p:.4f}" if p >= 1e-4 else "<5e-6 " if p <= 5e-6 else f"{p:.0e}")
    return f"{d:+.4f}[{lo:+.3f},{hi:+.3f}]{sig}p{ps}"


def row(name: str, r: dict) -> str:
    s = f"{name:22s} ctl {r['ctrl_fit']:.3f} " + " | ".join(cell(r[k]) for k, _ in COLUMNS)
    sd = r["seed"]
    return s + f" | seed {sd[0]:+.4f}[{sd[1]:+.3f},{sd[2]:+.3f}]"


def report(results: dict) -> None:
    print(" " * 32 + " | ".join(f"{n:^32s}" for _, n in COLUMNS))
    for k, r in results["subsets"].items():
        print(row(k, r))
    print()
    for k, r in results["labels"].items():
        print(row(k.replace("_rc_5shot_bpb", ""), r))
    print("\nsign counts (negative = arm better than the reference):")
    for k, n in COLUMNS:
        better = sum(1 for r in results["labels"].values() if r[k][0] < 0)
        sb = [lab.replace("_rc_5shot_bpb", "") for lab, r in results["labels"].items() if r[k][2] < 0]
        sw = [lab.replace("_rc_5shot_bpb", "") for lab, r in results["labels"].items() if r[k][1] > 0]
        print(f"  {n}: better on {better}/20; sig better {sb}; sig worse {sw}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--report-only", action="store_true", help="print tables from an existing --out file")
    args = ap.parse_args()
    results = json.loads(args.out.read_text(encoding="utf-8")) if args.report_only else compute(args.out)
    report(results)


if __name__ == "__main__":
    main()
