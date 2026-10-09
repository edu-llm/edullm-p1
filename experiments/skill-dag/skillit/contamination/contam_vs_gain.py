#!/usr/bin/env python3
"""Do the Skill-It arms' gains over LightGBM fall on the benchmarks whose contamination
exposure they raised?

Per benchmark b and domain d, the reservoir scan (``mixlaw/contamination/
results_olmo127b-reservoir.json``) gives the number of eval-item fields (stem + gold)
found in d. An arm's exposure density to benchmark b is

    E_b = sum_d w_d * hits_{b,d} / words_d      (matched item fields per training word)

with w the arm's time-weighted domain weights over its Skill-It segments, computed the
way ``trajectory_exposure.py`` does. The ratio arm / LightGBM per benchmark is compared
with the arm's per-benchmark bpb change against LightGBM (mean over that benchmark's
labels, from ``mixlaw/dynamic_arm_labels_results.json``) by Spearman rank correlation
over the 10 benchmarks. Contamination-driven gains would give a negative correlation
(more added exposure, lower bpb).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
SKILL_DAG = HERE.parents[1]
scan = json.loads((SKILL_DAG / "mixlaw/contamination/results_olmo127b-reservoir.json").read_text(encoding="utf-8"))
perf = json.loads((SKILL_DAG / "mixlaw/dynamic_arm_labels_results.json").read_text(encoding="utf-8"))
DOMS = list(scan["domains"])
BENCH = list(scan["totals"]["per_benchmark"])
FINAL = 2384


def trajectory(name: str):
    """Time-weighted domain weights of an arm and its starting weights."""
    rows = [json.loads(line) for line in (HERE / f"skillit_updates_{name}.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    steps = [r["step"] for r in rows] + [FINAL]
    w = np.zeros(len(DOMS))
    for r, s0, s1 in zip(rows, steps[:-1], steps[1:]):
        w += (s1 - s0) * np.array([r["p_after"][d] for d in DOMS])
    return w / FINAL, np.array([rows[0]["p_after"][d] for d in DOMS])


def density(w: np.ndarray, bench: str) -> float:
    total = 0.0
    for i, d in enumerate(DOMS):
        pb = scan["domains"][d]["per_benchmark"][bench]
        total += w[i] * (pb["stem_found"] + pb["gold_found"]) / scan["domains"][d]["words_scanned"]
    return total


def bench_of(label: str) -> str:
    for b in sorted(BENCH, key=len, reverse=True):
        if label.startswith(b):
            return b
    raise KeyError(label)


def main() -> None:
    w_probe, w_lgb = trajectory("offline-probe")
    w_deriv, _ = trajectory("online-derivative")
    labels_by_bench: dict[str, list[str]] = {}
    for label in perf["labels"]:
        labels_by_bench.setdefault(bench_of(label), []).append(label)

    overall = scan["totals"]["per_benchmark"]
    rows = []
    print(f"{'benchmark':12s} {'items found':>11s} {'LGB dens':>9s} | {'probe x':>7s} {'probe dbpb':>10s} | {'deriv x':>7s} {'deriv dbpb':>10s}")
    for b in BENCH:
        e_lgb = density(w_lgb, b)
        rp, rd = density(w_probe, b) / e_lgb, density(w_deriv, b) / e_lgb
        labels = labels_by_bench[b]
        dp = np.mean([perf["labels"][lab]["skillit-probe|lightgbm-l40s"][0] for lab in labels])
        dd = np.mean([perf["labels"][lab]["skillit-derivative|lightgbm-l40s"][0] for lab in labels])
        ov = overall[b]
        found = (ov["stem_found"] + ov["gold_found"]) / (ov["stem_assessable"] + ov["gold_assessable"])
        rows.append((b, found, e_lgb, rp, dp, rd, dd))
        print(f"{b:12s} {found:11.3f} {e_lgb * 1e9:9.2f} | {rp:7.3f} {dp:+10.4f} | {rd:7.3f} {dd:+10.4f}")

    for arm, ratio_i, delta_i in (("probe", 3, 4), ("derivative", 5, 6)):
        ratio = [r[ratio_i] for r in rows]
        delta = [r[delta_i] for r in rows]
        rho, p = spearmanr(ratio, delta)
        print(f"\n{arm}: Spearman(exposure ratio vs LightGBM, bpb change vs LightGBM) over {len(rows)} benchmarks: "
              f"rho={rho:+.2f} (p={p:.2f})")
        raised = [r[0] for r in rows if r[ratio_i] > 1.0]
        print(f"   benchmarks with raised exposure: {raised}")
        up = [r[delta_i] for r in rows if r[ratio_i] > 1]
        down = [r[delta_i] for r in rows if r[ratio_i] <= 1]
        print(f"   mean bpb change, raised: {np.mean(up):+.4f}; not raised: "
              f"{np.mean(down) if down else float('nan'):+.4f}")
        rho2, p2 = spearmanr([r[1] for r in rows], delta)
        print(f"   Spearman(overall item contamination, bpb change): rho={rho2:+.2f} (p={p2:.2f})")
    print("\nweights (time-avg):", {d: (round(a, 3), round(b, 3), round(c, 3))
                                   for d, a, b, c in zip(DOMS, w_lgb, w_probe, w_deriv)})


if __name__ == "__main__":
    main()
