#!/usr/bin/env python3
"""Side-by-side stacked-area domain-weight-over-time chart: Probe vs Derivative.

Each panel shows the cumulative domain-weight mixture Skill-It was training on
at each point in the run, as a discrete (step-function) stacked area so it's
clear the mixture only changes at update boundaries. Probe data reflects the
FarmShare arm 0 (job 1771665, run "probe-lgbref-farmshare-1771665" in eduLLM/skillit); the
Derivative panel uses arm 1's update history unchanged.

The Probe update data is read from `contamination/skillit_updates_offline-probe.jsonl`
(`step` and `p_after` copied from the arm's own `skillit_updates.jsonl` under
`<run_dir>/runs/<arm>/progress/`); the Derivative panel's snapshots are pasted below.

Revisions to the originally published panel: the bands now carry thin white
separators, so the three ~1% domains at the bottom of the stack (wiki,
AlgebraicStack, StarCoder) can at least be told apart from each other; their
exact weights are in the appendix tables. The dead vertical space between the
panels and the legend has also been removed.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SKILLIT = Path(__file__).resolve().parent
OUT_DIRS = [SKILLIT / "figures"]
FINAL_STEP = 2384

# Bottom-to-top stacking order, matching the original figure.
DOMAINS = ["wiki", "algebraic-stack", "open-web-math", "pes2o", "starcoder", "arxiv", "dclm"]
LABELS = {
    "wiki": "Wikipedia", "algebraic-stack": "AlgebraicStack", "open-web-math": "OpenWebMath",
    "pes2o": "pes2o", "starcoder": "StarCoder", "arxiv": "arXiv", "dclm": "DCLM",
}
COLORS = {
    "wiki": "#6B7280", "algebraic-stack": "#7C2D12", "open-web-math": "#A855F7",
    "pes2o": "#EF4444", "starcoder": "#14B8A6", "arxiv": "#F59E0B", "dclm": "#2563EB",
}
LEGEND_ORDER = ["wiki", "algebraic-stack", "open-web-math", "pes2o", "starcoder", "arxiv", "dclm"]


# Probe: FarmShare arm 0, job 1771665 (skillit_updates.jsonl p_after), read from the committed
# per-update log that the contamination analysis also uses.
def _load_updates(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [(int(r["step"]), r["p_after"]) for r in rows]


PROBE_UPDATES = _load_updates(SKILLIT / "contamination" / "skillit_updates_offline-probe.jsonl")

# Derivative: arm 1, job 1728144 (run "derivative" in eduLLM/skillit).
DERIVATIVE_UPDATES = [
    (1,    {"dclm": 0.5528504848, "arxiv": 0.2117848247, "starcoder": 0.0872393548, "pes2o": 0.0816333741, "open-web-math": 0.0417862386, "algebraic-stack": 0.0135710593, "wiki": 0.0111346329}),
    (501,  {"dclm": 0.5561979413, "arxiv": 0.1995353103, "starcoder": 0.0821934864, "pes2o": 0.0864803717, "open-web-math": 0.0471215174, "algebraic-stack": 0.0127861174, "wiki": 0.0156852882}),
    (876,  {"dclm": 0.5572664142, "arxiv": 0.1886009127, "starcoder": 0.0776893422, "pes2o": 0.0910638496, "open-web-math": 0.0520632602, "algebraic-stack": 0.0120854471, "wiki": 0.0212307665}),
    (1251, {"dclm": 0.5562109351, "arxiv": 0.1782047153, "starcoder": 0.07340689, "pes2o": 0.0952695683, "open-web-math": 0.057228189, "algebraic-stack": 0.0114192637, "wiki": 0.0282604527}),
    (1626, {"dclm": 0.5533024073, "arxiv": 0.1683511883, "starcoder": 0.0693479776, "pes2o": 0.0991969332, "open-web-math": 0.0621750467, "algebraic-stack": 0.0107878549, "wiki": 0.0368385985}),
    (2001, {"dclm": 0.5486896634, "arxiv": 0.1588883549, "starcoder": 0.0654500127, "pes2o": 0.1027220041, "open-web-math": 0.0668944418, "algebraic-stack": 0.0101814819, "wiki": 0.0471740402}),
]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 12,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.9,
})


def draw_panel(ax, updates, *, title):
    steps = [s for s, _ in updates] + [FINAL_STEP]
    weights = [w for _, w in updates] + [updates[-1][1]]

    prev_cum = np.zeros(len(steps))
    handles = {}
    for dom in DOMAINS:
        vals = np.array([w[dom] for w in weights])
        cum = prev_cum + vals
        h = ax.fill_between(steps, prev_cum, cum, step="post", color=COLORS[dom],
                            edgecolor="white", linewidth=0.5)
        handles[dom] = h
        prev_cum = cum

    ax.set_title(title, fontsize=15, fontweight="bold", pad=10)
    ax.set_xlabel("Training step", labelpad=8)
    ax.set_xlim(0, FINAL_STEP)
    ax.set_ylim(0, 1)
    return handles


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.6), dpi=200)
    draw_panel(axes[0], PROBE_UPDATES, title="Probe")
    handles = draw_panel(axes[1], DERIVATIVE_UPDATES, title="Derivative")
    axes[0].set_ylabel("Cumulative domain weight", labelpad=8)

    # One row, sitting just under the x-axis labels rather than a third of a
    # figure height below them.
    ordered_handles = [handles[d] for d in LEGEND_ORDER]
    ordered_labels = [LABELS[d] for d in LEGEND_ORDER]
    fig.legend(ordered_handles, ordered_labels, loc="lower center", ncol=7, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, 0.005), columnspacing=1.1,
               handlelength=1.4, handletextpad=0.4)

    fig.subplots_adjust(bottom=0.22, top=0.90, wspace=0.16)

    for out_dir in OUT_DIRS:
        if not out_dir.parent.exists():
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        for ext in ("png",):
            path = out_dir / f"domain_weights_probe_vs_derivative.{ext}"
            fig.savefig(path, facecolor="white")
            print(f"Wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
