#!/usr/bin/env python3
"""Side-by-side stacked-area domain-weight-over-time chart: Probe vs Derivative.

Each panel shows the cumulative domain-weight mixture Skill-It was training on
at each point in the run, as a discrete (step-function) stacked area so it's
clear the mixture only changes at update boundaries. Probe data reflects the
FarmShare rerun of arm 0 (job 1730368, run "probe" in eduLLM/skillit); the
Derivative panel uses arm 1's update history unchanged.

Update data is read directly from each arm's `skillit_updates.jsonl` (written
by the Skill-It controller under `<run_dir>/runs/<arm>/progress/`); paste in
new `p_after` snapshots there if this is ever regenerated for a different run.

Revision to the originally published panel: in a 0-1 stacked area the three
smallest domains (wiki, AlgebraicStack, StarCoder) are ~1% slivers at the
bottom of the stack, yet that is where much of the derivative arm's
reweighting actually happens -- wiki rises roughly fourfold while the stack
looks static. The stacked panels are unchanged in construction; they now
carry thin white separators so adjacent slivers can be told apart, and a
second row plots those three domains on their own scale.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SKILLIT = Path(__file__).resolve().parent
OUT_DIRS = [SKILLIT / "figures", SKILLIT.parents[2] / "p1-whitepaper-overleaf" / "figures"]
FINAL_STEP = 2384

# Bottom-to-top stacking order, matching the original figure.
DOMAINS = ["wiki", "algebraic-stack", "open-web-math", "pes2o", "starcoder", "arxiv", "dclm"]
LABELS = {
    "wiki": "Wiki", "algebraic-stack": "AlgebraicStack", "open-web-math": "OpenWebMath",
    "pes2o": "pes2o", "starcoder": "StarCoder", "arxiv": "arXiv", "dclm": "DCLM",
}
COLORS = {
    "wiki": "#6B7280", "algebraic-stack": "#7C2D12", "open-web-math": "#A855F7",
    "pes2o": "#EF4444", "starcoder": "#14B8A6", "arxiv": "#F59E0B", "dclm": "#2563EB",
}
LEGEND_ORDER = ["wiki", "algebraic-stack", "open-web-math", "pes2o", "starcoder", "arxiv", "dclm"]
# The three domains that are invisible in a 0-1 stack; broken out below it.
SMALL_DOMAINS = ["wiki", "algebraic-stack", "starcoder"]
SMALL_MARKERS = {"wiki": "o", "algebraic-stack": "s", "starcoder": "^"}

# Probe: FarmShare rerun of arm 0, job 1730368 (skillit_updates.jsonl p_after).
PROBE_UPDATES = [
    (0,    {"dclm": 0.5528505096102141, "arxiv": 0.21178482308348512, "starcoder": 0.08723935826031831, "pes2o": 0.08163337756774411, "open-web-math": 0.041786239404329344, "algebraic-stack": 0.013571059505826967, "wiki": 0.01113463256808213}),
    (500,  {"dclm": 0.6459461253946372, "arxiv": 0.16781238139729543, "starcoder": 0.05665913498302511, "pes2o": 0.07727008631788466, "open-web-math": 0.03104617306715885, "algebraic-stack": 0.010195690011512803, "wiki": 0.011070408828485898}),
    (875,  {"dclm": 0.7198701613362318, "arxiv": 0.1312047624368636, "starcoder": 0.036709905129594864, "pes2o": 0.07116631617229063, "open-web-math": 0.022842108444754147, "algebraic-stack": 0.0075794641199737394, "wiki": 0.010627282360291075}),
    (1250, {"dclm": 0.7794252168264567, "arxiv": 0.10092269002172854, "starcoder": 0.023489361157397377, "pes2o": 0.06409198604905098, "open-web-math": 0.016554981127748933, "algebraic-stack": 0.005548523013975088, "wiki": 0.00996724180364266}),
    (1625, {"dclm": 0.8268532718926198, "arxiv": 0.07654112577077739, "starcoder": 0.014866343692070724, "pes2o": 0.056702142661858995, "open-web-math": 0.011842439816005683, "algebraic-stack": 0.004008018883345155, "wiki": 0.009186657283322052}),
    (2000, {"dclm": 0.8640046143237808, "arxiv": 0.05748684666307869, "starcoder": 0.009327972256629744, "pes2o": 0.049564753592122174, "open-web-math": 0.00839236464681183, "algebraic-stack": 0.0028678558874201884, "wiki": 0.008355592630156597}),
]

# Derivative: arm 1, unaffected by the probe-arm defects that required a rerun.
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
    "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
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
                            edgecolor="white", linewidth=0.8)
        handles[dom] = h
        prev_cum = cum

    ax.set_title(title, fontsize=36, fontweight="bold", pad=14)
    ax.set_xlabel("Training step", fontsize=30, fontweight="bold")
    ax.set_xlim(0, FINAL_STEP)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=22)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return handles


def draw_small_panel(ax, updates, *, ymax) -> None:
    """The three smallest domains on their own scale, same step semantics."""
    steps = [s for s, _ in updates] + [FINAL_STEP]
    weights = [w for _, w in updates] + [updates[-1][1]]

    for dom in SMALL_DOMAINS:
        vals = [w[dom] for w in weights]
        ax.step(steps, vals, where="post", color=COLORS[dom], linewidth=3.0)
        # Markers only at the real update boundaries, not the final-step repeat.
        ax.plot([s for s, _ in updates], [w[dom] for _, w in updates],
                linestyle="none", marker=SMALL_MARKERS[dom], markersize=9,
                color=COLORS[dom], markeredgecolor="white", markeredgewidth=1.2)

    ax.set_xlim(0, FINAL_STEP)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Training step", fontsize=30, fontweight="bold")
    ax.tick_params(labelsize=22)
    ax.grid(axis="y", linestyle=":", linewidth=0.9, color="#c9c9c9")
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def main() -> None:
    fig, axes = plt.subplots(
        2, 2, figsize=(20, 13), dpi=150,
        gridspec_kw={"height_ratios": [3.0, 1.45]},
    )
    draw_panel(axes[0][0], PROBE_UPDATES, title="Probe")
    handles = draw_panel(axes[0][1], DERIVATIVE_UPDATES, title="Derivative")
    axes[0][0].set_ylabel("Cumulative domain weight", fontsize=30, fontweight="bold")
    for ax in axes[0]:
        ax.set_xlabel("")

    # Shared scale across the two lower panels so the arms stay comparable.
    small_max = max(
        w[dom]
        for updates in (PROBE_UPDATES, DERIVATIVE_UPDATES)
        for _, w in updates
        for dom in SMALL_DOMAINS
    )
    ymax = small_max * 1.15
    draw_small_panel(axes[1][0], PROBE_UPDATES, ymax=ymax)
    draw_small_panel(axes[1][1], DERIVATIVE_UPDATES, ymax=ymax)
    axes[1][0].set_ylabel("Three smallest\ndomains, rescaled", fontsize=24, fontweight="bold")

    ordered_handles = [handles[d] for d in LEGEND_ORDER]
    ordered_labels = [LABELS[d] for d in LEGEND_ORDER]
    fig.legend(ordered_handles, ordered_labels, loc="lower center", ncol=4, fontsize=24,
               frameon=False, bbox_to_anchor=(0.5, -0.055))

    fig.subplots_adjust(bottom=0.155, top=0.94, hspace=0.32, wspace=0.15)

    for out_dir in OUT_DIRS:
        if not out_dir.parent.exists():
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            path = out_dir / f"domain_weights_probe_vs_derivative.{ext}"
            fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight")
            print(f"Wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
