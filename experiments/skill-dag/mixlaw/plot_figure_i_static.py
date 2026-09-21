"""Figure I: task loss curves for the four static-mixture validation arms.

Two revisions to the originally published panel:

* The two fixed baselines (the Olmo-mix-1124 control and the Data Mixing
  Laws paper mixture) were both drawn as grey dashed lines with nearly the
  same value, which made them hard to tell apart at print size. They now
  differ in dash pattern, marker and lightness, so each is identifiable
  without relying on the legend order or on color.
* The control is drawn as the envelope of its two seeds (6198 and 12345)
  rather than as a bare average line. That band is the only uncertainty
  this figure can honestly show: the control ran at two seeds, while each
  fitted mixture ran once. It sets the scale a reader should apply to the
  single-seed arms, and deliberately no band is drawn for those arms.

Data comes from `figure_i_wandb_curves.json`, written by
`figure_i_control_seed_average.py`.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
COL_CONTROL = "#4b5563"
COL_CONTROL_BAND = "#9ca3af"
COL_DML_PAPER = "#adb3bd"
COL_MIXLAW = "#2563eb"
COL_LIGHTGBM = "#ea580c"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 12,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.9,
})

MIXLAW = Path(__file__).resolve().parent
OUT_DIRS = [MIXLAW / "figures", MIXLAW.parents[2] / "p1-whitepaper-overleaf" / "figures"]

wb = json.loads((MIXLAW / "figure_i_wandb_curves.json").read_text(encoding="utf-8"))

def series(key):
    pts = wb[key]
    s = np.array([p[0] for p in pts], dtype=float)
    y = np.array([p[1] for p in pts], dtype=float)
    keep = s >= 700
    return s[keep], y[keep]

def edge_interp(steps, y, xmin):
    """Clip to steps >= xmin, prepending a linearly-interpolated point at
    exactly x = xmin (using the nearest point just outside the window) so
    the plotted line reaches the left edge at the correct slope instead of
    starting wherever the first in-window checkpoint happens to be."""
    before = steps < xmin
    after = steps >= xmin
    if not before.any() or not after.any() or steps[after][0] == xmin:
        return steps[after], y[after]
    s0, v0 = steps[before][-1], y[before][-1]
    s1, v1 = steps[after][0], y[after][0]
    v_edge = v0 + (xmin - s0) / (s1 - s0) * (v1 - v0)
    return np.concatenate([[xmin], steps[after]]), np.concatenate([[v_edge], y[after]])

def control_band(xmin):
    """Per-step min/max envelope of the two control seeds."""
    a = {p[0]: p[1] for p in wb["olmo_mix_1124_seed6198"]}
    b = {p[0]: p[1] for p in wb["olmo_mix_1124_seed12345"]}
    st = np.array([s for s in sorted(set(a) & set(b)) if s >= xmin], dtype=float)
    lo = np.array([min(a[s], b[s]) for s in st], dtype=float)
    hi = np.array([max(a[s], b[s]) for s in st], dtype=float)
    return st, lo, hi


s_ctrl, y_ctrl = series("olmo_mix_1124")
s_ml, y_ml = series("mixlaw_optimum")
s_lgb, y_lgb = series("lightgbm")
s_dml, y_dml = series("dml_paper_mix01")

FINAL_STEP = 2384

fig, ax = plt.subplots(figsize=(9.2, 6.2), dpi=200)

bs, blo, bhi = control_band(650)
ax.fill_between(bs, blo, bhi, color=COL_CONTROL_BAND, alpha=0.40, linewidth=0, zorder=1)

ax.plot(s_ctrl, y_ctrl, color=COL_CONTROL, linestyle=(0, (5, 2)), linewidth=2.0, marker="s", markersize=4, zorder=3)
ax.plot(s_dml, y_dml, color=COL_DML_PAPER, linestyle=(0, (1, 1.8)), linewidth=2.2, marker="^", markersize=4.5, zorder=3)
ax.plot(s_lgb, y_lgb, color=COL_LIGHTGBM, linestyle="-", linewidth=2.2, marker="D", markersize=4, zorder=4)
ax.plot(s_ml, y_ml, color=COL_MIXLAW, linestyle="-", linewidth=2.4, marker="o", markersize=4, zorder=5)

ax.set_xlabel("Training step", labelpad=8)
ax.set_ylabel("Validation macro bits-per-byte\n(20-task OLMES avg, $\\downarrow$ lower is better)")
ax.set_title("Fitted static domain mixtures beat both fixed baselines", fontsize=15, fontweight="bold", pad=12)
ax.grid(True, linestyle=":", linewidth=0.7, color="#c9c9c9", alpha=0.9)
ax.set_axisbelow(True)
ax.set_xlim(650, FINAL_STEP + 60)
ax.set_ylim(1.585, 1.875)

legend_handles = [
    Patch(facecolor=COL_CONTROL_BAND, alpha=0.40, edgecolor="none",
          label="Control seed range (n=2)"),
    Line2D([0], [0], color=COL_CONTROL, linestyle=(0, (5, 2)), marker="s", markersize=6, linewidth=2.0,
           label="Olmo-mix-1124 (control, 2-seed avg)"),
    Line2D([0], [0], color=COL_DML_PAPER, linestyle=(0, (1, 1.8)), marker="^", markersize=6.5, linewidth=2.2,
           label="Data Mixing Laws paper mix"),
    Line2D([0], [0], color=COL_LIGHTGBM, linestyle="-", marker="D", markersize=6, linewidth=2.2,
           label="LightGBM fit"),
    Line2D([0], [0], color=COL_MIXLAW, linestyle="-", marker="o", markersize=6, linewidth=2.4,
           label="MixLaw fit (ours, best)"),
]
ax.legend(handles=legend_handles, loc="lower left", frameon=False, fontsize=10, labelspacing=0.7)

INSET_XMIN = 1700

axins = fig.add_axes([0.60, 0.505, 0.34, 0.255])
zoom_mask = lambda s: s >= INSET_XMIN
ibs, iblo, ibhi = control_band(INSET_XMIN)
axins.fill_between(ibs, iblo, ibhi, color=COL_CONTROL_BAND, alpha=0.40, linewidth=0, zorder=1)
for s, y, c, ls, mk in [
    (s_ctrl, y_ctrl, COL_CONTROL, (0, (5, 2)), "s"),
    (s_dml, y_dml, COL_DML_PAPER, (0, (1, 1.8)), "^"),
    (s_lgb, y_lgb, COL_LIGHTGBM, "-", "D"),
    (s_ml, y_ml, COL_MIXLAW, "-", "o"),
]:
    xs, ys = edge_interp(s, y, INSET_XMIN)
    axins.plot(xs, ys, color=c, linestyle=ls, linewidth=1.8, zorder=3)
    m = zoom_mask(s)
    axins.plot(s[m], y[m], color=c, linestyle="none", marker=mk, markersize=3.3, zorder=3)
axins.set_title("final steps, rescaled", fontsize=9.3, style="italic", pad=2)
axins.tick_params(labelsize=8)
axins.grid(True, linestyle=":", linewidth=0.6, color="#d5d5d5")
inset_right = FINAL_STEP + 130
axins.set_xlim(INSET_XMIN, inset_right)
axins.set_ylim(1.598, 1.67)

# Compare the MixLaw optimum against the Olmo-mix-1124 control (not the DML-paper mix).
gap = y_ctrl[-1] - y_ml[-1]
ax_x = FINAL_STEP + 20
axins.annotate("", xy=(ax_x, y_ml[-1]), xytext=(ax_x, y_ctrl[-1]),
               arrowprops=dict(arrowstyle="<->", color="#222", lw=1.0))
axins.text(ax_x + 3, (y_ml[-1] + y_ctrl[-1]) / 2, f"{gap:.3f}", fontsize=8, va="center")

fig.subplots_adjust(left=0.115, right=0.97, top=0.90, bottom=0.13)
for out_dir in OUT_DIRS:
    if not out_dir.parent.exists():
        continue
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = out_dir / f"figure_i_static_mixtures.{ext}"
        fig.savefig(path, facecolor="white")
        print(f"Wrote {path}")
