"""Figure II: Task loss curves for the dynamic-reweighting arms (Probe, Derivative)
compared to the static control (Data Mixing Laws paper mix / regmix-control).

Regenerated using the completed FarmShare probe rerun (job 1730368) and the
existing derivative/control runs. The stale third arm ("Probe with optimized
start", from the superseded RunPod-era run set) has been dropped.

The curve data lives alongside this script in `figure_ii_curves.py` so the
figure is reproducible from the repository rather than from a scratch
directory.

Two revisions to the originally published panel:

* The palette no longer pairs red against green. Red-green is the most
  common form of color vision deficiency, and the two arms that the figure
  most wants the reader to tell apart (the LightGBM static mixture and the
  derivative arm) were exactly that pair. Each series now also carries its
  own dash pattern and marker, so the panel survives grayscale printing.
* The control is drawn as the envelope of its two seeds rather than as a
  bare average line. That band is the only uncertainty in this figure that
  the data actually supports: the control was run at two seeds (6198 and
  12345), while every reweighting arm was run once. The band therefore sets
  the scale against which the single-seed arms should be read, and no band
  is drawn for those arms because none was measured.
"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figure_ii_curves import CURVES, OLMO_SEED6198, OLMO_SEED12345  # noqa: E402

SKILLIT = Path(__file__).resolve().parent
# Committed alongside the experiment; the Overleaf tree gets a copy too when
# it is present, so the paper and the repository never diverge.
OUT_DIRS = [SKILLIT / "figures", SKILLIT.parents[2] / "p1-whitepaper-overleaf" / "figures"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
})

# Colorblind-safe: grey control, then blue / orange / purple. No red-green pair.
# Distinct dash patterns and markers so the series also separate in grayscale.
SERIES = [
    ("olmo_average", "Olmo-mix-1124 average (control)", "#6B7280", (0, (5, 2)), "s"),
    ("lgbm_control", "LightGBM static", "#7C3AED", (0, (1, 1.6)), "D"),
    ("probe", "Probe", "#2563EB", "-", "o"),
    ("derivative", "Derivative", "#D97706", (0, (6, 1.6, 1.4, 1.6)), "^"),
]

CONTROL_BAND_COLOR = "#9CA3AF"


def control_band(min_step: int) -> tuple[list[int], list[float], list[float]]:
    """Per-step envelope of the two control seeds, on their shared steps."""
    a = dict(zip(OLMO_SEED6198["steps"], OLMO_SEED6198["curve"]))
    b = dict(zip(OLMO_SEED12345["steps"], OLMO_SEED12345["curve"]))
    steps = [s for s in sorted(set(a) & set(b)) if s >= min_step]
    lo = [min(a[s], b[s]) for s in steps]
    hi = [max(a[s], b[s]) for s in steps]
    return steps, lo, hi

fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=150)
fig.suptitle("Mid-training reweighting does not beat a static mixture", fontsize=19.5, fontweight="bold", y=0.97)

MIN_STEP = 700
# The LightGBM static run has a stray off-cadence eval at step 2375 (125
# steps before the final-step eval at 2384) that no other arm has; drop it
# so the curve doesn't show a spurious extra point right before the end.
DROP_STEPS = {"lgbm_control": {2375}}

band_steps, band_lo, band_hi = control_band(MIN_STEP)
ax.fill_between(band_steps, band_lo, band_hi, color=CONTROL_BAND_COLOR, alpha=0.38,
                linewidth=0, zorder=1,
                label="Control seed range (n=2)")

for key, label, color, ls, marker in SERIES:
    d = CURVES[key]
    drop = DROP_STEPS.get(key, set())
    steps = [s for s in d["steps"] if s >= MIN_STEP and s not in drop]
    vals = [v for s, v in zip(d["steps"], d["curve"]) if s >= MIN_STEP and s not in drop]
    ax.plot(steps, vals, color=color, linestyle=ls, marker=marker,
            markersize=3.5, linewidth=1.8, label=label, zorder=3)

ax.set_xlabel("Training step", fontsize=15)
ax.set_ylabel("Validation macro bits-per-byte\n(20-task OLMES avg, \u2193 lower is better)", fontsize=15)
ax.tick_params(labelsize=15)
ax.grid(True, color="0.85", linewidth=0.6)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
ax.legend(loc="lower left", fontsize=10.5, frameon=True, framealpha=0.92,
          edgecolor="none", facecolor="white", borderpad=0.5, labelspacing=0.35)
ax.set_xlim(MIN_STEP, 2450)

# Inset: final steps, rescaled
# (sized/positioned to leave room for the larger tick labels and title
# below without colliding with the main plot's right/top edges)
axins = ax.inset_axes([0.52, 0.52, 0.46, 0.46])
ins_steps, ins_lo, ins_hi = control_band(1900)
axins.fill_between(ins_steps, ins_lo, ins_hi, color=CONTROL_BAND_COLOR, alpha=0.38,
                   linewidth=0, zorder=1)
for key, label, color, ls, marker in SERIES:
    d = CURVES[key]
    drop = DROP_STEPS.get(key, set())
    steps = [s for s in d["steps"] if s >= 1900 and s not in drop]
    vals = [v for s, v in zip(d["steps"], d["curve"]) if s >= 1900 and s not in drop]
    axins.plot(steps, vals, color=color, linestyle=ls, marker=marker, markersize=3,
               linewidth=1.5, zorder=3)
axins.set_title("final steps, rescaled", fontsize=10.5, style="italic")
axins.tick_params(labelsize=9, length=2)
axins.grid(True, color="0.85", linewidth=0.5)
for spine in ("top", "right"):
    axins.spines[spine].set_visible(False)
axins.yaxis.set_major_locator(mticker.MaxNLocator(5))

for out_dir in OUT_DIRS:
    if not out_dir.parent.exists():
        continue
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "figure_ii_dynamic_reweighting.png"
    pdf_path = out_dir / "figure_ii_dynamic_reweighting.pdf"
    fig.savefig(png_path, dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(pdf_path, facecolor="white", bbox_inches="tight")
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")
plt.close(fig)
