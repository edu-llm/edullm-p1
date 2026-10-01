"""Figure III: does the MixLaw mixture reach the control's performance sooner?

The figure asks how many training steps the MixLaw 1%-floor mixture needs to
reach the Olmo-mix-1124 control's final loss, and conversely how long the
control needs to reach MixLaw's. Both answers come from one estimator, the same
one behind Table II: the power law ``y = a + b / step**alpha`` fitted on steps
>= 1000 over the shared ``ALPHA_GRID``, with the alpha-free residual bootstrap
of ``fit_and_bootstrap_370m.py``. The control is the average of its two data
seeds (42 and 69); its envelope is shaded because the control is the only arm
run twice, so no band is drawn for the single-seed MixLaw arm.

Crossings are bootstrapped with ``bootstrap_crossing``, pairing the draws of the
two arms, and are reported with the fraction of draws that cross at all (a
fitted asymptote at or above the target never reaches it).

Whether a crossing exists is decided from the fits, not assumed: if MixLaw's
fitted curve does not reach the control's final fitted loss within the run, the
script says so in its printed output, in the figure (no savings arrow) and in
``compute_savings_results.json`` (``crossing_exists: false``). With the current
curves MixLaw finishes above the control, so the figure instead marks the step
at which the control's fitted curve reaches MixLaw's final loss.

Reads only ``skill_dag_370m_wandb_curves.json``; runs offline.

Usage: python plot_figure_iii_compute_savings.py [--n-boot 200000] [--seed 0]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fit_and_bootstrap_370m import (
    ALPHA_GRID, CURVES_PATH, MIN_STEP, _ols_over_alpha, bootstrap_crossing, ci,
    diff_p, fit_and_bootstrap, fmt_p,
)
from flops import report as flops_report

MIXLAW = Path(__file__).resolve().parent
OUT_DIRS = [MIXLAW / "figures"]
RESULTS_PATH = MIXLAW / "compute_savings_results.json"

ap = argparse.ArgumentParser()
ap.add_argument("--n-boot", type=int, default=200_000)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

data = json.loads(CURVES_PATH.read_text(encoding="utf-8"))
RUNS = data["runs"]
FINAL_STEP = int(data["final_step"])
SEED_KEYS = ("olmo-mix-1124-s42", "olmo-mix-1124-s69")
ML_KEY = "mixlaw-fit"

COL_CONTROL = "#6b7280"      # gray
COL_MIXLAW = "#2563eb"       # blue
COL_CONTROL_BAND = "#9ca3af"  # lighter gray, control seed envelope
WINDOW_MIN = 700
X_LEFT = 650
X_RIGHT = 3000

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 12,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.9,
})


def lead_in(all_steps, all_y, window_min, x_left):
    """Segment running left from the first in-window point.

    Uses the slope to the last point *before* the window, i.e. the slope the
    curve would have had if that checkpoint were still plotted, and returns
    the two endpoints. The far endpoint is placed at x_left; when the implied
    value there is off the top of the panel, matplotlib's clipping is what
    makes the line leave through the top edge rather than the side.
    """
    before = all_steps < window_min
    after = all_steps >= window_min
    if not before.any() or not after.any():
        return None
    s0, y0 = all_steps[before][-1], all_y[before][-1]
    s1, y1 = all_steps[after][0], all_y[after][0]
    slope = (y1 - y0) / (s1 - s0)
    return (np.array([x_left, s1]), np.array([y1 + slope * (x_left - s1), y1]))


def powerlaw(x, a, b, alpha):
    return a + b / np.power(x, alpha)


def point_fit(steps, losses):
    """Point estimate (a, b, alpha) of the estimator inside fit_and_bootstrap."""
    s = np.asarray(steps, dtype=float)
    y = np.asarray(losses, dtype=float)
    keep = s >= MIN_STEP
    s, y = s[keep], y[keep]
    X = s[None, :] ** (-ALPHA_GRID[:, None])
    sse, slope, icept = _ols_over_alpha(X, y[None, :])
    best = int(np.argmin(sse[:, 0]))
    return float(icept[best, 0]), float(slope[best, 0]), float(ALPHA_GRID[best])


def num(x, nd=6):
    """JSON-safe rounded float (non-finite -> None)."""
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def finite_summary(x):
    """Median and 95% interval of the finite entries, plus the finite share."""
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    out = {"fraction_finite": num(finite.mean())}
    if finite.any():
        lo, hi = ci(x[finite])
        out.update(median=num(np.median(x[finite]), 1), ci95=[num(lo, 1), num(hi, 1)])
    else:
        out.update(median=None, ci95=None)
    return out


# ---------------------------------------------------------------- data
steps = np.array(RUNS[SEED_KEYS[0]]["steps"], dtype=float)
if RUNS[SEED_KEYS[1]]["steps"] != RUNS[SEED_KEYS[0]]["steps"] or \
        RUNS[ML_KEY]["steps"] != RUNS[SEED_KEYS[0]]["steps"]:
    raise SystemExit("control seeds and MixLaw arm are not evaluated at the same steps")
seed_y = [np.array(RUNS[k]["macro_bpb"], dtype=float) for k in SEED_KEYS]
ctrl = 0.5 * (seed_y[0] + seed_y[1])
band_lo = np.minimum(seed_y[0], seed_y[1])
band_hi = np.maximum(seed_y[0], seed_y[1])
ml = np.array(RUNS[ML_KEY]["macro_bpb"], dtype=float)

# ----------------------------------------------------------- estimator
# MixLaw keeps its own bootstrap stream from the curve file. The averaged
# control curve is a different series from either seed, so it gets its own
# independent stream (the next index after the file's streams).
n_streams = int(data["bootstrap_stream_count"])
streams = np.random.SeedSequence(args.seed).spawn(n_streams + 1)

fit_ml, finals_ml, a_ml, b_ml, al_ml = fit_and_bootstrap(
    steps, ml, final_step=FINAL_STEP, n_boot=args.n_boot,
    seed=streams[RUNS[ML_KEY]["bootstrap_stream"]], return_params=True)
fit_ct, finals_ct, a_ct, b_ct, al_ct = fit_and_bootstrap(
    steps, ctrl, final_step=FINAL_STEP, n_boot=args.n_boot,
    seed=streams[n_streams], return_params=True)

pt_ml = point_fit(steps, ml)
pt_ct = point_fit(steps, ctrl)
assert abs(powerlaw(FINAL_STEP, *pt_ml) - fit_ml) < 1e-9
assert abs(powerlaw(FINAL_STEP, *pt_ct) - fit_ct) < 1e-9

# Table II's control average is the mean of the two seeds' own fitted finals;
# the single power law fitted to the averaged curve (used here, so that the
# control is one curve that can be extrapolated and crossed) matches it closely.
table2_ctrl = float(np.mean([powerlaw(FINAL_STEP, *point_fit(steps, y)) for y in seed_y]))

# ------------------------------------------------- crossings, both directions
# (1) MixLaw reaching the control's final fitted loss. Draws are paired: each
#     MixLaw draw is crossed with the same-index control final.
cross_ml_pt = float(bootstrap_crossing(*pt_ml, fit_ct))
cross_ml_draws = bootstrap_crossing(a_ml, b_ml, al_ml, finals_ct)
within = cross_ml_draws <= FINAL_STEP
crossing_exists = bool(np.isfinite(cross_ml_pt) and cross_ml_pt <= FINAL_STEP)

# (2) Converse: the step at which the control's fitted power law reaches
#     MixLaw's final fitted loss.
cross_ct_pt = float(bootstrap_crossing(*pt_ct, fit_ml, at_or_above=True))
cross_ct_draws = bootstrap_crossing(a_ct, b_ct, al_ct, finals_ml, at_or_above=True)
cross_ct_summary = finite_summary(cross_ct_draws)

final_diff = finals_ml - finals_ct
fd_lo, fd_hi = ci(final_diff)
fd_p = diff_p(finals_ml, finals_ct)
fl = flops_report()

# ------------------------------------------------------------- printout
print("Figure III: does the MixLaw 1%-floor mixture reach the control's loss?")
print(f"  estimator: y = a + b/step^alpha on steps >= {MIN_STEP}, alpha grid "
      f"[{ALPHA_GRID[0]:.2f}, {ALPHA_GRID[-1]:.2f}] ({ALPHA_GRID.size} pts), "
      f"alpha-free residual bootstrap, n_boot={args.n_boot}, seed={args.seed}")
print(f"  control (2-seed average) fitted final: {fit_ct:.4f}  "
      f"95% CI [{ci(finals_ct)[0]:.4f}, {ci(finals_ct)[1]:.4f}]  "
      f"(mean of per-seed fits, Table II: {table2_ctrl:.4f})")
print(f"  MixLaw fitted final:                   {fit_ml:.4f}  "
      f"95% CI [{ci(finals_ml)[0]:.4f}, {ci(finals_ml)[1]:.4f}]")
print(f"  MixLaw - control at step {FINAL_STEP}: {fit_ml - fit_ct:+.4f}  "
      f"95% CI [{fd_lo:+.4f}, {fd_hi:+.4f}]  two-sided {fmt_p(fd_p, args.n_boot)}")
print(f"  fitted asymptotes: MixLaw {pt_ml[0]:.4f}, control {pt_ct[0]:.4f}")
if crossing_exists:
    print(f"  CROSSING EXISTS: MixLaw reaches the control's final fitted loss "
          f"at step {cross_ml_pt:.0f} of {FINAL_STEP} "
          f"({100 * (1 - cross_ml_pt / FINAL_STEP):.1f}% fewer steps); "
          f"{100 * within.mean():.1f}% of bootstrap draws cross within the run")
else:
    shown = ("never reaches it (its fitted asymptote is above the control's final loss)"
             if not np.isfinite(cross_ml_pt)
             else f"reaches it only at step {cross_ml_pt:.0f}, beyond the {FINAL_STEP}-step run")
    print(f"  NO CROSSING: MixLaw does not reach the control's final fitted loss "
          f"{fit_ct:.4f} within the run; its fitted curve {shown}.")
    print(f"    {100 * within.mean():.1f}% of bootstrap draws cross within the run; "
          f"{100 * np.isfinite(cross_ml_draws).mean():.1f}% cross at any step.")
print(f"  Converse: the control's fitted curve reaches MixLaw's final fitted loss "
      f"{fit_ml:.4f} at step {cross_ct_pt:.0f} "
      f"({100 * cross_ct_pt / FINAL_STEP:.1f}% of the run, "
      f"{FINAL_STEP - cross_ct_pt:.0f} steps before the end)")
if cross_ct_summary["median"] is not None:
    print(f"    bootstrap median step {cross_ct_summary['median']:.0f}, 95% CI "
          f"[{cross_ct_summary['ci95'][0]:.0f}, {cross_ct_summary['ci95'][1]:.0f}]; "
          f"{100 * (cross_ct_draws < FINAL_STEP).mean():.1f}% of draws reach it before "
          f"step {FINAL_STEP}")
print(f"  Pilot cost (arm-independent): {fl['pilot_grid_24']:.3e} FLOPs = "
      f"{100 * fl['pilot_grid_over_one_370m_arm']:.1f}% of one {fl['per_370m_arm']:.3e}-FLOP arm")

results = {
    "description": "Figure III: whether, and where, the MixLaw 1%-floor mixture reaches the "
                   "Olmo-mix-1124 control's loss, and the converse crossing for the control.",
    "estimator": f"y = a + b/step**alpha on steps >= {MIN_STEP}; alpha-free residual bootstrap; "
                 "paired draws; same estimator as Table II",
    "n_boot": args.n_boot,
    "seed": args.seed,
    "final_step": FINAL_STEP,
    "alpha_grid": {"lo": float(ALPHA_GRID[0]), "hi": float(ALPHA_GRID[-1]), "n": int(ALPHA_GRID.size)},
    "control": "Olmo-mix-1124 control, average of data seeds 42 and 69",
    "crossing_exists": crossing_exists,
    "crossing_definition": "the MixLaw fitted curve reaches the control's final fitted loss at or "
                           "before the final step",
    "fitted_final": {
        "control_average": num(fit_ct),
        "control_average_table_ii_mean_of_seed_fits": num(table2_ctrl),
        "mixlaw": num(fit_ml),
        "control_s42": num(powerlaw(FINAL_STEP, *point_fit(steps, seed_y[0]))),
        "control_s69": num(powerlaw(FINAL_STEP, *point_fit(steps, seed_y[1]))),
    },
    "fitted_final_ci95": {
        "control_average": [num(v) for v in ci(finals_ct)],
        "mixlaw": [num(v) for v in ci(finals_ml)],
    },
    "fitted_asymptote": {"control_average": num(pt_ct[0]), "mixlaw": num(pt_ml[0])},
    "fitted_alpha": {"control_average": num(pt_ct[2]), "mixlaw": num(pt_ml[2])},
    "mixlaw_minus_control_final": {
        "mean_diff_bpb": num(final_diff.mean()),
        "ci95": [num(fd_lo), num(fd_hi)],
        "p_value": num(fd_p),
        "p_display": fmt_p(fd_p, args.n_boot),
    },
    "mixlaw_reaches_control_final": {
        "target": num(fit_ct),
        "point_estimate_step": num(cross_ml_pt, 1),
        "reaches_within_run_point_estimate": crossing_exists,
        "probability_within_run": num(within.mean()),
        "fraction_of_draws_that_cross_at_any_step": num(np.isfinite(cross_ml_draws).mean()),
        "extrapolated_crossing_step_over_finite_draws": finite_summary(cross_ml_draws),
    },
    "control_reaches_mixlaw_final": {
        "target": num(fit_ml),
        "point_estimate_step": num(cross_ct_pt, 1),
        "steps_before_end_of_run": num(FINAL_STEP - cross_ct_pt, 1),
        "fraction_of_run": num(cross_ct_pt / FINAL_STEP),
        "bootstrap_step": cross_ct_summary,
        "probability_before_end_of_run": num((cross_ct_draws < FINAL_STEP).mean()),
    },
    "pilot_flops": {
        "pilot_grid_24": num(fl["pilot_grid_24"], 6),
        "one_370m_arm": num(fl["per_370m_arm"], 6),
        "pilot_grid_over_one_370m_arm": num(fl["pilot_grid_over_one_370m_arm"]),
    },
}
RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {RESULTS_PATH}")

# --------------------------------------------------------------- figure
fig, ax = plt.subplots(figsize=(9.2, 5.8), dpi=200)

keep = steps >= 700
ax.fill_between(steps[keep], band_lo[keep], band_hi[keep], color=COL_CONTROL_BAND,
                alpha=0.40, linewidth=0, zorder=1, label="Control seed range (n=2)")
ax.plot(steps[keep], ctrl[keep], color=COL_CONTROL, linestyle="-", marker="s", markersize=4,
        linewidth=2.0, zorder=3, label="Olmo-mix-1124 control (2-seed avg)")
ax.plot(steps[keep], ml[keep], color=COL_MIXLAW, linestyle="-", marker="o", markersize=4,
        linewidth=2.4, zorder=4, label="MixLaw fit (ours, observed)")

# Lead-in: extend both curves off the left edge at the slope implied by the
# checkpoint before the window, so the panel reads as a run already under way.
for _y, _color, _lw, _z in ((ctrl, COL_CONTROL, 2.0, 3), (ml, COL_MIXLAW, 2.4, 4)):
    _seg = lead_in(steps, _y, WINDOW_MIN, X_LEFT)
    if _seg is not None:
        ax.plot(_seg[0], _seg[1], color=_color, linestyle="-", linewidth=_lw, zorder=_z)
_lo_seg = lead_in(steps, band_lo, WINDOW_MIN, X_LEFT)
_hi_seg = lead_in(steps, band_hi, WINDOW_MIN, X_LEFT)
if _lo_seg is not None and _hi_seg is not None:
    ax.fill_between(_lo_seg[0], _lo_seg[1], _hi_seg[1], color=COL_CONTROL_BAND,
                    alpha=0.40, linewidth=0, zorder=1)

# Power-law extrapolation past the end of the run: dashed gray for the control,
# dotted blue for the MixLaw arm's own fit, so the gap between the two fitted
# asymptotes is visible.
ext = np.linspace(FINAL_STEP, X_RIGHT, 300)
ax.plot(ext, powerlaw(ext, *pt_ct), color=COL_CONTROL, linestyle=(0, (2, 2)), linewidth=1.8,
        zorder=2, label="Control, power-law extrapolation")
ax.plot(ext, powerlaw(ext, *pt_ml), color=COL_MIXLAW, linestyle=(0, (1, 2)), linewidth=1.8,
        zorder=2, label="MixLaw, power-law extrapolation")

if crossing_exists:
    # MixLaw's fitted curve reaches the control's final fitted loss inside the
    # run: mark the step and the shortfall, above both curves.
    window = (steps >= cross_ml_pt) & (steps <= FINAL_STEP)
    local_max = max(band_hi[window].max(), ml[window].max()) if window.any() else fit_ct
    faster_y = local_max + 0.018
    ax.annotate("", xy=(cross_ml_pt, faster_y), xytext=(FINAL_STEP, faster_y),
                arrowprops=dict(arrowstyle="<->", color="#111", lw=1.6))
    ax.text((cross_ml_pt + FINAL_STEP) / 2, faster_y + 0.010,
            f"{100 * (1 - cross_ml_pt / FINAL_STEP):.1f}% fewer steps",
            ha="center", fontsize=11, fontweight="bold")
    ax.axhline(fit_ct, color="#999999", linestyle=":", linewidth=1.0, zorder=1)
    title = "MixLaw reaches the control's final loss before the end of the run"
else:
    # No crossing, so no savings arrow. Mark the control's fitted final (the
    # level MixLaw does not reach) and where the control reaches MixLaw's final.
    ax.axhline(fit_ct, color=COL_CONTROL, linestyle=":", linewidth=1.0, zorder=1)
    ax.axhline(fit_ml, color=COL_MIXLAW, linestyle=":", linewidth=1.0, zorder=1)
    ax.text(X_RIGHT - 25, fit_ct + 0.0045, f"control fitted final {fit_ct:.4f}",
            ha="right", va="bottom", fontsize=9, color=COL_CONTROL)
    ax.text(X_RIGHT - 25, fit_ml + 0.0045, f"MixLaw fitted final {fit_ml:.4f}",
            ha="right", va="bottom", fontsize=9, color=COL_MIXLAW)
    ax.plot([cross_ct_pt], [fit_ml], marker="s", markersize=7, markerfacecolor="white",
            markeredgecolor=COL_CONTROL, markeredgewidth=1.6, zorder=6, linestyle="none")
    ax.annotate(f"control's fit reaches\nMixLaw's final loss\nat step {cross_ct_pt:.0f}",
                xy=(cross_ct_pt, fit_ml), xytext=(cross_ct_pt - 120, fit_ml + 0.075),
                ha="center", fontsize=9.5, color="#111",
                arrowprops=dict(arrowstyle="->", color="#111", lw=1.0))
    title = "MixLaw does not reach the control's final loss"

ax.set_xlabel("Training step", labelpad=8)
ax.set_ylabel("Validation macro bits-per-byte\n(20-task OLMES avg, $\\downarrow$ lower is better)")
ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
ax.grid(True, linestyle=":", linewidth=0.7, color="#c9c9c9", alpha=0.9)
ax.set_axisbelow(True)
ax.set_xlim(650, X_RIGHT)
ax.set_ylim(1.565, 1.90)
ax.legend(loc="upper right", frameon=False, fontsize=10)

fig.subplots_adjust(left=0.115, right=0.97, top=0.90, bottom=0.13)
for out_dir in OUT_DIRS:
    if not out_dir.parent.exists():
        continue
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "figure_iii_compute_savings.png"
    fig.savefig(path)
    print(f"Wrote {path}")
