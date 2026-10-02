"""Figure III: do the dynamic-reweighting arms reach the LightGBM static loss sooner?

The figure asks how many training steps each Skill-It arm (offline probe,
online derivative) needs to reach the final loss of the static 1%-floor
LightGBM mixture it starts from, and what that saving is worth once the 60M
runs each arm depends on are charged against it. Every number comes from one
estimator, the same one behind Tables II and III: the power law
``y = a + b / step**alpha`` fitted on steps >= 1000 over the shared
``ALPHA_GRID``, with the alpha-free residual bootstrap of
``fit_and_bootstrap_370m.py``. Each arm keeps its own bootstrap stream from the
curve file, so the fitted finals match the committed results JSON.

Crossings are bootstrapped with ``bootstrap_crossing``, pairing the draws of the
two arms, and are reported with the fraction of draws that cross within the run.
The step saving is ``1 - crossing / final_step``; training FLOPs are linear in
steps, so it is also the FLOP saving as a fraction of one 370M arm. The net
saving subtracts each arm's overhead, counted in 60M runs (``flops.py``):

- offline probe: 8 runs (the seven one-hot probes and the mix01 probe);
- online derivative: 24 runs (the MixLaw pilot grid its derivatives come from).

The figure shows the gross step savings and the converse for the probe arm: the
LightGBM fit extrapolated until it reaches the probe arm's final loss. The net
savings and the derivative arm's converse are in the JSON.

The script also keeps the earlier MixLaw-vs-control comparison in
``compute_savings_results.json`` (``mixlaw_vs_control``): the MixLaw mixture
never reaches the control's final loss, and the control's fitted curve reaches
MixLaw's final loss at step ~1755.

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
from flops import N_PILOTS, report as flops_report

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
REF_KEY = "lightgbm-l40s"

# Overhead per dynamic arm, in 60M runs (see the module docstring).
DYNAMIC = {
    "skillit-probe": {"name": "Skill-It probe", "overhead_runs": 8,
                      "overhead_desc": "7 one-hot probes + the mix01 probe"},
    "skillit-derivative": {"name": "Skill-It derivative", "overhead_runs": N_PILOTS,
                           "overhead_desc": f"the {N_PILOTS}-run MixLaw pilot grid"},
}

# Colors of the original Figure III: gray reference, blue comparison arm; the
# second Skill-It arm in orange (colorblind-safe against blue).
COL_REF = "#6b7280"
COL_PROBE = "#2563eb"
COL_DERIV = "#d97706"
WINDOW_MIN = 700
X_LEFT = 650
X_RIGHT = 3600

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


def finite_summary(x, nd=1):
    """Median and 95% interval of the finite entries, plus the finite share."""
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    out = {"fraction_finite": num(finite.mean())}
    if finite.any():
        lo, hi = ci(x[finite])
        out.update(median=num(np.median(x[finite]), nd), ci95=[num(lo, nd), num(hi, nd)])
    else:
        out.update(median=None, ci95=None)
    return out


# ---------------------------------------------------------------- data
steps = np.array(RUNS[SEED_KEYS[0]]["steps"], dtype=float)
for key in (SEED_KEYS[1], ML_KEY, REF_KEY, *DYNAMIC):
    if RUNS[key]["steps"] != RUNS[SEED_KEYS[0]]["steps"]:
        raise SystemExit(f"{key} is not evaluated at the same steps as the control")
curve = {k: np.array(RUNS[k]["macro_bpb"], dtype=float) for k in RUNS}
n_streams = int(data["bootstrap_stream_count"])
streams = np.random.SeedSequence(args.seed).spawn(n_streams + 1)
fl = flops_report()
PER_ARM, PER_RUN = fl["per_370m_arm"], fl["per_60m_run"]


def fit(key, y=None, stream=None):
    """Fit + bootstrap with the arm's own committed stream (or an explicit one)."""
    y = curve[key] if y is None else y
    s = streams[RUNS[key]["bootstrap_stream"]] if stream is None else stream
    fitted, finals, a, b, al = fit_and_bootstrap(
        steps, y, final_step=FINAL_STEP, n_boot=args.n_boot, seed=s, return_params=True)
    pt = point_fit(steps, y)
    assert abs(powerlaw(FINAL_STEP, *pt) - fitted) < 1e-9
    return {"fitted": fitted, "finals": finals, "a": a, "b": b, "alpha": al, "pt": pt}


# ------------------------------------- dynamic arms vs the LightGBM static run
ref = fit(REF_KEY)
print("Figure III: do the dynamic-reweighting arms reach the LightGBM static loss sooner?")
print(f"  estimator: y = a + b/step^alpha on steps >= {MIN_STEP}, alpha grid "
      f"[{ALPHA_GRID[0]:.2f}, {ALPHA_GRID[-1]:.2f}] ({ALPHA_GRID.size} pts), "
      f"alpha-free residual bootstrap, n_boot={args.n_boot}, seed={args.seed}")
print(f"  LightGBM static fitted final {ref['fitted']:.4f}  95% CI "
      f"[{ci(ref['finals'])[0]:.4f}, {ci(ref['finals'])[1]:.4f}]")
print(f"  FLOPs: {PER_RUN:.4e} per 60M run, {PER_ARM:.4e} per 370M arm")

dyn = {}
dyn_results = {}
for key, meta in DYNAMIC.items():
    arm = fit(key)
    dyn[key] = arm
    overhead = meta["overhead_runs"] * PER_RUN / PER_ARM

    # (1) The dynamic arm reaching LightGBM's final fitted loss.
    cross_pt = float(bootstrap_crossing(*arm["pt"], ref["fitted"]))
    cross_draws = bootstrap_crossing(arm["a"], arm["b"], arm["alpha"], ref["finals"])
    finite = np.isfinite(cross_draws)
    saved_draws = 1.0 - cross_draws[finite] / FINAL_STEP
    net_draws = saved_draws - overhead
    saved_pt = 1.0 - cross_pt / FINAL_STEP
    s_lo, s_hi = ci(saved_draws)
    n_lo, n_hi = ci(net_draws)
    within = float((cross_draws <= FINAL_STEP).mean())

    # (2) Converse: LightGBM's fit extrapolated to the dynamic arm's final loss.
    conv_pt = float(bootstrap_crossing(*ref["pt"], arm["fitted"], at_or_above=True))
    conv_draws = bootstrap_crossing(ref["a"], ref["b"], ref["alpha"], arm["finals"],
                                    at_or_above=True)
    conv_mult = finite_summary(conv_draws / FINAL_STEP, nd=3)

    d = arm["finals"] - ref["finals"]
    d_lo, d_hi = ci(d)
    print(f"\n  {meta['name']}: fitted final {arm['fitted']:.4f} "
          f"({arm['fitted'] - ref['fitted']:+.4f} vs LightGBM, 95% CI [{d_lo:+.4f}, {d_hi:+.4f}])")
    print(f"    reaches LightGBM's final loss at step {cross_pt:.0f} of {FINAL_STEP} "
          f"(bootstrap median {np.median(cross_draws[finite]):.0f}, 95% CI "
          f"[{ci(cross_draws[finite])[0]:.0f}, {ci(cross_draws[finite])[1]:.0f}]); "
          f"{100 * within:.1f}% of draws cross within the run")
    print(f"    gross step saving {100 * saved_pt:.1f}% [{100 * s_lo:.1f}, {100 * s_hi:.1f}] "
          f"= {saved_pt * PER_ARM:.3e} FLOPs")
    print(f"    overhead {meta['overhead_runs']} x 60M ({meta['overhead_desc']}) = "
          f"{meta['overhead_runs'] * PER_RUN:.3e} FLOPs = {100 * overhead:.2f}% of one arm; "
          f"break-even crossing step {FINAL_STEP * (1 - overhead):.0f}")
    print(f"    net saving {100 * (saved_pt - overhead):.1f}% [{100 * n_lo:.1f}, {100 * n_hi:.1f}] "
          f"= {(saved_pt - overhead) * PER_ARM:.3e} FLOPs; P(net > 0) = {(net_draws > 0).mean():.4f}")
    print(f"    converse: LightGBM's fit reaches {arm['fitted']:.4f} at "
          f"{conv_pt / FINAL_STEP:.2f}x the run ({conv_pt:.0f} steps); bootstrap "
          f"{conv_mult['median']}x [{conv_mult['ci95'][0]}, {conv_mult['ci95'][1]}], "
          f"{100 * (1 - conv_mult['fraction_finite']):.1f}% of draws never reach it")

    dyn_results[key] = {
        "label": meta["name"],
        "fitted_final": num(arm["fitted"]),
        "fitted_final_ci95": [num(v) for v in ci(arm["finals"])],
        "fitted_asymptote": num(arm["pt"][0]),
        "fitted_alpha": num(arm["pt"][2]),
        "minus_lightgbm_final": {"mean_diff_bpb": num(d.mean()), "ci95": [num(d_lo), num(d_hi)]},
        "reaches_lightgbm_final": {
            "target": num(ref["fitted"]),
            "point_estimate_step": num(cross_pt, 1),
            "bootstrap_step": finite_summary(cross_draws),
            "probability_within_run": num(within),
        },
        "gross_step_saving": {"point": num(saved_pt), "ci95": [num(s_lo), num(s_hi)],
                              "flops": num(saved_pt * PER_ARM, 6)},
        "overhead": {"runs_60m": meta["overhead_runs"], "description": meta["overhead_desc"],
                     "flops": num(meta["overhead_runs"] * PER_RUN, 6),
                     "fraction_of_one_370m_arm": num(overhead),
                     "break_even_crossing_step": num(FINAL_STEP * (1 - overhead), 1)},
        "net_saving": {"point": num(saved_pt - overhead), "ci95": [num(n_lo), num(n_hi)],
                       "flops": num((saved_pt - overhead) * PER_ARM, 6),
                       "probability_positive": num((net_draws > 0).mean())},
        "converse_lightgbm_reaches_arm_final": {
            "target": num(arm["fitted"]),
            "point_estimate_step": num(conv_pt, 1),
            "point_estimate_multiple_of_run": num(conv_pt / FINAL_STEP, 3),
            "bootstrap_multiple_of_run": conv_mult,
        },
    }

# --------------------------------------- MixLaw vs control (no crossing)
# MixLaw keeps its own bootstrap stream from the curve file. The averaged
# control curve is a different series from either seed, so it gets its own
# independent stream (the next index after the file's streams).
ctrl = 0.5 * (curve[SEED_KEYS[0]] + curve[SEED_KEYS[1]])
ml = fit(ML_KEY)
ct = fit(SEED_KEYS[0], y=ctrl, stream=streams[n_streams])
table2_ctrl = float(np.mean([powerlaw(FINAL_STEP, *point_fit(steps, curve[k])) for k in SEED_KEYS]))
cross_ml_pt = float(bootstrap_crossing(*ml["pt"], ct["fitted"]))
cross_ml_draws = bootstrap_crossing(ml["a"], ml["b"], ml["alpha"], ct["finals"])
ml_within = cross_ml_draws <= FINAL_STEP
cross_ct_pt = float(bootstrap_crossing(*ct["pt"], ml["fitted"], at_or_above=True))
cross_ct_draws = bootstrap_crossing(ct["a"], ct["b"], ct["alpha"], ml["finals"], at_or_above=True)
fd = ml["finals"] - ct["finals"]
fd_lo, fd_hi = ci(fd)
fd_p = diff_p(ml["finals"], ct["finals"])
print(f"\n  MixLaw vs control: MixLaw {ml['fitted']:.4f} vs control {ct['fitted']:.4f} "
      f"({ml['fitted'] - ct['fitted']:+.4f} [{fd_lo:+.4f}, {fd_hi:+.4f}], {fmt_p(fd_p, args.n_boot)}); "
      f"MixLaw asymptote {ml['pt'][0]:.4f}, {100 * ml_within.mean():.1f}% of draws reach the "
      f"control's final within the run; the control reaches MixLaw's final at step {cross_ct_pt:.0f}")

results = {
    "description": "Figure III: whether, and where, the two Skill-It dynamic-reweighting arms reach "
                   "the static LightGBM 1%-floor mixture's final loss, the step and FLOP savings "
                   "net of each arm's 60M-run overhead, and (mixlaw_vs_control) the MixLaw "
                   "1%-floor mixture against the Olmo-mix-1124 control.",
    "estimator": f"y = a + b/step**alpha on steps >= {MIN_STEP}; alpha-free residual bootstrap; "
                 "paired draws; same estimator as Tables II and III",
    "n_boot": args.n_boot,
    "seed": args.seed,
    "final_step": FINAL_STEP,
    "alpha_grid": {"lo": float(ALPHA_GRID[0]), "hi": float(ALPHA_GRID[-1]), "n": int(ALPHA_GRID.size)},
    "flops": {"per_60m_run": num(PER_RUN, 6), "per_370m_arm": num(PER_ARM, 6),
              "pilot_grid_24": num(fl["pilot_grid_24"], 6),
              "pilot_grid_over_one_370m_arm": num(fl["pilot_grid_over_one_370m_arm"])},
    "dynamic_vs_lightgbm": {
        "reference": REF_KEY,
        "reference_fitted_final": num(ref["fitted"]),
        "reference_fitted_final_ci95": [num(v) for v in ci(ref["finals"])],
        "reference_fitted_asymptote": num(ref["pt"][0]),
        "reference_fitted_alpha": num(ref["pt"][2]),
        "step_saving_definition": "1 - (step at which the arm's fitted curve reaches the "
                                  "reference's final fitted loss) / final_step; equal to the FLOP "
                                  "saving as a fraction of one 370M arm",
        "arms": dyn_results,
    },
    "mixlaw_vs_control": {
        "control": "Olmo-mix-1124 control, average of data seeds 42 and 69",
        "crossing_exists": bool(np.isfinite(cross_ml_pt) and cross_ml_pt <= FINAL_STEP),
        "fitted_final": {
            "control_average": num(ct["fitted"]),
            "control_average_table_ii_mean_of_seed_fits": num(table2_ctrl),
            "mixlaw": num(ml["fitted"]),
        },
        "fitted_final_ci95": {"control_average": [num(v) for v in ci(ct["finals"])],
                              "mixlaw": [num(v) for v in ci(ml["finals"])]},
        "fitted_asymptote": {"control_average": num(ct["pt"][0]), "mixlaw": num(ml["pt"][0])},
        "mixlaw_minus_control_final": {"mean_diff_bpb": num(fd.mean()), "ci95": [num(fd_lo), num(fd_hi)],
                                       "p_value": num(fd_p), "p_display": fmt_p(fd_p, args.n_boot)},
        "mixlaw_reaches_control_final": {
            "target": num(ct["fitted"]),
            "point_estimate_step": num(cross_ml_pt, 1),
            "probability_within_run": num(ml_within.mean()),
            "fraction_of_draws_that_cross_at_any_step": num(np.isfinite(cross_ml_draws).mean()),
        },
        "control_reaches_mixlaw_final": {
            "target": num(ml["fitted"]),
            "point_estimate_step": num(cross_ct_pt, 1),
            "steps_before_end_of_run": num(FINAL_STEP - cross_ct_pt, 1),
            "bootstrap_step": finite_summary(cross_ct_draws),
        },
    },
}
RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
print(f"\nWrote {RESULTS_PATH}")

# --------------------------------------------------------------- figure
# Same layout as the original Figure III (control vs MixLaw), with the LightGBM
# static run as the reference and the two Skill-It arms as the comparisons.
fig, ax = plt.subplots(figsize=(9.2, 5.8), dpi=200)

lgb, prb, drv = curve[REF_KEY], curve["skillit-probe"], curve["skillit-derivative"]
keep = steps >= 700
ax.plot(steps[keep], lgb[keep], color=COL_REF, linestyle="-", marker="s", markersize=4,
        linewidth=2.0, zorder=3, label="LightGBM static (1% floor, observed)")
ax.plot(steps[keep], prb[keep], color=COL_PROBE, linestyle="-", marker="o", markersize=4,
        linewidth=2.4, zorder=4, label="Skill-It probe (observed)")
ax.plot(steps[keep], drv[keep], color=COL_DERIV, linestyle="-", marker="^", markersize=4,
        linewidth=2.4, zorder=4, label="Skill-It derivative (observed)")

# Lead-in: extend the curves off the left edge at the slope implied by the
# checkpoint before the window, so the panel reads as a run already under way.
for _y, _color, _lw, _z in ((lgb, COL_REF, 2.0, 3), (prb, COL_PROBE, 2.4, 4), (drv, COL_DERIV, 2.4, 4)):
    _seg = lead_in(steps, _y, WINDOW_MIN, X_LEFT)
    if _seg is not None:
        ax.plot(_seg[0], _seg[1], color=_color, linestyle="-", linewidth=_lw, zorder=_z)

# Power-law extrapolation past the end of the run: dashed gray for LightGBM,
# dotted for each Skill-It arm's own fit.
ext = np.linspace(FINAL_STEP, X_RIGHT, 300)
ax.plot(ext, powerlaw(ext, *ref["pt"]), color=COL_REF, linestyle=(0, (2, 2)), linewidth=1.8,
        zorder=2, label="LightGBM, power-law extrapolation")
ax.plot(ext, powerlaw(ext, *dyn["skillit-probe"]["pt"]), color=COL_PROBE, linestyle=(0, (1, 2)),
        linewidth=1.8, zorder=2, label="Probe, power-law extrapolation")
ax.plot(ext, powerlaw(ext, *dyn["skillit-derivative"]["pt"]), color=COL_DERIV, linestyle=(0, (1, 2)),
        linewidth=1.8, zorder=2, label="Derivative, power-law extrapolation")

fit_lgb = ref["fitted"]
fit_prb = dyn_results["skillit-probe"]["fitted_final"]
ax.axhline(fit_lgb, color=COL_REF, linestyle=":", linewidth=1.0, zorder=1)
ax.axhline(fit_prb, color=COL_PROBE, linestyle=":", linewidth=1.0, zorder=1)
ax.text(X_RIGHT - 300, fit_lgb + 0.0045, f"LightGBM fitted final {fit_lgb:.4f}",
        ha="right", va="bottom", fontsize=9, color=COL_REF)
# LightGBM's label ends short of the right edge, clear of the arrow to its crossing;
# the probe's sits below its line, above the Skill-It extrapolations.
ax.text(X_RIGHT - 25, fit_prb - 0.0045, f"probe fitted final {fit_prb:.4f}",
        ha="right", va="top", fontsize=9, color=COL_PROBE)

annotations = []
for key, short, color, dx in (("skillit-probe", "probe", COL_PROBE, -330),
                              ("skillit-derivative", "derivative", COL_DERIV, 230)):
    r = dyn_results[key]
    xc = r["reaches_lightgbm_final"]["point_estimate_step"]
    annotations.append((xc, fit_lgb, color, dx,
                        f"{short}'s fit reaches\nLightGBM's final loss\nat step {xc:.0f}\n"
                        f"({100 * r['gross_step_saving']['point']:.1f}% fewer steps)"))
conv = dyn_results["skillit-probe"]["converse_lightgbm_reaches_arm_final"]
annotations.append((conv["point_estimate_step"], fit_prb, COL_REF, -120,
                    f"LightGBM's fit reaches\nthe probe's final loss\nat step {conv['point_estimate_step']:.0f}\n"
                    f"({conv['point_estimate_multiple_of_run']:.2f}\u00d7 the steps)"))
for x, y, color, dx, text in annotations:
    ax.plot([x], [y], marker="s", markersize=7, markerfacecolor="white",
            markeredgecolor=color, markeredgewidth=1.6, zorder=6, linestyle="none")
    ax.annotate(text, xy=(x, y), xytext=(x + dx, y + 0.075),
                ha="center", fontsize=9.5, color="#111",
                arrowprops=dict(arrowstyle="->", color="#111", lw=1.0))
title = "Dynamic reweighting reaches LightGBM's final loss sooner"

ax.set_xlabel("Training step", labelpad=8)
ax.set_ylabel("Validation macro bits-per-byte\n(20-task OLMES avg, $\downarrow$ lower is better)")
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
