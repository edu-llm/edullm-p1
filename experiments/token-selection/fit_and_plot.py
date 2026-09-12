#!/usr/bin/env python3
"""Token-selection 370M: power-law fits, bootstrap CIs, and paper Figures 1 and 2.

This is the single source of truth for the numbers in Table 1 and for
``figures/figure1_training_curves.png`` / ``figures/figure2_final_bar_chart.png``.

Experiment
----------
Seven 370M OLMo2 arms trained on a 10B-token RegMix corpus, differing only in
which tokens contribute to the loss:

  * Full-loss control  -- no masking (baseline)
  * Random control     -- random ~60% token keep rate (baseline; mask is free)
  * RHO-1              -- reference-model excess loss
  * Attention          -- last-block attention top-k
  * BLADE              -- reference-ensemble learnability
  * Perplexity         -- offline middle-perplexity band
  * REL-EMA            -- self-reference EMA excess loss

The metric is held-out macro bits-per-byte (``eval/macro_bpb``), lower better.

Fitting protocol (the "matched alpha-free protocol")
----------------------------------------------------
For each arm we fit the 3-parameter power law

    y = a + b * step ** (-alpha)

by profiling ``alpha`` over a grid: for each candidate ``alpha`` the model is
linear in ``(a, b)`` given the regressor ``X = step ** (-alpha)``, so ``(a, b)``
come from closed-form OLS and we keep the grid point with the lowest SSE.

  * Fit window: ``step >= 1000``. Earlier points are dominated by warmup and
    are not yet in the asymptotic power-law regime.
  * Alpha grid: ``np.linspace(0.05, 6.0, 1192)``.
  * 10,000 i.i.d. residual bootstrap draws. Residuals are resampled with
    replacement from the fit-window residuals and added back to the fitted
    curve.
  * ``alpha`` is RE-ESTIMATED on every bootstrap draw ("alpha-free"), rather
    than being frozen at the point-estimate value.
  * The fitted final value is evaluated at each arm's OWN final logged step
    (2384 for the full-loss control, 2360 for the other six).
  * CI = 2.5 / 97.5 percentile of the bootstrap distribution.
  * ``numpy`` ``default_rng`` seeded with 0, so the numbers are reproducible.

WHY THE ALPHA GRID WAS WIDENED FROM [0.05, 3.0] TO [0.05, 6.0]
--------------------------------------------------------------
The earlier analysis profiled ``alpha`` over ``np.linspace(0.05, 3.0, ...)``.
REL-EMA decays much more sharply than the other arms, and its profiled optimum
landed exactly on the upper edge of that grid -- it was BOUNDARY-PINNED at
alpha = 3.0, i.e. the grid, not the data, was choosing the exponent. A pinned
exponent also truncates the bootstrap: draws that "want" a steeper exponent all
pile up at the same boundary value, which artificially shrinks the interval.

Widening the grid to 6.0 lets REL-EMA settle at its true interior optimum
(alpha = 3.502) and un-truncates its bootstrap. Under the wider grid no arm
sits on a boundary (largest: REL-EMA at 3.502; smallest: the full-loss control
at 0.585), so the exponent is data-determined for every arm.

WHY ALPHA IS RE-ESTIMATED PER DRAW (ALPHA-FREE)
-----------------------------------------------
Freezing ``alpha`` at its point estimate treats a quantity that was estimated
from the same 11-12 points as if it were known exactly, so it understates
uncertainty. Re-estimating it per draw propagates that uncertainty. It is also
the more conservative of the two variants: mean CI width is 0.00989 bpb
alpha-free vs 0.00700 bpb alpha-fixed, so every interval reported here is the
WIDER of the two. That is the basis on which the protocol was chosen.

Data source
-----------
``token_selection_370m_wandb_curves.json`` (committed) is the default and is
identical to the live W&B histories on the fit window, so the script runs
offline with no credentials. Pass ``--source wandb`` to re-pull from
``eduLLM/token-selection`` instead.

Usage
-----
    python fit_and_plot.py                 # table + delta CIs + both figures
    python fit_and_plot.py --source wandb  # re-pull curves from W&B first
    python fit_and_plot.py --no-figures    # numbers only
    python fit_and_plot.py --write-json    # also refresh the bootstrap JSON
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CURVES_PATH = ROOT / "token_selection_370m_wandb_curves.json"
BOOTSTRAP_PATH = ROOT / "token_selection_370m_bootstrap_results.json"
FIG_DIR = ROOT / "figures"

# ---------------------------------------------------------------- protocol ---
MIN_STEP = 1000
N_BOOT = 10_000
SEED = 0
# Widened from [0.05, 3.0]; see module docstring (REL-EMA was pinned at 3.0).
ALPHA_GRID = np.linspace(0.05, 6.0, 1192)
ALPHA_FREE = True

WANDB_PROJECT = "eduLLM/token-selection"
WANDB_RUNS = {
    "control": "hh19uatg",
    "rho_1": "ebf1fa33048b3459f768cd471c2a8917",
    "random_control": "fa841187ff07e9164da282efd353c217",
    "attention": "01e18e7141fdbf9b988f17c32bb0c084",
    "blade": "005xjces",
    "middle_ppl": "2bbd4ec49b531d37115a44f73a0512e2",
    "rel_ema": "cc52d5537a03ad8e57cc87a025668b2e",
}

# Display order == ascending fitted-final bpb. Colors are the matplotlib tab10
# family and are shared between Figure 1 and Figure 2.
ARMS = [
    # key,             bar/table label,     figure-1 legend label,          color,     ls,   marker
    ("control",        "Full-loss control", "Full-loss control (baseline)", "#000000", "--", "s"),
    ("rho_1",          "RHO-1",             "RHO-1",                        "#1f77b4", "-",  "o"),
    ("random_control", "Random control",    "Random control (baseline)",    "#7f7f7f", ":",  "^"),
    ("attention",      "Attention",         "Attention",                    "#d62728", "-",  "D"),
    ("blade",          "BLADE",             "BLADE",                        "#9467bd", "-",  "v"),
    ("middle_ppl",     "Perplexity",        "Perplexity",                   "#2ca02c", "-",  "P"),
    ("rel_ema",        "REL-EMA",           "REL-EMA",                      "#bcbd22", "-",  "*"),
]
ORDER = [a[0] for a in ARMS]
LABEL = {a[0]: a[1] for a in ARMS}
LEGEND = {a[0]: a[2] for a in ARMS}
COLOR = {a[0]: a[3] for a in ARMS}
LINESTYLE = {a[0]: a[4] for a in ARMS}
MARKER = {a[0]: a[5] for a in ARMS}


# -------------------------------------------------------------------- data ---
def load_curves_from_json(path: Path = CURVES_PATH) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Read the committed curve cache -> {key: (steps, macro_bpb)}."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for key, arm in payload["arms"].items():
        steps = np.asarray(arm["steps"], dtype=float)
        loss = np.asarray(arm["curve"], dtype=float)
        idx = np.argsort(steps)
        out[key] = (steps[idx], loss[idx])
    return out


def load_curves_from_wandb() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Pull eval/macro_bpb histories live from W&B."""
    import wandb  # imported lazily so the offline path needs no wandb

    api = wandb.Api()
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for key in ORDER:
        run = api.run(f"{WANDB_PROJECT}/{WANDB_RUNS[key]}")
        rows = [
            (int(r["_step"]), float(r["eval/macro_bpb"]))
            for r in run.scan_history(keys=["_step", "eval/macro_bpb"])
            if r.get("eval/macro_bpb") is not None
        ]
        rows.sort()
        out[key] = (
            np.asarray([r[0] for r in rows], dtype=float),
            np.asarray([r[1] for r in rows], dtype=float),
        )
        print(f"  W&B {key:<15} n={len(rows):3d} final_step={rows[-1][0]} final={rows[-1][1]:.4f}")
    return out


# --------------------------------------------------------------------- fit ---
def _ols_batch(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form OLS of y = a + b*x for a batch of responses.

    ``x`` is the (n,) regressor shared by every row; ``y`` is (B, n).
    Returns ``a`` (B,) and ``b`` (B,).
    """
    n = x.size
    sx = x.sum()
    sxx = (x * x).sum()
    sy = y.sum(axis=1)
    sxy = y @ x
    den = n * sxx - sx * sx
    b = (n * sxy - sx * sy) / den
    a = (sy - b * sx) / n
    return a, b


def profile_fit(steps: np.ndarray, loss: np.ndarray) -> tuple[float, float, float]:
    """Profile alpha over ALPHA_GRID; return (alpha, a, b) minimizing SSE."""
    y = loss[None, :]
    best: tuple[float, float, float, float] | None = None
    for alpha in ALPHA_GRID:
        x = steps ** (-alpha)
        a, b = _ols_batch(x, y)
        resid = y - (a[:, None] + b[:, None] * x[None, :])
        sse = float((resid * resid).sum())
        if best is None or sse < best[0]:
            best = (sse, float(alpha), float(a[0]), float(b[0]))
    assert best is not None
    _, alpha_hat, a_hat, b_hat = best
    return alpha_hat, a_hat, b_hat


def bootstrap_arm(
    steps: np.ndarray,
    loss: np.ndarray,
    final_step: float,
    alpha_free: bool = ALPHA_FREE,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict:
    """Residual-bootstrap the fitted final value for a single arm."""
    alpha, a, b = profile_fit(steps, loss)
    pred = a + b * steps ** (-alpha)
    resid = loss - pred

    rng = np.random.default_rng(seed)
    draws = pred[None, :] + resid[rng.integers(0, resid.size, (n_boot, resid.size))]

    if alpha_free:
        # Re-profile alpha independently for every bootstrap draw.
        best_sse = np.full(n_boot, np.inf)
        finals = np.empty(n_boot)
        for cand in ALPHA_GRID:
            x = steps ** (-cand)
            aa, bb = _ols_batch(x, draws)
            r = draws - (aa[:, None] + bb[:, None] * x[None, :])
            sse = (r * r).sum(axis=1)
            better = sse < best_sse
            if better.any():
                best_sse[better] = sse[better]
                finals[better] = (aa + bb * final_step ** (-cand))[better]
    else:
        x = steps ** (-alpha)
        aa, bb = _ols_batch(x, draws)
        finals = aa + bb * final_step ** (-alpha)

    lo, hi = np.percentile(finals, [2.5, 97.5])
    return {
        "alpha": alpha,
        "a": a,
        "b": b,
        "fitted_final": a + b * final_step ** (-alpha),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "boot": finals,
    }


def fit_all(curves: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for key in ORDER:
        steps, loss = curves[key]
        mask = steps >= MIN_STEP
        res = bootstrap_arm(steps[mask], loss[mask], float(steps[-1]))
        res.update(
            key=key,
            label=LABEL[key],
            final_step=float(steps[-1]),
            observed=float(loss[-1]),
            n_fit_points=int(mask.sum()),
            steps=steps,
            loss=loss,
        )
        results[key] = res
    return results


# ------------------------------------------------------------------ report ---
def print_table1(results: dict[str, dict]) -> None:
    print()
    print("=" * 94)
    print("TABLE 1 -- fitted-final validation macro bpb (matched alpha-free protocol)")
    print("=" * 94)
    print(
        f"{'Arm':<20} {'alpha':>6} {'n':>3} {'final_step':>10} "
        f"{'fitted':>8} {'observed':>9}  {'95% CI':>18} {'width':>7}"
    )
    print("-" * 94)
    for key in ORDER:
        r = results[key]
        print(
            f"{r['label']:<20} {r['alpha']:6.3f} {r['n_fit_points']:3d} "
            f"{int(r['final_step']):10d} {r['fitted_final']:8.4f} {r['observed']:9.4f}  "
            f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}] {r['ci_hi'] - r['ci_lo']:7.4f}"
        )
    print("-" * 94)
    widths = [results[k]["ci_hi"] - results[k]["ci_lo"] for k in ORDER]
    print(f"mean CI width = {np.mean(widths):.5f} bpb")

    print()
    print("Overlapping CI pairs (the only pairs not separated at 95%):")
    found = False
    for i, ka in enumerate(ORDER):
        for kb in ORDER[i + 1 :]:
            ra, rb = results[ka], results[kb]
            if not (ra["ci_hi"] < rb["ci_lo"] or rb["ci_hi"] < ra["ci_lo"]):
                print(f"  {ra['label']} <-> {rb['label']}")
                found = True
    if not found:
        print("  (none)")


def delta_stats(arm_boot: np.ndarray, base_boot: np.ndarray) -> dict:
    """Paired bootstrap difference (arm - base).

    ``p_one_sided`` is the bootstrap mass on the side OPPOSITE the point
    estimate, i.e. the probability the effect does not have the observed sign.
    ``p_two_sided`` is twice that, capped at 1. Defining it relative to the
    observed sign keeps the convention identical for arms that come out above
    and below the baseline.
    """
    d = arm_boot - base_boot
    lo, hi = np.percentile(d, [2.5, 97.5])
    point = float(d.mean())
    p_one = float((d <= 0).mean()) if point > 0 else float((d >= 0).mean())
    return {
        "delta": point,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "p_one_sided": p_one,
        "p_two_sided": min(1.0, 2 * p_one),
    }


def print_deltas(results: dict[str, dict]) -> None:
    """Paired difference CIs; bootstrap draws are paired by index."""

    def report(base_key: str, title: str) -> None:
        print()
        print("=" * 94)
        print(title)
        print("=" * 94)
        base = results[base_key]["boot"]
        for key in ORDER:
            if key == base_key:
                continue
            st = delta_stats(results[key]["boot"], base)
            flag = "" if st["ci_lo"] < 0 < st["ci_hi"] else "  significant"
            print(
                f"  {results[key]['label']:<20} delta={st['delta']:+.4f}  "
                f"95% CI [{st['ci_lo']:+.4f}, {st['ci_hi']:+.4f}]  "
                f"one-sided p={st['p_one_sided']:.4f}{flag}"
            )

    report("control", "Paired difference vs FULL-LOSS CONTROL (10k paired draws)")
    report("random_control", "Paired difference vs RANDOM CONTROL (10k paired draws)")

    # The headline null result: token selection vs a random mask of the same rate.
    st = delta_stats(results["rho_1"]["boot"], results["random_control"]["boot"])
    print()
    print(
        f"RHO-1 vs Random control: delta={st['delta']:+.4f}  "
        f"95% CI [{st['ci_lo']:+.4f}, {st['ci_hi']:+.4f}]  "
        f"one-sided p={st['p_one_sided']:.4f}  two-sided p={st['p_two_sided']:.4f}"
    )


# ----------------------------------------------------------------- figures ---
def figure1(results: dict[str, dict], fig_dir: Path) -> None:
    """Training curves, with a rescaled inset over the final steps."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(12, 6.9333), dpi=150)

    for key in ORDER:
        r = results[key]
        m = r["steps"] >= 500
        ax.plot(
            r["steps"][m],
            r["loss"][m],
            color=COLOR[key],
            linestyle=LINESTYLE[key],
            marker=MARKER[key],
            markersize=10 if MARKER[key] == "*" else 7,
            linewidth=1.8,
            label=LEGEND[key],
        )

    ax.set_xlabel("Training step", fontsize=17)
    ax.set_ylabel("Validation macro bits-per-byte (lower is better)", fontsize=15)
    ax.set_title(
        "Training curves: validation macro bpb by token-filtering arm\n"
        "(370M OLMo2, 10B-token corpus)",
        fontsize=19,
    )
    ax.set_xlim(500, 2420)
    ax.tick_params(labelsize=14)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=12.5, ncol=2, loc="upper center", bbox_to_anchor=(0.32, 0.99))

    # Rescaled inset over the final steps.
    lo_x, hi_x = 1880, 2420
    inset = ax.inset_axes((0.60, 0.50, 0.38, 0.37))
    for key in ORDER:
        r = results[key]
        m = r["steps"] >= lo_x
        inset.plot(
            r["steps"][m],
            r["loss"][m],
            color=COLOR[key],
            linestyle=LINESTYLE[key],
            marker=MARKER[key],
            markersize=9 if MARKER[key] == "*" else 6,
            linewidth=1.6,
        )
    tail = [
        v
        for key in ORDER
        for s, v in zip(results[key]["steps"], results[key]["loss"])
        if s >= lo_x and v < 1.80
    ]
    inset.set_xlim(lo_x, hi_x)
    inset.set_ylim(min(tail) - 0.008, max(tail) + 0.010)
    inset.set_title("final steps, rescaled", fontsize=12.5, style="italic")
    inset.tick_params(labelsize=11)

    # Indicate the inset region on the main axes.
    y0, y1 = inset.get_ylim()
    ax.add_patch(
        Rectangle(
            (lo_x, y0),
            hi_x - lo_x,
            y1 - y0,
            fill=False,
            edgecolor="#555555",
            linewidth=1.1,
            zorder=5,
        )
    )

    fig.tight_layout()
    _save(fig, fig_dir, "figure1_training_curves")
    plt.close(fig)


def figure2(results: dict[str, dict], fig_dir: Path) -> None:
    """Fitted-final bar chart with 95% bootstrap CI error bars."""
    import matplotlib.pyplot as plt

    fitted = np.array([results[k]["fitted_final"] for k in ORDER])
    ci_lo = np.array([results[k]["ci_lo"] for k in ORDER])
    ci_hi = np.array([results[k]["ci_hi"] for k in ORDER])

    fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
    x = np.arange(len(ORDER))

    ax.bar(
        x,
        fitted,
        width=0.8,
        color=[COLOR[k] for k in ORDER],
        edgecolor="black",
        linewidth=1.2,
        zorder=3,
    )
    ax.errorbar(
        x,
        fitted,
        yerr=[fitted - ci_lo, ci_hi - fitted],
        fmt="none",
        ecolor="black",
        elinewidth=1.8,
        capsize=6,
        capthick=1.8,
        zorder=4,
    )

    # The bars are deliberately not zero-based: the between-arm differences are
    # small relative to the absolute bpb level. Pad the CI span so the panel is
    # filled the way the published figure was.
    span = ci_hi.max() - ci_lo.min()
    ax.set_ylim(ci_lo.min() - 0.34 * span, ci_hi.max() + 0.43 * span)

    for xi, val, hi in zip(x, fitted, ci_hi):
        ax.text(
            xi,
            hi + 0.012 * span,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=21,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[k] for k in ORDER], fontsize=16, rotation=20, ha="right")
    ax.set_ylabel("Fitted-final validation macro bpb (lower is better)", fontsize=15)
    ax.set_title(
        "Final macro-bpb by arm: power-law fit ± 95% CI\n"
        "(370M OLMo2, 10B-token corpus)",
        fontsize=20,
    )
    ax.tick_params(axis="y", labelsize=14.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.tight_layout()
    _save(fig, fig_dir, "figure2_final_bar_chart")
    plt.close(fig)


def _save(fig, fig_dir: Path, stem: str) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = fig_dir / f"{stem}.{ext}"
        fig.savefig(path, facecolor="white")
        print(f"  wrote {path}")


# -------------------------------------------------------------------- json ---
# Full run provenance. The curve cache's own "source" field omits the REL-EMA
# polarity note, so it is spelled out here rather than inherited.
SOURCE_STRING = (
    "wandb eduLLM/token-selection + hpo-ladder control (re-run 2026-09-05 as "
    "full-loss-control-regmix10b-v2, replaces mixlaw-1 clone); rel_ema re-run "
    "2026-09-05 with corrected selection polarity (current-minus-history, "
    "matching rho_excess/blade), replaces the inverted-polarity run"
)

METHOD_STRING = (
    "Fit y = a + b*step^(-alpha) on steps >= 1000, profiling alpha over "
    "np.linspace(0.05, 6.0, 1192) with (a, b) from closed-form OLS at each grid "
    "point. Fitted final is evaluated at each arm's own final logged step. "
    "95% CI = 2.5/97.5 percentile of 10,000 i.i.d. residual bootstrap draws with "
    "alpha RE-ESTIMATED on every draw (alpha-free), numpy default_rng seed 0. The "
    "alpha grid was widened from the earlier [0.05, 3.0] because REL-EMA's "
    "profiled optimum was boundary-pinned at exactly 3.0, which also truncated "
    "its bootstrap; on the wider grid it settles at an interior 3.502 and no arm "
    "sits on a grid boundary. Alpha-free was chosen over alpha-fixed because it "
    "is the more conservative of the two (mean CI width 0.00989 vs 0.00700 bpb)."
)


def write_bootstrap_json(results: dict[str, dict]) -> None:
    payload = {
        "source": SOURCE_STRING,
        "metric": "macro task-loss bits per byte (lower is better)",
        "method": METHOD_STRING,
        "protocol": {
            "form": "y = a + b * step**(-alpha)",
            "fit_window": "steps >= 1000",
            "alpha_grid": "np.linspace(0.05, 6.0, 1192)",
            "n_boot": N_BOOT,
            "resample": "i.i.d. residual bootstrap on the fit-window residuals",
            "alpha_treatment": "re-estimated on every bootstrap draw (alpha-free)",
            "fitted_final_at": "each arm's own final logged step",
            "ci": "2.5/97.5 percentile of the bootstrap distribution",
            "rng_seed": SEED,
        },
        "generated_by": "experiments/token-selection/fit_and_plot.py --write-json",
        "arms": [],
    }
    for key in ORDER:
        r = results[key]
        payload["arms"].append(
            {
                "key": key,
                "label": r["label"],
                "wandb_path": f"{WANDB_PROJECT}/{WANDB_RUNS[key]}",
                "final_step": int(r["final_step"]),
                "n_fit_points": r["n_fit_points"],
                "alpha": round(r["alpha"], 4),
                "fitted_final": round(r["fitted_final"], 4),
                "observed": round(r["observed"], 4),
                "ci_lo": round(r["ci_lo"], 4),
                "ci_hi": round(r["ci_hi"], 4),
            }
        )
    BOOTSTRAP_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"  wrote {BOOTSTRAP_PATH}")


# -------------------------------------------------------------------- main ---
def main() -> None:
    ap = argparse.ArgumentParser(description="Token-selection 370M fits and figures.")
    ap.add_argument(
        "--source",
        choices=("json", "wandb"),
        default="json",
        help="curve source (default: the committed JSON cache)",
    )
    ap.add_argument("--no-figures", action="store_true", help="skip figure rendering")
    ap.add_argument(
        "--write-json",
        action="store_true",
        help="refresh token_selection_370m_bootstrap_results.json",
    )
    ap.add_argument("--fig-dir", type=Path, default=FIG_DIR)
    args = ap.parse_args()

    if args.source == "wandb":
        print("Pulling eval/macro_bpb from W&B ...")
        curves = load_curves_from_wandb()
    else:
        print(f"Reading curves from {CURVES_PATH.name}")
        curves = load_curves_from_json()

    results = fit_all(curves)
    print_table1(results)
    print_deltas(results)

    if not args.no_figures:
        print()
        print("Rendering figures ...")
        figure1(results, args.fig_dir)
        figure2(results, args.fig_dir)

    if args.write_json:
        print()
        print("Updating JSON ...")
        write_bootstrap_json(results)


if __name__ == "__main__":
    main()
