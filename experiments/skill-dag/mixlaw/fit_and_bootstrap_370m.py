#!/usr/bin/env python3
"""Power-law fit + alpha-free residual bootstrap for the 370M validation arms.

Reproduces Tables II and III of the paper from the committed curve file
``skill_dag_370m_wandb_curves.json``. No network or W&B access required.

Every arm is a static or dynamic 370M run on FarmShare 4xL40S with the
vendored Skill-It trainer, so all comparisons share hardware, initialization
and training code. Table III compares the two dynamic-reweighting (Skill-It)
arms against the single static 1%-floor LightGBM mixture (``lightgbm-l40s``)
they start from and share data seed with.

Method
------
For each arm we fit

    y = a + b / step**alpha

to the eval points with ``step >= 1000`` (earlier points are dominated by
initialization noise). Uncertainty is a **residual bootstrap** (Efron, 1979;
Freedman, 1981, for the regression form): residuals about the point fit are
resampled with replacement, added back to the fitted curve, and the model is
re-fit on each of ``--n-boot`` draws. The 95% CI is the 2.5th/97.5th percentile
of the resulting distribution of the fitted value at ``final_step``.

The bootstrap is **alpha-free**: ``alpha`` is re-selected on every draw rather
than frozen at the point estimate. Holding alpha fixed understates the interval
by roughly a third at this sample size, so every arm in the paper uses the
alpha-free form.

Each arm is resampled from its **own independent random stream**, spawned from
``--seed`` via ``SeedSequence.spawn``. This matters: the arms are separate
training runs with no shared randomness, so their bootstrap distributions must
be independent. Drawing every arm's resample indices from one seeded generator
(which happens by default when all arms have the same number of eval points)
silently couples them and distorts every between-arm interval -- here it made
the derivative arm's bootstrap draws correlate +0.91 with the control's and the
probe arm's -0.43, shrinking one difference interval and inflating the other.

Each run's stream index is fixed explicitly by its ``bootstrap_stream`` field
in the curve JSON (``SeedSequence(seed).spawn(bootstrap_stream_count)[index]``),
not by its position in the file or in ``VS_CONTROL``/``VS_STATIC`` below.
Probe, derivative and lightgbm-l40s keep the stream indices (5, 6, 7) they held
in earlier committed versions of this file, so their own fitted-final CIs are
bit-identical to every previously reported number; index 4 is intentionally
unused, and the four newer arms take indices 0-3. Only comparisons against the
control changed.

The Olmo-mix-1124 control is the *average of the two data seeds*: its
bootstrap distribution is the element-by-element mean of the two seeds' own
alpha-free bootstrap distributions, and its CI is the 2.5/97.5 percentiles of
that averaged distribution. The two control runs share hardware, initialization
and training code, and differ only in data seed -- though even matched runs on
this stack are not bit-reproducible (see the skillit README's run-to-run note),
so their difference (``seed_variance_estimate``) still carries some run-to-run
noise beyond pure data order.

p-values are two-sided bootstrap tests on the difference of two independent
distributions, ``p = 2 * min(P(diff <= 0), P(diff >= 0))``, floored at the
bootstrap resolution ``1/n_boot``.

Usage
-----
    python fit_and_bootstrap_370m.py                       # writes results JSON
    python fit_and_bootstrap_370m.py --n-boot 10000 --seed 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CURVES_PATH = HERE / "skill_dag_370m_wandb_curves.json"
RESULTS_PATH = HERE / "skill_dag_370m_bootstrap_results.json"

ALPHA_GRID = np.linspace(0.05, 3.0, 400)
MIN_STEP = 1000

# Arms compared against the Olmo-mix-1124 seed average.
VS_CONTROL = [
    "data-mixing-laws-paper",
    "mixlaw-fit",
    "skillit-probe",
    "skillit-derivative",
    "lightgbm-l40s",
]

# (arm, reference) pairs reported as arm - reference: both dynamic arms
# against the single static LightGBM mixture they start from.
VS_STATIC = [
    ("skillit-probe", "lightgbm-l40s"),
    ("skillit-derivative", "lightgbm-l40s"),
]


def _ols_over_alpha(X: np.ndarray, Y: np.ndarray):
    """Closed-form 2-parameter OLS of every row of Y on every row of X.

    X: (A, n) candidate regressors (step**-alpha). Y: (B, n) targets.
    Returns (sse, slope, intercept), each (A, B).
    """
    xm = X.mean(axis=1, keepdims=True)
    ym = Y.mean(axis=1, keepdims=True)
    Xc, Yc = X - xm, Y - ym
    Sxx = (Xc * Xc).sum(axis=1)
    Sxy = Xc @ Yc.T
    Syy = (Yc * Yc).sum(axis=1)
    slope = Sxy / Sxx[:, None]
    sse = Syy[None, :] - (Sxy**2) / Sxx[:, None]
    intercept = ym.T - slope * xm
    return sse, slope, intercept


def fit_and_bootstrap(steps, losses, *, final_step, n_boot, seed, chunk=20_000,
                       return_params=False):
    """Point fit + alpha-free residual bootstrap. Returns (fitted, finals).

    ``seed`` should be a per-arm :class:`numpy.random.SeedSequence` so that arms
    are resampled independently; see the module docstring. Draws are generated
    in blocks of ``chunk`` to bound peak memory at large ``n_boot``.

    With ``return_params=True``, also returns the per-draw ``(a, b, alpha)``
    arrays (each shape ``(n_boot,)``), so a caller can evaluate the bootstrapped
    power law at any step, not just ``final_step`` -- e.g. to bootstrap a
    step-count crossing between two arms (see ``bootstrap_crossing`` below).
    The returned tuple is then ``(fitted, finals, a_draws, b_draws, alpha_draws)``;
    default behavior and every existing caller are unaffected.
    """
    s = np.asarray(steps, dtype=float)
    y = np.asarray(losses, dtype=float)
    mask = s >= MIN_STEP
    s, y = s[mask], y[mask]
    if s.size < 3:
        raise ValueError(f"need >=3 points with step >= {MIN_STEP}, got {s.size}")

    X = s[None, :] ** (-ALPHA_GRID[:, None])          # (A, n)
    sse0, slope0, icept0 = _ols_over_alpha(X, y[None, :])
    best = int(np.argmin(sse0[:, 0]))
    alpha0 = float(ALPHA_GRID[best])
    a0, b0 = float(icept0[best, 0]), float(slope0[best, 0])
    pred = a0 + b0 * s ** (-alpha0)
    resid = y - pred
    fitted = a0 + b0 * final_step ** (-alpha0)

    rng = np.random.default_rng(seed)
    finals = np.empty(n_boot, dtype=float)
    if return_params:
        a_draws = np.empty(n_boot, dtype=float)
        b_draws = np.empty(n_boot, dtype=float)
        alpha_draws = np.empty(n_boot, dtype=float)
    for lo in range(0, n_boot, chunk):
        hi = min(lo + chunk, n_boot)
        draws = rng.integers(0, resid.size, (hi - lo, resid.size))
        Y = pred[None, :] + resid[draws]               # (B, n)
        sse, slope, icept = _ols_over_alpha(X, Y)
        pick = np.argmin(sse, axis=0)                  # alpha-free: per-draw alpha
        cols = np.arange(hi - lo)
        finals[lo:hi] = icept[pick, cols] + slope[pick, cols] * final_step ** (-ALPHA_GRID[pick])
        if return_params:
            a_draws[lo:hi] = icept[pick, cols]
            b_draws[lo:hi] = slope[pick, cols]
            alpha_draws[lo:hi] = ALPHA_GRID[pick]
    if return_params:
        return fitted, finals, a_draws, b_draws, alpha_draws
    return fitted, finals


def bootstrap_crossing(a_from, b_from, alpha_from, target, *, at_or_above=False):
    """Per-draw step at which ``a_from + b_from / step**alpha_from`` first
    reaches ``target``, given per-draw fit parameters and a per-draw target
    value (e.g. the other arm's own bootstrapped final, paired draw-by-draw
    as ``ci``/``diff_p`` already pair independent arms' distributions).

    The power law is monotonic in ``step`` for ``b > 0, alpha > 0`` (decreasing
    as step grows), so the crossing has a closed form:
    ``step = (b / (target - a)) ** (1 / alpha)``. Used for two directions:

    - **A given arm reaches a fixed target from above** (the "N% fewer steps"
      metric): the fitted curve is decreasing, so it reaches ``target`` once
      ``target > a`` (otherwise the curve never gets that high again past the
      fit window, and by construction ``target`` -- the other arm's own
      final -- should already be within the descending regime).
    - **Extrapolating an arm forward until it reaches a lower target it
      hasn't achieved within the observed budget** (the "N times longer"
      metric): same formula, just typically evaluated far beyond
      ``final_step``; ``at_or_above=True`` documents that direction for
      readability at the call site, but the math is identical.

    A row where ``target <= a`` (asymptote at or above the target -- this arm's
    fitted curve never reaches it, even in the infinite-step limit) has no
    finite crossing and is returned as ``np.inf``, so it never spuriously wins
    a ``min``/percentile and shows up as a visibly non-finite tail instead of a
    silently wrong number. Callers should report the fraction of non-finite
    draws alongside the CI.
    """
    a_from = np.asarray(a_from, dtype=float)
    b_from = np.asarray(b_from, dtype=float)
    alpha_from = np.asarray(alpha_from, dtype=float)
    target = np.asarray(target, dtype=float)
    gap = target - a_from
    with np.errstate(divide="ignore", invalid="ignore"):
        step = np.where(gap > 0, (b_from / np.maximum(gap, 1e-300)) ** (1.0 / alpha_from), np.inf)
    return step


def ci(finals):
    lo, hi = np.percentile(finals, [2.5, 97.5])
    return float(lo), float(hi)


def diff_p(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided bootstrap p-value for a - b, floored at 1/n.

    ``a`` and ``b`` are independent bootstrap distributions (one per arm), so
    the difference is taken draw-by-draw only to build its sampling
    distribution -- the draws are not paired observations.
    """
    d = a - b
    n = d.size
    p = 2.0 * min((d >= 0).mean(), (d <= 0).mean())
    return float(max(min(p, 1.0), 1.0 / n))


def diff_p_one_sided(a: np.ndarray, b: np.ndarray) -> float:
    """One-sided bootstrap p-value that arm ``a`` has lower loss than ``b``, floored at 1/n.

    The fraction of draws of ``a - b`` that are >= 0 (same independent-draws
    construction as ``diff_p``). It equals half the two-sided ``diff_p`` whenever the
    mean difference is negative; an arm that is not better has no one-sided p-value
    worth quoting (``fmt_p_one_sided`` prints an em dash for it).
    """
    d = a - b
    return float(max((d >= 0).mean(), 1.0 / d.size))


def fmt_p(p: float, n_boot: int) -> str:
    floor = 1.0 / n_boot
    return f"p < {floor:g}" if p <= floor else f"p = {p:.4f}"


def fmt_p_one_sided(p: float, mean_diff: float, n_boot: int) -> str:
    """Table form of a one-sided p: an em dash when the arm did not beat the reference."""
    return "—" if mean_diff >= 0 else fmt_p(p, n_boot)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--curves", type=Path, default=CURVES_PATH)
    ap.add_argument("--out", type=Path, default=RESULTS_PATH)
    args = ap.parse_args()

    data = json.loads(args.curves.read_text(encoding="utf-8"))
    final_step = int(data["final_step"])
    runs = data["runs"]

    # One independent stream per arm, indexed by each run's own
    # bootstrap_stream field rather than by position -- see the module
    # docstring for why.
    stream_count = int(data.get("bootstrap_stream_count", len(runs)))
    streams = np.random.SeedSequence(args.seed).spawn(stream_count)

    fitted, finals, observed = {}, {}, {}
    for key, run in runs.items():
        stream = streams[run["bootstrap_stream"]]
        f, dist = fit_and_bootstrap(
            run["steps"], run["macro_bpb"],
            final_step=final_step, n_boot=args.n_boot, seed=stream,
        )
        fitted[key], finals[key] = f, dist
        observed[key] = float(run["macro_bpb"][-1])

    # Control = element-by-element average of the two seeds' bootstrap draws.
    s1, s2 = "olmo-mix-1124-s42", "olmo-mix-1124-s69"
    finals["olmo-mix-1124-average"] = 0.5 * (finals[s1] + finals[s2])
    fitted["olmo-mix-1124-average"] = 0.5 * (fitted[s1] + fitted[s2])
    observed["olmo-mix-1124-average"] = 0.5 * (observed[s1] + observed[s2])

    out = {
        "method": "power-law fit (y = a + b/step**alpha) on steps >= 1000; "
                  "alpha-free residual bootstrap; one independent resampling "
                  "stream per arm",
        "n_boot": args.n_boot,
        "seed": args.seed,
        "final_step": final_step,
        "alpha_grid": {"lo": 0.05, "hi": 3.0, "n": int(ALPHA_GRID.size)},
        "arms": {},
        "comparisons": {},
    }

    order = list(runs) + ["olmo-mix-1124-average"]
    print(f"{'arm':28s} {'fitted':>8s} {'observed':>9s}   95% CI")
    for key in order:
        lo, hi = ci(finals[key])
        label = runs[key]["label"] if key in runs else "Olmo-mix-1124 average"
        out["arms"][key] = {
            "label": label,
            "fitted_final": round(fitted[key], 6),
            "observed_final": round(observed[key], 6),
            "ci95": [round(lo, 6), round(hi, 6)],
        }
        print(f"{key:28s} {fitted[key]:8.4f} {observed[key]:9.4f}   [{lo:.4f}, {hi:.4f}]")

    ctrl = finals["olmo-mix-1124-average"]
    print()
    for key in VS_CONTROL:
        d = finals[key] - ctrl
        p = diff_p(finals[key], ctrl)
        lo, hi = ci(d)
        p1 = diff_p_one_sided(finals[key], ctrl)
        out["comparisons"][f"{key}_vs_olmo_average"] = {
            "mean_diff_bpb": round(float(d.mean()), 6),
            "ci95": [round(lo, 6), round(hi, 6)],
            "p_value": p,
            "p_display": fmt_p(p, args.n_boot),
            "p_one_sided": p1,
            "p_one_sided_display": fmt_p_one_sided(p1, float(d.mean()), args.n_boot),
        }
        print(f"{key:28s} vs olmo avg: {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]  "
              f"{fmt_p(p, args.n_boot)} two-sided, {fmt_p_one_sided(p1, float(d.mean()), args.n_boot)} one-sided")

    # Dynamic arms against the static LightGBM mixture they start from.
    print()
    for key, ref in VS_STATIC:
        d = finals[key] - finals[ref]
        p = diff_p(finals[key], finals[ref])
        lo, hi = ci(d)
        p1 = diff_p_one_sided(finals[key], finals[ref])
        out["comparisons"][f"{key}_vs_{ref}"] = {
            "mean_diff_bpb": round(float(d.mean()), 6),
            "ci95": [round(lo, 6), round(hi, 6)],
            "p_value": p,
            "p_display": fmt_p(p, args.n_boot),
            "p_one_sided": p1,
            "p_one_sided_display": fmt_p_one_sided(p1, float(d.mean()), args.n_boot),
        }
        print(f"{key:20s} vs {ref:14s}: {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]  "
              f"{fmt_p(p, args.n_boot)} two-sided, {fmt_p_one_sided(p1, float(d.mean()), args.n_boot)} one-sided")

    # Seed-variance estimate quoted in the paper.
    d = finals[s1] - finals[s2]
    p = diff_p(finals[s1], finals[s2])
    lo, hi = ci(d)
    out["seed_variance_estimate"] = {
        "description": "Olmo-mix-1124 data seed 42 minus data seed 69 (same hardware, "
                       "initialization and training code; not a pure data-order "
                       "contrast, since matched runs on this stack are not "
                       "bit-reproducible)",
        "mean_diff_bpb": round(float(d.mean()), 6),
        "ci95": [round(lo, 6), round(hi, 6)],
        "p_value": p,
        "p_display": fmt_p(p, args.n_boot),
    }
    print(f"\nseed variance (42 - 69): {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]  {fmt_p(p, args.n_boot)}")

    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
