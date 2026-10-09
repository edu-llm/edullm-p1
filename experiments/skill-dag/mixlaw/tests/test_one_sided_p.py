"""One-sided p-values in the 370M bootstrap: half the two-sided p when the arm is better,
an em dash when it is not."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fit_and_bootstrap_370m import diff_p, diff_p_one_sided, fmt_p_one_sided  # noqa: E402


def _draws(mean: float, n: int = 100_000, sd: float = 0.01, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(mean, sd, n)


def test_one_sided_p_is_half_the_two_sided_p_when_the_arm_is_better():
    arm, ref = _draws(1.60, seed=1), _draws(1.61, seed=2)
    assert np.isclose(diff_p_one_sided(arm, ref), diff_p(arm, ref) / 2)
    assert diff_p_one_sided(arm, ref) < 0.5


def test_one_sided_p_is_floored_at_one_over_n():
    arm, ref = np.full(1000, 1.0), np.full(1000, 2.0)
    assert diff_p_one_sided(arm, ref) == 1.0 / 1000


def test_em_dash_when_the_arm_did_not_beat_the_reference():
    arm, ref = _draws(1.62, seed=3), _draws(1.61, seed=4)
    p1 = diff_p_one_sided(arm, ref)
    assert fmt_p_one_sided(p1, float((arm - ref).mean()), 100_000) == "—"


def test_display_matches_the_two_sided_format_when_better():
    arm, ref = _draws(1.60, seed=5), _draws(1.61, seed=6)
    p1 = diff_p_one_sided(arm, ref)
    shown = fmt_p_one_sided(p1, float((arm - ref).mean()), 100_000)
    assert shown.startswith("p ")
