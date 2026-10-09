#!/usr/bin/env python3
"""Print the offline (probe) and online (derivative) adjacency matrices side by side.

The online matrix is a *snapshot*: ``A_ij = max(0, -t_ij (L_j(r) - c_j))`` depends
on the mixture ``r`` it is evaluated at, and the derivative arm recomputes it at the
current weights on every update. It is evaluated here at ``LGB-min1pct``, the
LightGBM-optimized mixture both Skill-It arms start from (the matrix the derivative arm
uses at its first update, step 500, and the one Figure IV plots). The offline matrix is
the probe arm's: each one-hot probe's extrapolated loss compared with the probe trained
on the same LightGBM mixture (``build_adjacency.py --reference-run probe_lgb_start``),
so both panels refer to the same mixture.

Prints both matrices, the number of nonzero cells in each, the Pearson correlation
between them, and the number of cells that disagree on whether an edge is present.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "mixlaw"), str(ROOT / "skillit")]
from mixlaw_common import CURVE_FAMILIES, DOMAINS  # noqa: E402
from skillit_math import default_mixlaw_fit_path, load_fit_json, online_A_from_fit  # noqa: E402

LGB_MIN1PCT = {
    "dclm": 0.5529,
    "arxiv": 0.2118,
    "starcoder": 0.0872,
    "pes2o": 0.0816,
    "open-web-math": 0.0418,
    "algebraic-stack": 0.0136,
    "wiki": 0.0111,
}


def _describe(A: np.ndarray, reference: np.ndarray, label: str) -> None:
    nonzero = int((A > 0).sum())
    disagree = int(((A > 0) != (reference > 0)).sum())
    r = float(np.corrcoef(A.ravel(), reference.ravel())[0, 1])
    print(
        f"{label}: {nonzero}/{A.size} nonzero ({nonzero / A.size:.1%} dense), "
        f"Pearson r vs offline = {r:.3f}, {disagree}/{A.size} cells disagree on edge presence"
    )


def main() -> None:
    fit = load_fit_json(default_mixlaw_fit_path())
    A_off = np.load(Path(__file__).parent / "artifacts/probes_full/A_offline.npy")
    r = np.array([LGB_MIN1PCT[d] for d in DOMAINS], dtype=np.float64)
    matrices = [
        ("OFFLINE (probe)", A_off),
        ("ONLINE @ LGB-min1pct (derivative; the point Figure IV plots)", online_A_from_fit(fit, r)),
    ]

    width = max(len(d) for d in DOMAINS)
    for name, A in matrices:
        print(f"\n{name}")
        print(" " * width, *[f"{f[:9]:>9s}" for f in CURVE_FAMILIES])
        for i, d in enumerate(DOMAINS):
            print(f"{d:<{width}}", *[f"{A[i, j]:9.4f}" for j in range(len(CURVE_FAMILIES))])

    print()
    _describe(matrices[0][1], A_off, "OFFLINE (probe)")
    _describe(matrices[1][1], A_off, "ONLINE @ LGB-min1pct")


if __name__ == "__main__":
    main()
