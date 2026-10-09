#!/usr/bin/env python3
"""Reproduce the robustness check on the derivative adjacency matrix in Appendix E.

The derivative arm used ``A_ij = max(0, -t_ij (L_j(r) - c_j))``, which depends on how
the fit's regularization pins the shift ``t -> t + q`` it cannot identify. Perturbing
the mixture along the simplex, ``r(eps) = (1 - eps) r + eps e_i``, gives
``A_ij = max(0, (t_bar_j(r) - t_ij)(L_j(r) - c_j))`` with
``t_bar_j(r) = sum_q r_q t_qj``, which is invariant to that shift
(``online_A_from_fit(..., gauge_invariant=True)``). Both forms are evaluated at the
domain weights in effect at each of the derivative arm's five updates.

Writes ``artifacts/appendix_e_robustness.json``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SKILLIT = ROOT / "skillit"
MIXLAW = ROOT / "mixlaw"
sys.path[:0] = [str(MIXLAW), str(SKILLIT)]
from mixlaw_common import CURVE_FAMILIES, DOMAINS  # noqa: E402
from skillit_math import (  # noqa: E402
    default_mixlaw_fit_path,
    load_fit_json,
    online_A_from_fit,
)

EDGE_EPS = 1e-4

# Domain weights in effect at the derivative arm's updates at steps 500, 875,
# 1250, 1625 and 2000 (logged p_before; same values as DERIVATIVE_UPDATES in
# plot_domain_weights_probe_vs_derivative.py).
DERIVATIVE_WEIGHTS_AT_UPDATE = {
    500: {"dclm": 0.5528504848, "arxiv": 0.2117848247, "starcoder": 0.0872393548, "pes2o": 0.0816333741, "open-web-math": 0.0417862386, "algebraic-stack": 0.0135710593, "wiki": 0.0111346329},
    875: {"dclm": 0.5561979413, "arxiv": 0.1995353103, "starcoder": 0.0821934864, "pes2o": 0.0864803717, "open-web-math": 0.0471215174, "algebraic-stack": 0.0127861174, "wiki": 0.0156852882},
    1250: {"dclm": 0.5572664142, "arxiv": 0.1886009127, "starcoder": 0.0776893422, "pes2o": 0.0910638496, "open-web-math": 0.0520632602, "algebraic-stack": 0.0120854471, "wiki": 0.0212307665},
    1625: {"dclm": 0.5562109351, "arxiv": 0.1782047153, "starcoder": 0.07340689, "pes2o": 0.0952695683, "open-web-math": 0.057228189, "algebraic-stack": 0.0114192637, "wiki": 0.0282604527},
    2000: {"dclm": 0.5533024073, "arxiv": 0.1683511883, "starcoder": 0.0693479776, "pes2o": 0.0991969332, "open-web-math": 0.0621750467, "algebraic-stack": 0.0107878549, "wiki": 0.0368385985},
}


def _compare(A: np.ndarray, reference: np.ndarray) -> dict:
    return {
        "pearson_r": float(np.corrcoef(A.ravel(), reference.ravel())[0, 1]),
        "density": float((A > EDGE_EPS).mean()),
        "reference_density": float((reference > EDGE_EPS).mean()),
        "edge_disagreements": int(((A > EDGE_EPS) != (reference > EDGE_EPS)).sum()),
        "max_abs_diff": float(np.abs(A - reference).max()),
    }


def simplex_derivative_check() -> dict:
    fit = load_fit_json(default_mixlaw_fit_path())
    per_update = {}
    for step, weights in DERIVATIVE_WEIGHTS_AT_UPDATE.items():
        r = np.array([weights[d] for d in DOMAINS])
        r = r / r.sum()
        used = online_A_from_fit(fit, r, domains=DOMAINS, families=CURVE_FAMILIES)
        simplex = online_A_from_fit(fit, r, domains=DOMAINS, families=CURVE_FAMILIES, gauge_invariant=True)
        per_update[str(step)] = _compare(simplex, used)
    return {
        "per_update": per_update,
        "min_pearson_r": min(v["pearson_r"] for v in per_update.values()),
    }


def main() -> None:
    out = {"simplex_derivative": simplex_derivative_check()}
    for step, v in out["simplex_derivative"]["per_update"].items():
        print(f"derivative at update {step:>4}: simplex vs used Pearson r = {v['pearson_r']:.3f}, "
              f"{v['edge_disagreements']}/42 edges differ")
    path = SKILLIT / "artifacts" / "appendix_e_robustness.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {path.relative_to(ROOT.parents[1]).as_posix()}")


if __name__ == "__main__":
    main()
