#!/usr/bin/env python3
"""Plot the Chinchilla-extrapolated probe task-loss curves of a finished adjacency build.

Re-plots ``--artifacts-dir`` (as written by ``build_adjacency.py --write-intermediate``),
labeling the reference by the run it was measured on.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_SCRIPT = Path(__file__).resolve().parent
_MIXLAW = _SCRIPT.parent / "mixlaw"
for p in (_MIXLAW, _SCRIPT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fit_mixing_law import extrapolate  # noqa: E402
from mixlaw_common import CURVE_FAMILIES, DOMAINS, PROBE_STEPS, task_family  # noqa: E402

CHINCHILLA_STEP = 5806
PILOT_STEP = PROBE_STEPS

ONEHOT_PROBES: tuple[str, ...] = tuple(f"probe_{d}" for d in DOMAINS)


def family_series(curve: list[dict], family: str) -> tuple[list[int], list[float]]:
    steps, vals = [], []
    for pt in curve:
        losses = pt.get("task_loss_bpb") or {}
        for label, v in losses.items():
            if task_family(label) == family:
                steps.append(int(pt["step"]))
                vals.append(float(v))
                break
    return steps, vals


def plot_chinchilla_curves(
    report: dict, L_reg: dict[str, float], out_path: Path, ref_label: str = "RegMix"
) -> None:
    cmap = plt.get_cmap("tab10")
    colors = {run["run_name"]: cmap(i) for i, run in enumerate(report["runs"])}

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    chin_step = int(report["chinchilla_steps"])
    eval_steps = np.arange(120, PILOT_STEP + 1, 120)
    smooth = np.linspace(120, chin_step, 200)

    for ax, fam in zip(axes.flat, CURVE_FAMILIES):
        for run in report["runs"]:
            dom = run["run_name"].replace("probe_", "")
            entry = run["families"][fam]
            pts = entry.get("curve_points") or []
            if not pts:
                continue
            xs = [int(p["step"]) for p in pts]
            ys = [float(p["loss"]) for p in pts]
            color = colors[run["run_name"]]
            ax.plot(xs, ys, "o-", ms=3, lw=1.2, color=color, label=dom)
            law = entry.get("step_law")
            chin = entry.get("chinchilla")
            if law is not None and chin is not None:
                ext_y = [extrapolate(law, int(s)) for s in smooth]
                ax.plot(smooth, ext_y, "--", lw=1.0, color=color, alpha=0.75)
                ax.scatter([chin_step], [chin], marker="*", s=70, color=color, zorder=5)
        ax.axhline(L_reg[fam], color="0.35", ls=":", lw=1.2, alpha=0.9)
        ax.set_title(fam.replace("_", " "))
        ax.grid(True, alpha=0.25)
        ax.set_ylabel("task loss (bpb)")

    for ax in axes[1]:
        ax.set_xlabel("training step")
    axes[0, 0].set_xlim(0, chin_step * 1.02)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.suptitle(
        "Skill-It one-hot probes: measured evals (solid) + Chinchilla extrapolation "
        f"(dashed → step {chin_step}); dotted = {ref_label} reference",
        fontsize=11,
        y=0.99,
    )
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=4,
        fontsize=8,
        frameon=False,
    )
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_chinchilla_macro(
    report: dict, L_reg: dict[str, float], out_path: Path, ref_label: str = "RegMix"
) -> None:
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(10, 5))
    reg_macro = float(np.mean([L_reg[f] for f in CURVE_FAMILIES]))
    chin_step = int(report["chinchilla_steps"])
    smooth = np.linspace(120, chin_step, 200)

    for i, run in enumerate(report["runs"]):
        dom = run["run_name"].replace("probe_", "")
        color = cmap(i)
        macro_pts_x, macro_pts_y = [], []
        by_step: dict[int, list[float]] = {}
        for fam in CURVE_FAMILIES:
            for p in run["families"][fam].get("curve_points") or []:
                by_step.setdefault(int(p["step"]), []).append(float(p["loss"]))
        for step in sorted(by_step):
            macro_pts_x.append(step)
            macro_pts_y.append(float(np.mean(by_step[step])))
        if macro_pts_x:
            ax.plot(macro_pts_x, macro_pts_y, "o-", ms=3, lw=1.2, color=color, label=dom)
        chin_vals = [
            float(run["families"][fam]["chinchilla"])
            for fam in CURVE_FAMILIES
            if run["families"][fam].get("chinchilla") is not None
        ]
        if len(chin_vals) == len(CURVE_FAMILIES):
            chin_macro = float(np.mean(chin_vals))
            ext_y = []
            for s in smooth:
                vals = []
                for fam in CURVE_FAMILIES:
                    law = run["families"][fam].get("step_law")
                    if law is not None:
                        vals.append(extrapolate(law, int(s)))
                if vals:
                    ext_y.append(float(np.mean(vals)))
            if ext_y:
                ax.plot(smooth[: len(ext_y)], ext_y, "--", lw=1.0, color=color, alpha=0.75)
            ax.scatter([chin_step], [chin_macro], marker="*", s=70, color=color, zorder=5)

    ax.axhline(reg_macro, color="0.35", ls=":", lw=1.2, label=f"{ref_label} (chin)")
    ax.set_xlabel("training step")
    ax.set_ylabel("macro mean task loss (bpb)")
    ax.set_title(f"Macro mean over 6 families (Chinchilla @ step {chin_step})")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"wrote {out_path}")


def print_A(A: np.ndarray, L_reg: dict[str, float]) -> None:
    print("\nL_j(RegMix) @ Chinchilla:")
    for fam in CURVE_FAMILIES:
        print(f"  {fam:22s} {L_reg[fam]:.4f}")
    print("\nA_ij = max(0, L_j(RegMix) - L_j(i)) @ Chinchilla step 5806")
    hdr = " ".join(f"{f[:10]:>10}" for f in CURVE_FAMILIES)
    print(f"{'domain':<18} {hdr}")
    for i, dom in enumerate(DOMAINS):
        row = " ".join(f"{A[i, j]:10.4f}" for j in range(len(CURVE_FAMILIES)))
        print(f"{dom:<18} {row}")


if __name__ == "__main__":
    main()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--artifacts-dir",
        type=Path,
        required=True,
        help="directory holding probe_chinchilla_extrapolated.json + A_offline.json",
    )
    args = ap.parse_args()
    report = json.loads((args.artifacts_dir / "probe_chinchilla_extrapolated.json").read_text(encoding="utf-8"))
    detail = json.loads((args.artifacts_dir / "A_offline.json").read_text(encoding="utf-8"))
    L_ref = {fam: float(v) for fam, v in detail["reference_losses"].items()}
    label = detail["reference"]
    plot_chinchilla_curves(report, L_ref, args.artifacts_dir / "task_loss_chinchilla_by_family.png", label)
    plot_chinchilla_macro(report, L_ref, args.artifacts_dir / "task_loss_chinchilla_macro.png", label)
    print_A(np.array(detail["A"]) if "A" in detail else np.load(args.artifacts_dir / "A_offline.npy"), L_ref)


if __name__ == "__main__":
    main()
