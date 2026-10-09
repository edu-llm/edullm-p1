#!/usr/bin/env python3
"""Build offline Skill-It adjacency A from the seven one-hot DataDecide-60M probe runs.

Pipeline:
  1. Collect each ``../domain_probes/runs/onehot_<domain>/task_loss.jsonl`` (the in-run test
     curves, steps 120-1440, every item of each test label; the last point is the final loss).
  2. Chinchilla-extrapolate each curve family to step 5806 (tpp=20) via
     ``mixlaw/extrapolate_chinchilla.py`` logic.
  3. ``A_ij = max(0, L_j(ref) - L_i_j)`` for domains i and families j. By default
     ``L_j(ref)`` is the MixLaw fit's prediction (``mixlaw_fit_chinchilla.json``) at the
     Data Mixing Laws paper mixture, which this code calls ``regmix``. With
     ``--reference-run <name>`` it is instead the extrapolated loss of that 60M run (for
     example one trained at the LightGBM starting mixture), under the same protocol as the
     one-hot probes.
  4. Write ``artifacts/A_offline.npy`` plus named JSON on runtime scratch and
     upload the directory to W&B.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_SKILLIT = Path(__file__).resolve().parent
_MIXLAW = _SKILLIT.parent / "mixlaw"
_TS_ROOT = _SKILLIT.parents[1] / "token-selection"
for p in (_MIXLAW, _SKILLIT, _TS_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from extrapolate_chinchilla import extrapolate_runs  # noqa: E402
from mixlaw_common import (  # noqa: E402
    CURVE_FAMILIES,
    DOMAIN_PROBES_DIR,
    DOMAINS,
    load_probe_run,
    macro_curve,
    token_budget,
)
from skillit_math import (  # noqa: E402
    default_mixlaw_fit_path,
    load_fit_json,
    offline_A_from_extrapolated,
    regmix_family_losses_from_fit,
)

CHINCHILLA_STEP = 5806  # token_budget(20) → 5806


def collect_probe_runs(runs_dir: Path, reference_run: str | None = None) -> dict:
    """Gather the seven one-hot runs (plus the reference run, if named) into a
    mixlaw_data-compatible payload. One-hot runs are keyed ``probe_<domain>`` downstream."""
    registry = json.loads((DOMAIN_PROBES_DIR / "runs.json").read_text(encoding="utf-8"))
    weights_by_name = {r["name"]: r["weights"] for r in registry["runs"]}
    names = [f"onehot_{d}" for d in DOMAINS] + ([reference_run] if reference_run else [])
    runs, missing = [], []
    for i, name in enumerate(names):
        run_dir = runs_dir / name
        if not (run_dir / "task_loss.jsonl").is_file():
            missing.append(name)
            continue
        run = load_probe_run(run_dir)
        runs.append(
            {
                "id": i,
                "tag": name,
                "run_name": f"probe_{name.removeprefix('onehot_')}" if name.startswith("onehot_") else name,
                "weights": weights_by_name.get(name) or (run["meta"] or {}).get("weights", []),
                "task_loss_labels": run["task_loss_labels"],
                "task_loss_families": run["task_loss_families"],
                "macro_mean": macro_curve(run["task_loss_families"]),
                "curve": run["curve"],
            }
        )
    if missing:
        raise SystemExit(f"missing finished runs under {runs_dir}: {missing}")
    return {
        "domain_order": list(DOMAINS),
        "curve_families": list(CURVE_FAMILIES),
        "runs": runs,
    }


def build_A_from_extrapolated(
    report: dict,
    L_reg: dict[str, float],
    *,
    chinchilla_step: int | None = None,
    reference_label: str = "regmix",
) -> tuple[np.ndarray, dict]:
    """Construct A (7×6) vs the reference losses (RegMix fit unless named otherwise)."""
    return offline_A_from_extrapolated(
        report,
        L_reg,
        domains=DOMAINS,
        families=CURVE_FAMILIES,
        reference_label=reference_label,
        chinchilla_step=chinchilla_step or report.get("chinchilla_steps", CHINCHILLA_STEP),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--runs-dir",
        type=Path,
        default=DOMAIN_PROBES_DIR / "runs",
        help="Root with <run_name>/task_loss.jsonl (default: ../domain_probes/runs)",
    )
    ap.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Optional pre-collected mixlaw_data.json (skips collect)",
    )
    ap.add_argument(
        "--fit-json",
        type=Path,
        default=None,
        help="Mixlaw fit for L_j(r_RegMix) (default: mixlaw/mixlaw_fit_chinchilla.json)",
    )
    ap.add_argument(
        "--reference-run",
        default=None,
        help=(
            "Use this run (a directory under --runs-dir, e.g. one trained at the LightGBM "
            "starting mixture) as the reference L_j instead of the MixLaw fit; it is "
            "extrapolated to the Chinchilla step like the one-hot probes"
        ),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=_SKILLIT / "artifacts",
        help="Write A_offline.npy plus named JSON artifacts here",
    )
    ap.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "skillit"))
    ap.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY") or None)
    ap.add_argument("--wandb-run-name", default="skillit-build-offline-A")
    ap.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=os.environ.get("WANDB_MODE", "online"),
    )
    ap.add_argument(
        "--allow-local-only",
        action="store_true",
        help="Explicit local analysis mode; permits offline/disabled W&B.",
    )
    ap.add_argument("--step", type=int, default=CHINCHILLA_STEP)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--write-intermediate",
        action="store_true",
        help="Also write collected + extrapolated JSON under out-dir",
    )
    args = ap.parse_args()
    if not args.allow_local_only and args.wandb_mode != "online":
        ap.error(
            "production artifact publication requires --wandb-mode online; "
            "use --allow-local-only only for local analysis"
        )

    fit_path = args.fit_json or default_mixlaw_fit_path()
    L_reg = None
    if args.reference_run is None:
        fit = load_fit_json(fit_path)
        L_reg = regmix_family_losses_from_fit(fit, domains=DOMAINS, families=CURVE_FAMILIES)

    if args.data is not None:
        data = json.loads(args.data.read_text(encoding="utf-8"))
    else:
        data = collect_probe_runs(args.runs_dir, args.reference_run)

    _, default_chin_step, _ = token_budget(20.0)
    target_step = int(args.step) if args.step is not None else default_chin_step
    report = extrapolate_runs(data, target_step, seed=args.seed)
    reference_label = "regmix"
    if args.reference_run is not None:
        reference_label = args.reference_run
        ref = next((r for r in report["runs"] if r["run_name"] == reference_label), None)
        if ref is None:
            raise SystemExit(f"reference run {reference_label!r} is not in the collected probe data")
        L_reg = {}
        for fam in CURVE_FAMILIES:
            loss = ref["families"][fam].get("chinchilla")
            if loss is None:
                raise SystemExit(f"{reference_label}::{fam}: no Chinchilla extrapolation ({ref['families'][fam].get('note')})")
            L_reg[fam] = float(loss)
    A, detail = build_A_from_extrapolated(
        report, L_reg, chinchilla_step=target_step, reference_label=reference_label
    )
    if args.reference_run is None:
        detail["fit_json"] = str(fit_path)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    npy_path = args.out_dir / "A_offline.npy"
    json_path = args.out_dir / "adjacency.json"
    publish_json_path = args.out_dir / "A_offline.json"
    np.save(npy_path, A)
    payload = {
        **detail,
        "npy": str(npy_path.name),
        "shape": list(A.shape),
        "eta_default": 0.2,
        "w_default": 1.0,
    }
    json_text = json.dumps(payload, indent=2) + "\n"
    json_path.write_text(json_text, encoding="utf-8")
    publish_json_path.write_text(json_text, encoding="utf-8")
    print(f"wrote {npy_path} shape={A.shape}")
    print(f"wrote {json_path}")
    print(f"wrote {publish_json_path}")

    if args.write_intermediate:
        (args.out_dir / "probe_data.json").write_text(
            json.dumps(data, indent=2) + "\n", encoding="utf-8"
        )
        (args.out_dir / "probe_chinchilla_extrapolated.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote intermediate JSONs under {args.out_dir}")

    if args.wandb_mode != "disabled":
        if args.wandb_mode == "online" and not os.environ.get("WANDB_API_KEY"):
            raise SystemExit("WANDB_API_KEY is required for online artifact publication")
        try:
            import wandb
        except ImportError as exc:
            if not args.allow_local_only:
                raise SystemExit("wandb is required for artifact publication") from exc
            print("wandb unavailable; artifacts remain on local scratch")
        else:
            os.environ.setdefault("WANDB_MODE", args.wandb_mode)
            run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name,
                job_type="skillit-analysis",
                config={"chinchilla_step": target_step, "seed": args.seed},
            )
            try:
                artifact = wandb.Artifact("skillit-offline-A", type="analysis")
                artifact.add_dir(str(args.out_dir))
                logged = run.log_artifact(artifact)
                wait = getattr(logged, "wait", None)
                if callable(wait):
                    wait()
            finally:
                run.finish()
    elif not args.allow_local_only:
        raise SystemExit("W&B cannot be disabled for production artifact publication")

    print("\nA (rows=domains, cols=families); positive = domain beats the reference @ Chinchilla:")
    hdr = " ".join(f"{f[:8]:>8}" for f in CURVE_FAMILIES)
    print(f"{'domain':<18} {hdr}")
    for i, d in enumerate(DOMAINS):
        row = " ".join(f"{A[i, j]:8.4f}" for j in range(len(CURVE_FAMILIES)))
        print(f"{d:<18} {row}")


if __name__ == "__main__":
    main()
