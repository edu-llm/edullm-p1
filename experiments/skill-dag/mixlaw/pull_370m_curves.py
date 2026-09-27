#!/usr/bin/env python3
"""Pull the 370M validation-arm curves from W&B and write the three committed
curve files this repo's fits and figures read.

Replaces five hand-assembled, overlapping curve files with one pull. Modeled
on ``experiments/token-selection/fit_and_plot.py``'s ``load_curves_from_wandb``.

Arms and where they log
------------------------
Two dynamic (Skill-It) arms log to ``eduLLM/skillit``, unchanged since they
predate this rerun. The five static arms -- the matched 4x L40S LightGBM
rerun and the four new all-L40S validation runs -- all log to
``eduLLM/mixlaw-new``. The four new runs are resolved by their exact W&B
display name rather than a hardcoded run ID, since their IDs are assigned by
W&B at creation and are not known until each run starts:

    static-olmo-mix-1124-s42-farmshare-1745704
    static-olmo-mix-1124-s69-farmshare-1745710
    static-mix01-s42-farmshare-1745708
    static-ml-min1pct-s42-farmshare-1745706

Per-run quirks handled here
----------------------------
- The probe run's W&B history stops at step 2383 (job 1730368 was cancelled
  right after the step-2384 checkpoint and eval were saved). Its step-2384
  row was appended to the run's own history after the fact (see
  ``artifacts/probe_arm/step2384_task_loss.json`` and the PROVENANCE note on
  ``eduLLM/mixlaw-new``'s "Added after vendoring" section); this script reads
  it from W&B like any other row and does not need a file fallback.
- The probe run's history additionally contains exact duplicate rows for
  steps 2000-2383 after that append (a resume side effect). Every arm is
  de-duplicated by step, keeping the last logged row for a given step.
- Summary values differ from history values by one ULP on some keys (a W&B
  serialization quirk observed on zgmte13g, the probe run, and its clone).
  This script reads only ``scan_history``, never ``run.summary``.

Every macro value is checked against the mean of its own 20 per-label values,
and the whole ladder (0, 125, ..., 2250, 2384) is required for every arm.

Usage
-----
    python pull_370m_curves.py               # pull from W&B, write all three files
    python pull_370m_curves.py --dry-run      # pull and validate, write nothing
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CURVES_PATH = HERE / "skill_dag_370m_wandb_curves.json"
HELDOUT_LABEL_PATH = HERE / "heldout_label_curves.json"
HELDOUT_PERLABEL_PATH = HERE / "heldout_perlabel_curves.json"

EVAL_LADDER = [0, 125, 250, 375, 500, 625, 750, 875, 1000, 1125, 1250, 1375,
               1500, 1625, 1750, 1875, 2000, 2125, 2250, 2384]

# The six fitted-skill families supply these 12 (task, split) labels.
TARGETED_LABELS = [
    "arc_challenge_val_rc_5shot_bpb", "arc_challenge_test_rc_5shot_bpb",
    "arc_easy_val_rc_5shot_bpb", "arc_easy_test_rc_5shot_bpb",
    "mmlu_humanities_val_rc_5shot_bpb", "mmlu_humanities_test_rc_5shot_bpb",
    "mmlu_other_val_rc_5shot_bpb", "mmlu_other_test_rc_5shot_bpb",
    "mmlu_social_sciences_val_rc_5shot_bpb", "mmlu_social_sciences_test_rc_5shot_bpb",
    "mmlu_stem_val_rc_5shot_bpb", "mmlu_stem_test_rc_5shot_bpb",
]
# The remaining 8 labels were never optimized against.
NEVER_TARGETED_LABELS = [
    "boolq_val_rc_5shot_bpb", "csqa_val_rc_5shot_bpb", "hellaswag_val_rc_5shot_bpb",
    "openbookqa_val_rc_5shot_bpb", "openbookqa_test_rc_5shot_bpb",
    "piqa_val_rc_5shot_bpb", "socialiqa_val_rc_5shot_bpb", "winogrande_val_rc_5shot_bpb",
]
ALL_LABELS = TARGETED_LABELS + NEVER_TARGETED_LABELS
assert len(ALL_LABELS) == 20 and len(set(ALL_LABELS)) == 20

HISTORY_KEYS = ["_step", "eval/macro_bpb"] + [f"eval/bpb/{label}" for label in ALL_LABELS]

# key -> (wandb_project, id_or_None, display_name_or_None). Exactly one of
# id/display_name is set; display_name is resolved to a run at pull time.
RUNS = {
    "skillit-probe": ("skillit", "87ad0201c4b5781a3df50d7bb394776c", None,
                       "Skill-It offline probe"),
    "skillit-derivative": ("skillit", "c0844ce36f24d6773c7f45cb31d810f4", None,
                            "Skill-It online derivative"),
    "lightgbm-l40s": ("mixlaw-new", "zgmte13g", None,
                       "LightGBM fit (ours), 4x L40S rerun"),
    "olmo-mix-1124-s42": ("mixlaw-new", None, "static-olmo-mix-1124-s42-farmshare-1745704",
                           "Olmo-mix-1124 control, data seed 42"),
    "olmo-mix-1124-s69": ("mixlaw-new", None, "static-olmo-mix-1124-s69-farmshare-1745710",
                           "Olmo-mix-1124 control, data seed 69"),
    "data-mixing-laws-paper": ("mixlaw-new", None, "static-mix01-s42-farmshare-1745708",
                                "Data Mixing Laws paper mixture"),
    "mixlaw-fit": ("mixlaw-new", None, "static-ml-min1pct-s42-farmshare-1745706",
                    "MixLaw fit (ours), 1%-floor optimum"),
}
# Order fixes each arm's alpha-free bootstrap stream in fit_and_bootstrap_370m.py
# (SeedSequence(seed).spawn(len(runs))[index]). Keeping the three already-reported
# arms (probe, derivative, lightgbm-l40s) at the end preserves their existing
# fitted-final CIs bit-for-bit; only their comparisons against the new control
# change.
ORDER = [
    "olmo-mix-1124-s42", "olmo-mix-1124-s69", "data-mixing-laws-paper", "mixlaw-fit",
    "skillit-probe", "skillit-derivative", "lightgbm-l40s",
]
assert set(ORDER) == set(RUNS)


def _resolve_run(api, project: str, run_id: str | None, display_name: str | None):
    if run_id is not None:
        return api.run(f"eduLLM/{project}/{run_id}")
    matches = list(api.runs(f"eduLLM/{project}", filters={"display_name": display_name}))
    if len(matches) != 1:
        raise SystemExit(
            f"expected exactly one eduLLM/{project} run named {display_name!r}, "
            f"found {len(matches)}"
        )
    return matches[0]


def pull_arm(api, key: str) -> dict:
    project, run_id, display_name, label = RUNS[key]
    run = _resolve_run(api, project, run_id, display_name)
    by_step: dict[int, dict] = {}
    for row in run.scan_history(keys=HISTORY_KEYS):
        if row.get("eval/macro_bpb") is None:
            continue
        step = int(row["_step"])
        by_step[step] = row  # last write wins -- de-dupes resume-repeated rows

    missing = [s for s in EVAL_LADDER if s not in by_step]
    if missing:
        raise SystemExit(f"{key} ({run.id}): missing eval steps {missing}")

    steps, macro_bpb, per_label = [], [], {}
    for s in EVAL_LADDER:
        row = by_step[s]
        labels = {lbl: float(row[f"eval/bpb/{lbl}"]) for lbl in ALL_LABELS}
        macro = float(row["eval/macro_bpb"])
        mean20 = sum(labels.values()) / 20.0
        if abs(macro - mean20) > 1e-6:
            raise SystemExit(
                f"{key} ({run.id}) @ step {s}: eval/macro_bpb={macro} != "
                f"mean(20 labels)={mean20}"
            )
        steps.append(s)
        macro_bpb.append(macro)
        per_label[s] = labels

    print(f"  {key:<24} {run.entity}/{run.project}/{run.id:<10} "
          f"n={len(steps)} final={macro_bpb[-1]:.6f}")
    return {
        "label": label,
        "wandb_run": f"eduLLM/{project}/{run.id}",
        "steps": steps,
        "macro_bpb": macro_bpb,
        "_per_label": per_label,  # consumed below; not written to the curves file
    }


def build_curves_payload(arms: dict[str, dict]) -> dict:
    return {
        "metric": "eval/macro_bpb",
        "description": (
            "Macro mean bits-per-byte over the 20 OLMES-style ranked-classification "
            "labels, per eval step, for every OLMo2-370M / 10B-token validation arm "
            "reported in the paper (Tables II and III, Figures I-III). All 7 arms "
            "run the vendored Skill-It trainer on FarmShare 4xL40S: the five static "
            "arms in eduLLM/mixlaw-new, the two dynamic (Skill-It) arms in "
            "eduLLM/skillit."
        ),
        "final_step": 2384,
        "excluded_steps": [2375],
        "excluded_steps_note": (
            "Step 2375 is a near-duplicate checkpoint 9 steps before the true "
            "final step (2384). Keeping both double-weights the end of the run, "
            "so it is excluded from every power-law fit and from these series."
        ),
        "fit_window": {"min_step": 1000, "form": "y = a + b / step**alpha"},
        "runs": {
            key: {k: v for k, v in arms[key].items() if not k.startswith("_")}
            for key in ORDER
        },
    }


def build_heldout_payloads(arms: dict[str, dict]) -> tuple[dict, dict]:
    key_alias = {
        "olmo-mix-1124-s42": "olmo_s42",
        "olmo-mix-1124-s69": "olmo_s69",
        "data-mixing-laws-paper": "dml_paper",
        "mixlaw-fit": "mixlaw",
        "lightgbm-l40s": "lightgbm",
        "skillit-probe": "probe",
        "skillit-derivative": "derivative",
    }
    label_curves: dict[str, dict] = {}
    perlabel_curves: dict[str, dict] = {}
    for key, alias in key_alias.items():
        per_label = arms[key]["_per_label"]
        label_curves[alias] = {}
        perlabel_curves[alias] = {}
        for step, labels in per_label.items():
            targeted = sum(labels[l] for l in TARGETED_LABELS) / len(TARGETED_LABELS)
            held_out = sum(labels[l] for l in NEVER_TARGETED_LABELS) / len(NEVER_TARGETED_LABELS)
            macro20 = sum(labels.values()) / 20.0
            label_curves[alias][str(step)] = {
                "held_out": held_out, "targeted": targeted, "macro20": macro20,
            }
            perlabel_curves[alias][str(step)] = {l: labels[l] for l in NEVER_TARGETED_LABELS}
    return label_curves, perlabel_curves


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="pull and validate only")
    args = ap.parse_args()

    import wandb  # imported lazily so --help needs no wandb / network

    api = wandb.Api()
    print("Pulling 370M validation curves from W&B:")
    arms = {key: pull_arm(api, key) for key in ORDER}

    curves_payload = build_curves_payload(arms)
    label_payload, perlabel_payload = build_heldout_payloads(arms)

    if args.dry_run:
        print("\n--dry-run: validated, nothing written.")
        return

    CURVES_PATH.write_text(json.dumps(curves_payload, indent=1) + "\n", encoding="utf-8")
    HELDOUT_LABEL_PATH.write_text(json.dumps(label_payload, indent=1) + "\n", encoding="utf-8")
    HELDOUT_PERLABEL_PATH.write_text(json.dumps(perlabel_payload, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {CURVES_PATH.name}, {HELDOUT_LABEL_PATH.name}, {HELDOUT_PERLABEL_PATH.name}")


if __name__ == "__main__":
    main()
