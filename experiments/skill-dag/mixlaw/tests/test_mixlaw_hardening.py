"""Focused local test for the shared MixLaw W&B eval-logging contract.

The recovery/production-contract tests that used to live here covered
mixlaw_runtime.py and train_mixlaw_validation_370m.py, the curriculum-derived
370M trainer that no reported run used. Both files
were removed once every reported arm ran on
the Skill-It trainer. This test
covers mixlaw_wandb.py, which the 60M pilots and probes still use.
"""
from __future__ import annotations

import sys
from pathlib import Path

_MIXLAW = Path(__file__).resolve().parents[1]
_TS_ROOT = _MIXLAW.parents[1] / "token-selection"
for _path in (_MIXLAW, _TS_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import mixlaw_wandb as mix_wandb  # noqa: E402
from token_selection.olmo_ext.wandb_logging import TASK_LOSS_RAW_LABELS  # noqa: E402


def test_mixlaw_wandb_macro_uses_only_complete_raw_suite(monkeypatch) -> None:
    class Run:
        def __init__(self) -> None:
            self.logged: list[tuple[int, dict[str, float]]] = []

        def log(self, metrics, step=None):
            self.logged.append((step, dict(metrics)))

    run = Run()
    partial = {"macro_mean": 0.1, "labels": {TASK_LOSS_RAW_LABELS[0]: 2.0}}
    mix_wandb.wandb_log_eval(run, partial, step=1)
    assert "eval/macro_bpb" not in run.logged[-1][1]

    labels = {label: float(index + 1) for index, label in enumerate(TASK_LOSS_RAW_LABELS)}
    mix_wandb.wandb_log_eval(
        run,
        {"macro_mean": -999.0, "labels": labels},
        step=2,
    )
    assert run.logged[-1][1]["eval/macro_bpb"] == sum(labels.values()) / 20
