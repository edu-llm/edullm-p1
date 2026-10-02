"""The in-run task-loss curve is scraped from the trainer log, so its parser has to
follow the log format of the installed ai2-olmo (0.6.0 splits the step and the eval
metrics across separate records)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mixlaw_common import parse_trainer_log_record  # noqa: E402

LABEL = "arc_challenge_val_rc_5shot_bpb"


def test_step_record_sets_step() -> None:
    step, loss, task = parse_trainer_log_record("[step=120/1451,epoch=0]\n    train/CrossEntropyLoss=3.25")
    assert (step, loss, task) == (120, 3.25, {})


def test_eval_record_without_step_takes_last_step() -> None:
    msg = f"{LABEL}\n    eval/downstream_bpb/{LABEL}_bpb=2.371"
    assert parse_trainer_log_record(msg, last_step=120) == (120, None, {LABEL: 2.371})
    assert parse_trainer_log_record(msg) == (None, None, {LABEL: 2.371})


def test_eval_progress_record_is_not_a_step() -> None:
    assert parse_trainer_log_record("[eval_step=3/16]", last_step=120) == (120, None, {})


def test_single_record_format_still_parses() -> None:
    msg = f"step=240, eval/downstream_bpb/{LABEL}_bpb=2.1"
    assert parse_trainer_log_record(msg) == (240, None, {LABEL: 2.1})
