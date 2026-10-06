"""The step-law input is the run's in-run curve (domain_probes), final point included."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_MIXLAW = Path(__file__).resolve().parents[1]
if str(_MIXLAW) not in sys.path:
    sys.path.insert(0, str(_MIXLAW))

from extrapolate_chinchilla import _curve_points  # noqa: E402
from mixlaw_common import PROBE_STEPS, PROBE_TASK_LOSS_LABELS, load_probe_run  # noqa: E402

ARC = "arc_challenge_test_rc_5shot_bpb"


def test_curve_points_follow_the_in_run_curve() -> None:
    run = {
        "curve": [
            {"step": 120, "task_loss_bpb": {ARC: 2.0}},
            {"step": PROBE_STEPS, "task_loss_bpb": {ARC: 1.5}},
        ],
    }
    per_family = _curve_points(run)
    assert per_family["arc_challenge"] == [(120, 2.0), (PROBE_STEPS, 1.5)]


def _write_run(path: Path, last_step: int, drop: str | None = None) -> None:
    path.mkdir(parents=True)
    rows = []
    for step in range(120, last_step + 1, 120):
        labels = {lb: 3.0 - step / 1000 for lb in PROBE_TASK_LOSS_LABELS if lb != drop}
        rows.append({"step": step, "task_loss_bpb": labels})
    (path / "task_loss.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_load_probe_run_takes_the_last_eval_as_final(tmp_path: Path) -> None:
    _write_run(tmp_path / "mix01", PROBE_STEPS)
    run = load_probe_run(tmp_path / "mix01")
    assert len(run["curve"]) == PROBE_STEPS // 120 == 12
    assert set(run["task_loss_labels"]) == set(PROBE_TASK_LOSS_LABELS)
    assert run["task_loss_families"]["arc_challenge"] == pytest.approx(3.0 - PROBE_STEPS / 1000)


def test_load_probe_run_rejects_unfinished_or_incomplete_runs(tmp_path: Path) -> None:
    _write_run(tmp_path / "short", PROBE_STEPS - 120)
    with pytest.raises(SystemExit, match="expected"):
        load_probe_run(tmp_path / "short")
    _write_run(tmp_path / "gap", PROBE_STEPS, drop=ARC)
    with pytest.raises(SystemExit, match="missing labels"):
        load_probe_run(tmp_path / "gap")
