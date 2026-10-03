#!/usr/bin/env python3
"""Turn the probe runs' trainer logs into the per-probe files ``build_adjacency.py`` reads.

The trainer's curve logger only matched an older ai2-olmo log format, so the 60M probes
trained with ai2-olmo 0.6.0 left no ``task_loss.jsonl``. The curve is still in each
probe's Slurm log, one ``eval/downstream_bpb/<label>_bpb_bpb=<value>`` line per label
after each ``[step=N/T,...]`` record, so it is rebuilt from there.

Input (stdin or ``--pull``) is the text printed by the cluster-side puller, per probe::

    ### <probe_id>
    CURVE {"step": 120, "task_loss_bpb": {"<label>": <bpb>, ...}}     (one per eval)
    FINAL {<contents of task_loss_final.json>}
    META  {<contents of run_meta.json>}                               (optional)

Output: ``<out>/<probe_id>/progress/{task_loss.jsonl,task_loss_final.json,run_meta.json}``.
Fails closed: every probe must have the full in-run curve (steps 120..1440, all labels)
and a final evaluation, otherwise nothing is written for any probe.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SKILLIT = Path(__file__).resolve().parent
_MIXLAW = _SKILLIT.parent / "mixlaw"
if str(_MIXLAW) not in sys.path:
    sys.path.insert(0, str(_MIXLAW))

from mixlaw_common import CURVE_FAMILIES, CURVE_TASK_LOSS_LABELS  # noqa: E402

EXPECTED_STEPS = tuple(range(120, 1441, 120))


def parse_pull(text: str) -> dict[str, dict]:
    probes: dict[str, dict] = {}
    current: dict | None = None
    for line in text.splitlines():
        line = line.rstrip("\r")
        if line.startswith("### "):
            current = probes.setdefault(line[4:].strip(), {"curve": []})
        elif current is None:
            continue
        elif line.startswith("CURVE "):
            current["curve"].append(json.loads(line[6:]))
        elif line.startswith("FINAL ") and line[6:].strip():
            current["final"] = json.loads(line[6:])
        elif line.startswith("META ") and line[5:].strip():
            current["meta"] = json.loads(line[5:])
    return probes


def check_probe(name: str, probe: dict) -> list[str]:
    problems: list[str] = []
    steps = tuple(row["step"] for row in probe["curve"])
    if steps != EXPECTED_STEPS:
        problems.append(f"curve steps {steps} != {EXPECTED_STEPS}")
    for row in probe["curve"]:
        missing = set(CURVE_TASK_LOSS_LABELS) - set(row["task_loss_bpb"])
        if missing:
            problems.append(f"step {row['step']} lacks labels {sorted(missing)}")
    final = probe.get("final")
    if not final:
        problems.append("no task_loss_final.json")
    else:
        if set(final.get("task_families") or {}) != set(CURVE_FAMILIES):
            problems.append(f"final families {sorted(final.get('task_families') or {})}")
        if final.get("smoke"):
            problems.append("final evaluation is a smoke run")
    return [f"{name}: {p}" for p in problems]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pull", type=Path, default=None, help="pull output file (default: stdin)")
    ap.add_argument("--out", type=Path, required=True, help="runs directory to write")
    ap.add_argument(
        "--expect",
        nargs="+",
        default=None,
        help="probe ids that must all be present (default: whatever the input contains)",
    )
    args = ap.parse_args()

    text = args.pull.read_text(encoding="utf-8") if args.pull else sys.stdin.read()
    probes = parse_pull(text)
    if not probes:
        raise SystemExit("no probes in the input")
    if args.expect is not None:
        absent = sorted(set(args.expect) - set(probes))
        if absent:
            raise SystemExit(f"missing probes: {absent}")
    problems = [p for name, probe in probes.items() for p in check_probe(name, probe)]
    if problems:
        raise SystemExit("refusing to write:\n  " + "\n  ".join(problems))

    for name, probe in sorted(probes.items()):
        progress = args.out / name / "progress"
        progress.mkdir(parents=True, exist_ok=True)
        (progress / "task_loss.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in probe["curve"]), encoding="utf-8"
        )
        (progress / "task_loss_final.json").write_text(
            json.dumps(probe["final"], indent=2) + "\n", encoding="utf-8"
        )
        if "meta" in probe:
            (progress / "run_meta.json").write_text(
                json.dumps(probe["meta"], indent=2) + "\n", encoding="utf-8"
            )
        print(f"{name}: {len(probe['curve'])} curve points + final -> {progress}")


if __name__ == "__main__":
    main()
