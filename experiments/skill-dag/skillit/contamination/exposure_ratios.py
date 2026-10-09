#!/usr/bin/env python3
"""Print each Skill-It arm's time-weighted exposure relative to its references.

Reads the committed outputs only; recomputes nothing upstream:

  - `exposure_<arm>.json` (from `trajectory_exposure.py`) for each arm's
    time-weighted average span and document rates, and its step-0 segment;
  - `../../mixlaw/contamination/exposure_by_arm.json` (from
    `mixture_exposure.py`) for the `olmo-mix-1124` baseline and the
    `LGB-min1pct` mixture both arms start from.

For each arm it prints the ratio of the time-weighted average to the
baseline and to the arm's own start, for both rates. These are the
"0.78x baseline; 0.99x its own start"-style figures in `README.md`.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

log = logging.getLogger("exposure_ratios")

HERE = Path(__file__).resolve().parent
RATE_KEYS = ("matched_span_word_rate", "matched_document_rate")
ARMS = ("offline-probe", "online-derivative")
BASELINE = "olmo-mix-1124"
START = "LGB-min1pct"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--static-arms",
        default=str(HERE.parent.parent / "mixlaw" / "contamination" / "exposure_by_arm.json"),
        help="exposure_by_arm.json from mixture_exposure.py",
    )
    parser.add_argument("--dir", default=str(HERE), help="directory holding exposure_<arm>.json")
    parser.add_argument("--json", action="store_true", help="print the ratios as JSON")
    args = parser.parse_args()

    static = json.loads(Path(args.static_arms).read_text(encoding="utf-8"))["arms"]
    for name in (BASELINE, START):
        if name not in static:
            raise RuntimeError(f"{args.static_arms}: no arm {name!r}")

    report: dict[str, dict] = {}
    for arm in ARMS:
        path = Path(args.dir) / f"exposure_{arm}.json"
        traj = json.loads(path.read_text(encoding="utf-8"))
        step0 = traj["segments"][0]
        if step0["step"] != 0:
            raise RuntimeError(f"{path}: first segment is not step 0")
        entry: dict[str, dict] = {}
        for key in RATE_KEYS:
            # Both arms start from LGB-min1pct; the step-0 segment must carry its exposure.
            if not math.isclose(step0[key], static[START][key], rel_tol=1e-12):
                raise RuntimeError(
                    f"{path}: step-0 {key} {step0[key]!r} is not {START}'s {static[START][key]!r}"
                )
            avg = traj[f"time_weighted_avg_{key}"]
            entry[key] = {
                "time_weighted_avg": avg,
                "vs_baseline": avg / static[BASELINE][key],
                "vs_own_start": avg / step0[key],
            }
        report[arm] = entry

    if args.json:
        print(json.dumps({"baseline": BASELINE, "start": START, "arms": report}, indent=1))
        return 0

    log.info("baseline %s, start %s", BASELINE, START)
    log.info("%-18s %-24s %12s %8s %8s", "arm", "rate", "avg", "vs base", "vs start")
    for arm, entry in report.items():
        for key in RATE_KEYS:
            e = entry[key]
            log.info(
                "%-18s %-24s %12.4e %7.2fx %7.2fx",
                arm,
                key,
                e["time_weighted_avg"],
                e["vs_baseline"],
                e["vs_own_start"],
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
