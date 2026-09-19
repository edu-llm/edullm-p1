#!/usr/bin/env python3
"""Token-weighted average contaminated exposure for a time-varying mixture.

`../../mixlaw/contamination/mixture_exposure.py` computes exposure for a FIXED
mixture: exposure = sum_d weight(d) * rate(d). Skill-It reweights domains
mid-training, so a single static weight vector does not describe a whole
run -- the sampler holds each mixture fixed only between update steps, then
jumps to a new one. This script computes the same per-segment exposure and
then a token-weighted average across segments, which is the exposure a
run with this trajectory actually trained under in aggregate.

Input is `skillit_updates_<arm>.jsonl`, the arm's own update log written by
Skill-It's training loop (schema shared with `plot_weights.py`): one JSON
object per update step, key fields `step` and `p_after` (the realized
mixture -- possibly unnormalized floating-point weights -- that the sampler
uses from this step until the next logged step, or until the run ends for
the last one).

Segment lengths come from consecutive `step` values, with the final segment
running to `--final-step` (this run's total step budget -- not logged in the
update file, since the file only records update EVENTS). Token weighting
reduces to step-count weighting because every step draws the same number of
tokens.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

log = logging.getLogger("trajectory_exposure")

RATE_KEYS = ("matched_span_word_rate", "matched_document_rate")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-domain", required=True, help="../../mixlaw/contamination/results_olmo127b-reservoir.json")
    parser.add_argument("--updates", required=True, help="skillit_updates_<arm>.jsonl")
    parser.add_argument("--arm", required=True, help="label for the report, e.g. skillit-deriv")
    parser.add_argument("--final-step", type=int, required=True, help="this run's total step budget")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    per_domain = json.loads(Path(args.per_domain).read_text(encoding="utf-8"))["domains"]
    rates = {key: {d: float(per_domain[d][key]) for d in per_domain} for key in RATE_KEYS}

    rows = [json.loads(line) for line in Path(args.updates).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"{args.updates}: no update rows")
    steps = [int(r["step"]) for r in rows]
    if steps != sorted(steps) or len(set(steps)) != len(steps):
        raise RuntimeError(f"{args.updates}: step column is not strictly increasing: {steps}")
    if steps[0] != 0:
        raise RuntimeError(f"{args.updates}: first row must be step 0 (the run's starting mixture)")
    segment_ends = steps[1:] + [args.final_step]
    if segment_ends[-1] <= steps[-1]:
        raise RuntimeError(
            f"--final-step {args.final_step} is not after the last logged update "
            f"step {steps[-1]}; the last segment would have non-positive length"
        )
    segment_lengths = [e - s for s, e in zip(steps, segment_ends)]

    segments = []
    for row, length in zip(rows, segment_lengths):
        weights = row["p_after"]
        missing = [d for d in weights if d not in per_domain]
        if missing:
            raise RuntimeError(f"step {row['step']}: no measured rate for {missing}")
        total = sum(weights.values())
        normalized = {d: w / total for d, w in weights.items()}
        exposure = {key: sum(w * rates[key][d] for d, w in normalized.items()) for key in RATE_KEYS}
        segments.append(
            {
                "step": row["step"],
                "length_steps": length,
                "weights": normalized,
                **exposure,
            }
        )

    avg = {
        key: sum(seg[key] * seg["length_steps"] for seg in segments) / args.final_step
        for key in RATE_KEYS
    }

    log.info("%s (final_step=%d)", args.arm, args.final_step)
    log.info("%6s %6s %14s %14s", "step", "len", "span_rate", "doc_rate")
    for seg in segments:
        log.info(
            "%6d %6d %14.4e %14.4e",
            seg["step"],
            seg["length_steps"],
            seg["matched_span_word_rate"],
            seg["matched_document_rate"],
        )
    log.info("")
    log.info("time-weighted avg span rate: %.4e", avg["matched_span_word_rate"])
    log.info("time-weighted avg doc  rate: %.4e", avg["matched_document_rate"])

    report = {
        "arm": args.arm,
        "final_step": args.final_step,
        "segments": segments,
        "time_weighted_avg_matched_span_word_rate": avg["matched_span_word_rate"],
        "time_weighted_avg_matched_document_rate": avg["matched_document_rate"],
    }
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    log.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
