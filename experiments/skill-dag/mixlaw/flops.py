#!/usr/bin/env python3
"""Analytic training FLOPs for the 60M pilots/probes and the 370M validation arms.

Uses the 6ND estimate of Kaplan et al. (2020) plus the attention term of
Chowdhery et al. (2023, PaLM):

    C = 6 * N_nonemb * D + 12 * n_layers * s * d_model * D

with ``N_nonemb`` the trained model's non-embedding parameter count (OLMo's
"model size": every parameter except ``wte``, so the untied LM head counts),
``s`` the sequence length and ``D`` the tokens actually trained on (steps x
tokens per step). Evaluation passes are not counted.

Prints per-run and aggregate FLOPs for the numbers the READMEs quote:
1.74e17 per 60M run, 4.17e18 for the 24 pilots (~16% of one 370M arm),
1.22e18 for the 7 Skill-It one-hot probes, and 2.63e19 per 370M arm.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

from mixlaw_common import D_MODEL, N_LAYERS, SEQ_LEN, TOKENS_PER_STEP


@dataclass(frozen=True)
class Architecture:
    name: str
    n_nonemb: int
    n_layers: int
    seq_len: int
    d_model: int
    steps: int
    tokens_per_step: int

    @property
    def tokens(self) -> int:
        return self.steps * self.tokens_per_step


# DataDecide-60M body with the dolma2 embedding (the 24 pilots and the 7 Skill-It
# probes): 76,296,576 non-embedding parameters, 1451 steps of 96 x 2048 tokens
# (tokens/param = 5 against DataDecide's 57,078,144 -> 285,278,208 tokens).
PROBE = Architecture(
    name="60M pilot / probe",
    n_nonemb=76_296_576,
    n_layers=N_LAYERS,
    seq_len=SEQ_LEN,
    d_model=D_MODEL,
    steps=1451,
    tokens_per_step=TOKENS_PER_STEP,
)

# OLMo-2 370M validation arm: 371,262,464 non-embedding parameters, 2384 steps
# of 4,194,304 tokens (2048 sequences x 2048), about one 10B-token epoch.
VALIDATION = Architecture(
    name="370M validation arm",
    n_nonemb=371_262_464,
    n_layers=16,
    seq_len=2048,
    d_model=1024,
    steps=2384,
    tokens_per_step=4_194_304,
)

N_PILOTS = 24
N_SKILLIT_PROBES = 7


def training_flops(arch: Architecture, tokens: int | None = None) -> float:
    """``6 N D + 12 n_layers s d_model D`` for ``tokens`` (default: the trained count)."""
    d = arch.tokens if tokens is None else tokens
    return float(6 * arch.n_nonemb * d + 12 * arch.n_layers * arch.seq_len * arch.d_model * d)


def report() -> dict:
    per_pilot = training_flops(PROBE)
    per_arm = training_flops(VALIDATION)
    pilots = N_PILOTS * per_pilot
    return {
        "formula": "6*N_nonemb*D + 12*n_layers*s*d_model*D",
        "architectures": {
            "probe": {**asdict(PROBE), "tokens": PROBE.tokens},
            "validation": {**asdict(VALIDATION), "tokens": VALIDATION.tokens},
        },
        "per_60m_run": per_pilot,
        "pilot_grid_24": pilots,
        "skillit_probes_7": N_SKILLIT_PROBES * per_pilot,
        "per_370m_arm": per_arm,
        "pilot_grid_over_one_370m_arm": pilots / per_arm,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args()
    r = report()
    if args.json:
        print(json.dumps(r, indent=2))
        return
    for key in ("probe", "validation"):
        a = r["architectures"][key]
        print(
            f"{a['name']:20s} N_nonemb={a['n_nonemb']:,} n_layers={a['n_layers']} "
            f"s={a['seq_len']} d_model={a['d_model']} D={a['tokens']:,} "
            f"({a['steps']} steps x {a['tokens_per_step']:,})"
        )
    print(f"per 60M run          {r['per_60m_run']:.4e}")
    print(f"24 pilots            {r['pilot_grid_24']:.4e}")
    print(f"7 Skill-It probes    {r['skillit_probes_7']:.4e}")
    print(f"per 370M arm         {r['per_370m_arm']:.4e}")
    print(f"24 pilots / 370M arm {r['pilot_grid_over_one_370m_arm']:.1%}")


if __name__ == "__main__":
    main()
