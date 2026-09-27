"""Immutable scientific identities for the approved 370M arms.

Every arm below runs through the same entrypoint, the same
``TokenWeightedTrainModule``, and the same FarmShare 4xL40S hardware contract
(``PRODUCTION_WORLD_SIZE`` in ``recipe.py``). There is no RunPod path and no
AWS/S3 code anywhere in this tree: corpora are read from local FarmShare
paths staged by ``farmshare/stage_local.py``, bound by the per-file sha256
recorded in this repository's ``datasets/manifests/<corpus>/outputs.json``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

Method = Literal[
    "rho_excess",
    "rel_ema",
    "middle_ppl",
    "attention_topk",
    "blade",
    "random",
    "full",
]

REGMIX = "pretrain/regmix-10b"
# BLADE's L_val stream, and the Instruct reference arm's own training corpus.
REFHQ_INSTRUCT = "pretrain/refhq-instruct"
# The HQ reference arm's own training corpus.
REFHQ_5P5B = "pretrain/refhq-regmix-5p5b"

# Symbolic reference contracts, resolved to a local, materialized .pt file by
# farmshare/stage_local.py once the reference arm below has produced the
# checkpoint. Nothing here is an S3 URI.
INSTRUCT_REFERENCE_CONTRACT = "instruct-reference-370m/checkpoints/step940"
HQ_REFERENCE_CONTRACT = "hq-reference-370m/checkpoints/average-1000-1125-1315"
HQ_REFERENCE_AVERAGED_STEPS = (1000, 1125, 1315)


@dataclass(frozen=True)
class ArmSpec:
    name: str
    method: Method
    dataset_id: str
    run_id: str
    keep_fraction: float = 1.0
    max_tokens: Optional[int] = 9_900_000_000
    wandb_project_override: Optional[str] = None
    reference_contract: Optional[str] = None
    ema_seed: Optional[Literal["zero"]] = None
    ema_tau: Optional[float] = None
    requires_refhq_stream: bool = False
    init_seed: int = 6198
    data_seed: int = 42
    rank_microbatch_tokens: int = 16_384

    @property
    def wandb_project(self) -> str:
        return self.wandb_project_override or "token-selection"


ARM_SPECS: dict[str, ArmSpec] = {
    # ---- References (trained in this study; no other trainer produces them) ----
    # Same "full" method / stock-equivalent loss as the full-loss control, just
    # on a different corpus and for a different step count.
    "hq-reference": ArmSpec(
        "hq-reference",
        "full",
        REFHQ_5P5B,
        "hq-reference-370m",
        keep_fraction=1.0,
        max_tokens=None,  # one whole-stream epoch of refhq-regmix-5p5b; see recipe.py
    ),
    "instruct-reference": ArmSpec(
        "instruct-reference",
        "full",
        REFHQ_INSTRUCT,
        "instruct-reference-370m",
        keep_fraction=1.0,
        max_tokens=None,  # one whole-stream epoch of refhq-instruct's train split
    ),
    # ---- Reported arms: full-loss control, five selection arms, two random-control seeds ----
    "full-loss-control": ArmSpec(
        "full-loss-control",
        "full",
        REGMIX,
        "full-loss-control-regmix10b",
        keep_fraction=1.0,
    ),
    "rho-1": ArmSpec(
        "rho-1",
        "rho_excess",
        REGMIX,
        "rho-1-regmix10b",
        keep_fraction=0.6,
        reference_contract=INSTRUCT_REFERENCE_CONTRACT,
    ),
    "rel-ema-exp": ArmSpec(
        "rel-ema-exp",
        "rel_ema",
        REGMIX,
        "rel-ema-exp-10b-scratch-v1",
        keep_fraction=0.6,
        ema_seed="zero",
        ema_tau=300.0,
    ),
    "perplexity": ArmSpec(
        "perplexity",
        "middle_ppl",
        REGMIX,
        "perplexity-regmix10b",
        keep_fraction=0.6,
        reference_contract=HQ_REFERENCE_CONTRACT,
    ),
    "attention": ArmSpec(
        "attention",
        "attention_topk",
        REGMIX,
        "attention-regmix10b",
        keep_fraction=0.6,
    ),
    "blade": ArmSpec(
        "blade",
        "blade",
        REGMIX,
        "blade-regmix10b",
        keep_fraction=0.6,
        requires_refhq_stream=True,
    ),
    "random-control": ArmSpec(
        "random-control",
        "random",
        REGMIX,
        "random-control-regmix10b-v1",
        keep_fraction=0.6,
        init_seed=6198,
        data_seed=42,
    ),
    "random-control-seed69": ArmSpec(
        "random-control-seed69",
        "random",
        REGMIX,
        "random-control-regmix10b-seed69-v1",
        keep_fraction=0.6,
        init_seed=12345,
        data_seed=69,
    ),
}


def get_arm(name: str) -> ArmSpec:
    try:
        return ARM_SPECS[name]
    except KeyError:
        raise ValueError(
            f"unknown token-selection arm {name!r}; expected one of {sorted(ARM_SPECS)}"
        ) from None
