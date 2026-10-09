"""Check every YAML under experiments/token-selection/arms/ against the vendored
token_selection_370m.arms.ARM_SPECS, the actual source of truth every run reads."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

TOKEN_SELECTION_ROOT = Path(__file__).resolve().parents[2]
ARMS_DIR = TOKEN_SELECTION_ROOT / "arms"
OLMO_CORE_TOKEN_SELECTION = TOKEN_SELECTION_ROOT / "olmo_core_token_selection"
if str(OLMO_CORE_TOKEN_SELECTION) not in sys.path:
    sys.path.insert(0, str(OLMO_CORE_TOKEN_SELECTION))

from token_selection_370m.arms import ARM_SPECS  # noqa: E402

# instruct-reference is an ArmSpec too (arms.py's own "References" section) but isn't
# in the reported-arm table under the same run_id-derived name;
# map YAML stem -> ARM_SPECS key explicitly so a rename on either side is caught.
YAML_TO_SPEC_KEY = {
    "instruct-reference": "instruct-reference",
    "full-loss-control": "full-loss-control",
    "random-control": "random-control",
    "random-control-seed69": "random-control-seed69",
    "rho-1": "rho-1",
    "rel-ema-exp": "rel-ema-exp",
    "perplexity": "perplexity",
    "attention": "attention",
    "blade": "blade",
}


def _load_yaml(stem: str) -> dict:
    path = ARMS_DIR / f"{stem}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_every_arm_spec_has_a_yaml() -> None:
    assert set(YAML_TO_SPEC_KEY) == set(ARM_SPECS)
    for stem in YAML_TO_SPEC_KEY:
        assert (ARMS_DIR / f"{stem}.yaml").is_file(), f"missing arms/{stem}.yaml"


@pytest.mark.parametrize("stem,spec_key", sorted(YAML_TO_SPEC_KEY.items()))
def test_yaml_matches_arm_spec(stem: str, spec_key: str) -> None:
    doc = _load_yaml(stem)
    spec = ARM_SPECS[spec_key]

    assert doc["name"] == spec.name
    assert doc["method"] == spec.method
    assert doc["dataset_id"] == spec.dataset_id
    assert doc["run_id"] == spec.run_id
    assert doc["keep_fraction"] == pytest.approx(spec.keep_fraction)
    assert doc["init_seed"] == spec.init_seed
    assert doc["data_seed"] == spec.data_seed
    assert doc["rank_microbatch_tokens"] == spec.rank_microbatch_tokens
    assert doc["wandb_project"] == spec.wandb_project

    yaml_max_tokens = doc.get("max_tokens")
    if spec.max_tokens is None:
        assert yaml_max_tokens is None
    else:
        assert yaml_max_tokens == spec.max_tokens

    if spec.requires_refhq_stream:
        assert doc.get("requires_refhq_stream") is True

    if spec.reference_contract is not None:
        # The YAML records the contract under a nested "reference" or
        # "frozen_reference" block; the exact contract string must appear somewhere.
        assert spec.reference_contract in yaml.dump(doc)


def test_checkpoint_ladders_start_at_zero_and_end_at_total_steps() -> None:
    for stem in YAML_TO_SPEC_KEY:
        doc = _load_yaml(stem)
        ladder = doc.get("checkpoint_ladder")
        if ladder is None:
            continue
        assert ladder[0] == 0
        assert ladder[-1] == doc["total_steps"]
        assert ladder == sorted(ladder)
