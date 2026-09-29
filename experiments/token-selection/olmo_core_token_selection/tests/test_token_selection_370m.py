from __future__ import annotations

import ast
import contextlib
import hashlib
import json
import math
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch import nn

EDULLM_ROOT = Path(__file__).resolve().parents[1]
if str(EDULLM_ROOT) not in sys.path:
    sys.path.insert(0, str(EDULLM_ROOT))

import token_selection_370m.selection as selection_module  # noqa: E402
from token_selection_370m import reference_scores as reference_scores_module  # noqa: E402
from token_selection_370m.arms import (  # noqa: E402
    ARM_SPECS,
    INSTRUCT_REFERENCE_CONTRACT,
    REFERENCE_SCORES_CONTRACT,
    REFHQ_INSTRUCT,
)
from token_selection_370m.blade import (  # noqa: E402
    BLADE_CHECKPOINT_FORMAT,
    BLADE_REFERENCE_MICROBATCH_TOKENS,
    BLADE_SELECTION_MICROBATCH_TOKENS,
    BLADE_SYNC_STEPS,
    BladeCallback,
    BladeSchedule,
    ResumableBatchStream,
    _full_proxy_state,
)
from token_selection_370m.recipe import (  # noqa: E402
    ALPHA_F,
    GLOBAL_BATCH_TOKENS,
    PEAK_LR,
    PRODUCTION_WORLD_SIZE,
    SEQUENCE_LENGTH,
    WARMUP_STEPS,
    Z_LOSS,
    immutable_corpus_binding,
    reference_digest,
    scientific_identity,
    total_steps,
    write_identity,
)
from token_selection_370m.reference_scores import ReferenceScoreTable  # noqa: E402
from token_selection_370m.selection import (  # noqa: E402
    AttentionPositionBaseline,
    EMAHistory,
    align_log_attention_mass,
    attention_received_from_qk,
    capture_last_attention,
    ema_alpha,
    per_instance_tiebreak,
    per_row_middle,
    per_row_topk,
    selection_weights,
    uniform_attention_normalizer,
    uniform_prior_log_mass,
)
from token_selection_370m.train_module import (  # noqa: E402
    TokenSelectionConfig,
    TokenSelectionState,
    TokenWeightedTrainModule,
)


def test_exact_approved_arm_family_and_wandb_routing() -> None:
    assert tuple(
        (name, spec.method, spec.dataset_id, spec.keep_fraction) for name, spec in ARM_SPECS.items()
    ) == (
        ("instruct-reference", "full", "pretrain/refhq-instruct", 1.0),
        ("full-loss-control", "full", "pretrain/regmix-10b", 1.0),
        ("rho-1", "rho_excess", "pretrain/regmix-10b", 0.6),
        ("rel-ema-exp", "rel_ema", "pretrain/regmix-10b", 0.6),
        ("perplexity", "middle_ppl", "pretrain/regmix-10b", 0.6),
        ("attention", "attention_topk", "pretrain/regmix-10b", 0.6),
        ("blade", "blade", "pretrain/regmix-10b", 0.6),
        ("random-control", "random", "pretrain/regmix-10b", 0.6),
        ("random-control-seed69", "random", "pretrain/regmix-10b", 0.6),
    )
    assert all(spec.wandb_project == "token-selection" for spec in ARM_SPECS.values())
    assert ARM_SPECS["rho-1"].reference_contract == INSTRUCT_REFERENCE_CONTRACT
    assert ARM_SPECS["perplexity"].reference_contract == INSTRUCT_REFERENCE_CONTRACT
    # Both read the offline per-instance table (reference_scores.py), not a
    # resident model.
    assert ARM_SPECS["rho-1"].reference_scores_contract == REFERENCE_SCORES_CONTRACT
    assert ARM_SPECS["perplexity"].reference_scores_contract == REFERENCE_SCORES_CONTRACT
    assert all(
        spec.reference_scores_contract is None
        for name, spec in ARM_SPECS.items()
        if name not in ("rho-1", "perplexity")
    )
    assert ARM_SPECS["blade"].requires_refhq_stream is True
    # Every arm runs 4xL40S; only the two random-control seeds differ in seed.
    assert ARM_SPECS["random-control"].init_seed == 6198
    assert ARM_SPECS["random-control"].data_seed == 42
    assert ARM_SPECS["random-control-seed69"].init_seed == 12345
    assert ARM_SPECS["random-control-seed69"].data_seed == 69
    assert all(
        spec.init_seed == 6198 and spec.data_seed == 42
        for name, spec in ARM_SPECS.items()
        if name not in ("random-control-seed69",)
    )


def test_one_recipe_constants_and_2360_step_budget() -> None:
    assert SEQUENCE_LENGTH == 2048
    assert GLOBAL_BATCH_TOKENS == 4_194_304
    assert PEAK_LR == 4e-4
    assert WARMUP_STEPS == 24
    assert ALPHA_F == 0.1
    assert Z_LOSS == 1e-5
    assert PRODUCTION_WORLD_SIZE == 4
    assert total_steps(9_900_000_000) == 2360
    # The reference arm uses its whole corpus as the budget (Corpus.rows, via
    # arm.max_tokens=None), not the fixed 9.9e9 the other arms share.
    assert ARM_SPECS["instruct-reference"].max_tokens is None
    assert total_steps(3_942_810_012) == 940
    # total_steps floors (2360, not 2361); build_trainer's max_duration must
    # use Duration.steps(total_steps(...)), not Duration.tokens(max_tokens)
    # (which the trainer runs as ceil(tokens/batch) -- one step past every
    # ladder/eval/BLADE-schedule/FLOP computation that uses this floor).
    recipe_source = (EDULLM_ROOT / "token_selection_370m" / "recipe.py").read_text(
        encoding="utf-8"
    )
    assert "max_duration=Duration.steps(steps)" in recipe_source
    assert "max_duration=Duration.tokens" not in recipe_source
    assert "main_loader.total_batches" in recipe_source  # the no-wrap guard


def test_custom_module_backpropagates_differentiable_total_loss() -> None:
    source = (EDULLM_ROOT / "token_selection_370m" / "train_module.py").read_text(encoding="utf-8")
    assert 'token_loss = self._loss_tensor(output, micro_labels, "loss")' in source
    assert "loss = (token_loss.float() * weights).sum() / divisor" in source


def test_weight_swaps_reshard_fsdp_before_restoring_parameters() -> None:
    # WeightShadow (the online reference-scoring path) is gone: rho_excess
    # and middle_ppl now read an offline table (reference_scores.py) instead
    # of keeping a resident reference model. Only EMAHistory.swap_to still
    # needs to reshard.
    source = (EDULLM_ROOT / "token_selection_370m" / "selection.py").read_text(encoding="utf-8")
    assert "class WeightShadow" not in source
    assert source.count("_reshard(model)") == 1
    assert "owner.unshard()" in source
    assert "owner.reshard()" in source


def test_weight_accounting_synchronizes_once_per_batch() -> None:
    source = (EDULLM_ROOT / "token_selection_370m" / "train_module.py").read_text(encoding="utf-8")
    assert "observed_weight += weights.sum()" in source
    assert source.count("observed_weight.item()") == 1


def test_all_token_ce_feeds_skip_step_optimizer_not_the_kept_subset() -> None:
    """E3: a discontinuity in the *kept* set (a BLADE resync, REL-EMA's
    growing alpha) must not by itself look like a loss spike to
    SkipStepOptimizer, so it watches all-token CE, not the selection's own
    (much smaller, policy-dependent) kept-token CE."""
    source = (EDULLM_ROOT / "token_selection_370m" / "train_module.py").read_text(encoding="utf-8")
    assert 'self.record_metric(\n            "CE loss (all tokens)", all_token_ce_batch' in source
    assert "self.optim.latest_loss = all_token_ce_batch" in source
    assert "self.optim.latest_loss = ce_batch" not in source


def test_ema_history_restores_local_fsdp_parameter_after_forward(tmp_path: Path) -> None:
    if not dist.is_available():
        pytest.skip("torch.distributed is unavailable")
    if dist.is_initialized():
        pytest.skip("test requires ownership of the local process group")

    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    dist.init_process_group(
        "gloo",
        init_method=(tmp_path / "fsdp-store").as_uri(),
        rank=0,
        world_size=1,
    )
    try:
        training = nn.Linear(2, 2, bias=False)
        training.weight.data.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        reference_state = {"weight": torch.tensor([[5.0, 6.0], [7.0, 8.0]])}
        baseline = nn.Linear(2, 2, bias=False)
        baseline.load_state_dict(reference_state)
        inputs = torch.tensor([[2.0, -1.0]])
        expected = baseline(inputs)

        fully_shard(training, mesh=init_device_mesh("cpu", (1,)))
        original_shard = training.weight.to_local().detach().clone()

        ema = EMAHistory(training, seed=reference_state)
        with torch.no_grad(), ema.swap_to(training):
            actual_ema = training(inputs)
        assert torch.equal(actual_ema, expected)
        assert torch.equal(training.weight.to_local(), original_shard)
    finally:
        dist.destroy_process_group()


def test_reference_score_table_gathers_by_global_instance_index(tmp_path: Path) -> None:
    """The A1 consumer side: build a tiny table, then read it back by index."""
    rows, sequence_length = 5, 4
    table = torch.arange(rows * sequence_length, dtype=torch.float32).reshape(rows, sequence_length)
    root = tmp_path / "table"
    root.mkdir()
    import numpy as np

    array = np.memmap(
        reference_scores_module.table_file(root), mode="w+", dtype=np.float32, shape=(rows, sequence_length)
    )
    array[:] = table.numpy()
    array.flush()
    corpus_binding = {"dataset_id": "pretrain/regmix-10b", "version": "v1"}
    reference_scores_module.write_manifest(
        root,
        {
            "schema_version": reference_scores_module.MANIFEST_SCHEMA_VERSION,
            "algorithm": reference_scores_module.ALGORITHM,
            "complete": True,
            "reference_sha256": "deadbeef",
            "corpus_binding": corpus_binding,
            "sequence_length": sequence_length,
            "rows": rows,
            "dtype": "float32",
        },
    )

    loaded = ReferenceScoreTable(
        root, expected_reference_sha256="deadbeef", expected_corpus_binding=corpus_binding
    )
    gathered = loaded.gather(torch.tensor([3, 0, 3]), device=torch.device("cpu"))
    assert torch.equal(gathered, table[[3, 0, 3]])

    with pytest.raises(ValueError, match="different reference"):
        ReferenceScoreTable(
            root, expected_reference_sha256="wrong", expected_corpus_binding=corpus_binding
        )
    with pytest.raises(ValueError, match="different corpus"):
        ReferenceScoreTable(
            root,
            expected_reference_sha256="deadbeef",
            expected_corpus_binding={**corpus_binding, "version": "v2"},
        )


def test_reference_score_table_fails_closed_when_not_complete(tmp_path: Path) -> None:
    root = tmp_path / "table"
    root.mkdir()
    reference_scores_module.write_manifest(
        root,
        {
            "schema_version": reference_scores_module.MANIFEST_SCHEMA_VERSION,
            "algorithm": reference_scores_module.ALGORITHM,
            "complete": False,
            "reference_sha256": "deadbeef",
            "corpus_binding": {},
            "sequence_length": 4,
            "rows": 1,
            "dtype": "float32",
        },
    )
    with pytest.raises(ValueError, match="not complete"):
        ReferenceScoreTable(root, expected_reference_sha256="deadbeef", expected_corpus_binding={})


def test_token_selection_state_opens_reference_table_for_rho_and_middle_ppl(tmp_path: Path) -> None:
    rows, sequence_length = 2, 4
    root = tmp_path / "table"
    root.mkdir()
    import numpy as np

    array = np.memmap(
        reference_scores_module.table_file(root), mode="w+", dtype=np.float32, shape=(rows, sequence_length)
    )
    array[:] = 0.0
    array.flush()
    corpus_binding = {"dataset_id": "pretrain/regmix-10b", "version": "v1"}
    reference_scores_module.write_manifest(
        root,
        {
            "schema_version": reference_scores_module.MANIFEST_SCHEMA_VERSION,
            "algorithm": reference_scores_module.ALGORITHM,
            "complete": True,
            "reference_sha256": "abc",
            "corpus_binding": corpus_binding,
            "sequence_length": sequence_length,
            "rows": rows,
            "dtype": "float32",
        },
    )

    for method in ("rho_excess", "middle_ppl"):
        config = TokenSelectionConfig(
            method=method,
            keep_fraction=0.6,
            total_steps=10,
            reference_scores_path=str(root),
            reference_scores_reference_sha256="abc",
            reference_scores_corpus_binding=corpus_binding,
        )
        state = TokenSelectionState(config, Tiny())
        assert state.reference_table is not None

    # A reference-scoring method with no table configured fails closed at
    # construction, not at first use.
    with pytest.raises(ValueError, match="requires a reference-scores table"):
        TokenSelectionState(
            TokenSelectionConfig(method="rho_excess", keep_fraction=0.6, total_steps=10),
            Tiny(),
        )

    # Non-reference-scoring methods never touch a table.
    full_state = TokenSelectionState(
        TokenSelectionConfig(method="full", keep_fraction=1.0, total_steps=10), Tiny()
    )
    assert full_state.reference_table is None


def test_batch_index_dry_run_zeros_and_real_step_requires_index() -> None:
    """A1: Trainer.fit's dry-run mock batch has no 'index', so use zeros
    there; every real step must carry one, since it's the table's lookup key."""
    zeros = TokenWeightedTrainModule._batch_index({}, rows=3, dry_run=True)
    assert torch.equal(zeros, torch.zeros(3, dtype=torch.int64))

    with pytest.raises(RuntimeError, match="global instance index"):
        TokenWeightedTrainModule._batch_index({}, rows=3, dry_run=False)

    supplied = torch.tensor([7, 8, 9])
    assert torch.equal(
        TokenWeightedTrainModule._batch_index({"index": supplied}, rows=3, dry_run=False), supplied
    )
    # A real "index" always wins, even in a dry run.
    assert torch.equal(
        TokenWeightedTrainModule._batch_index({"index": supplied}, rows=3, dry_run=True), supplied
    )


def test_attention_capture_hooks_compiled_block_boundary() -> None:
    class Attention(nn.Module):
        def forward(self, x, **_kwargs):
            return x

    class CompiledLikeBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attention = Attention()

        def forward(self, x, **kwargs):
            # Simulate a compiled graph that bypasses child-module __call__ hooks.
            return self.attention.forward(x, **kwargs)

    model = nn.Module()
    model.blocks = nn.ModuleDict({"0": CompiledLikeBlock()})
    expected = torch.randn(2, 3, 4)
    with capture_last_attention(model) as capture:
        model.blocks["0"](expected, start_pos=0)
    assert torch.equal(capture.x, expected)
    assert capture.owner is model.blocks["0"]
    assert capture.kwargs == {"start_pos": 0}


def test_method_polarities_and_per_sequence_selection() -> None:
    valid = torch.ones(2, 4, dtype=torch.bool)
    index = torch.tensor([0, 1])
    current = torch.tensor([[4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0]])
    reference = torch.ones_like(current)
    rho = selection_weights(
        "rho_excess",
        valid=valid,
        keep_fraction=0.5,
        seed=42,
        index=index,
        current=current,
        reference=reference,
    )
    assert rho.bool().tolist() == [
        [True, True, False, False],
        [False, False, True, True],
    ]
    rel = selection_weights(
        "rel_ema",
        valid=valid,
        keep_fraction=0.5,
        seed=42,
        index=index,
        current=current,
        history=reference * 3,
    )
    assert rel.sum(dim=1).tolist() == [2, 2]
    # "blade" is deliberately not a selection_weights method: BLADE computes
    # its own weights in BladeCallback and supplies them via
    # batch["token_weight"], read before selection_weights() is ever called.
    with pytest.raises(ValueError, match="unsupported"):
        selection_weights(
            "blade", valid=valid, keep_fraction=0.5, seed=42, index=index, current=current,
            reference=reference * 3,
        )


def test_middle_ppl_drops_easy_and_hard_and_random_is_index_deterministic() -> None:
    valid = torch.ones(1, 10, dtype=torch.bool)
    middle = selection_weights(
        "middle_ppl",
        valid=valid,
        keep_fraction=0.6,
        seed=42,
        index=torch.tensor([0]),
        reference=torch.arange(10.0).unsqueeze(0),
    )
    assert middle.bool().tolist() == [
        [False, False, True, True, True, True, True, True, False, False]
    ]
    # A2: the mask depends only on (seed, index), never on world size, rank,
    # step, or microbatch position -- calling with the same index twice must
    # give the identical mask, and a different index must (almost certainly)
    # give a different one.
    first = selection_weights(
        "random", valid=valid, keep_fraction=0.6, seed=42, index=torch.tensor([125])
    )
    resumed = selection_weights(
        "random", valid=valid, keep_fraction=0.6, seed=42, index=torch.tensor([125])
    )
    assert torch.equal(first, resumed)
    different_index = selection_weights(
        "random", valid=valid, keep_fraction=0.6, seed=42, index=torch.tensor([126])
    )
    assert not torch.equal(first, different_index)
    different_seed = selection_weights(
        "random", valid=valid, keep_fraction=0.6, seed=69, index=torch.tensor([125])
    )
    assert not torch.equal(first, different_seed)


class Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.0, 0.0]))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.weight


def test_relative_ema_variants_and_resume_state() -> None:
    assert ema_alpha(0, tau=300, constant=None) == 0.0
    assert ema_alpha(300, tau=300, constant=None) == pytest.approx(1 - math.exp(-1))
    assert ema_alpha(100, tau=None, constant=0.9985) == 0.9985

    model = Tiny()
    ema = EMAHistory(model)
    model.weight.data.fill_(2)
    ema.update(model, 0.5)
    model.weight.data.fill_(4)
    ema.update(model, 0.5)
    state = ema.state_dict()
    restored = EMAHistory(Tiny())
    restored.load_state_dict(state)
    assert restored.correction == ema.correction
    assert torch.equal(restored.shadow["weight"], ema.shadow["weight"])

    seeded = EMAHistory(Tiny(), seed={"weight": torch.tensor([3.0, 5.0])})
    assert seeded.correction == 1.0
    with seeded.swap_to(model):
        assert torch.equal(model.weight, torch.tensor([3.0, 5.0]))
    assert torch.equal(model.weight, torch.tensor([4.0, 4.0]))


def _canonical_tensor_bytes(tensor: torch.Tensor) -> bytes:
    """Canonicalize a tensor for hashing.

    Cast to float64, move to CPU, force C-contiguous layout, and prefix with a
    ``shape``/``dtype``/``order`` header (so two tensors that happen to share
    raw bytes but differ in shape can never collide). ``.numpy().tobytes()``
    on a C-contiguous CPU array is deterministic across runs/platforms in the
    native (little-endian on every machine this project runs on) byte order.
    """
    arr = tensor.detach().to("cpu", dtype=torch.float64).contiguous()
    header = f"shape={tuple(arr.shape)};dtype=float64;order=C;".encode("utf-8")
    return header + arr.numpy().tobytes()


def _sha256_of(*chunks: bytes) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()


def _random_mask_hash(*, data_seed: int) -> str:
    """Exercise ``selection_weights("random", ...)`` the way ``train_module.py``
    actually calls it post-A2: ``seed=config.seed`` (the arm's own
    ``data_seed``, never combined with a microbatch offset), ``index`` the
    batch's global instance ids. Includes a padded row, boundary indices (0,
    1), a large instance id near the corpus's real size, and a row-order
    permutation (to confirm the mask for a given index never depends on which
    row it lands in)."""
    valid = torch.ones(3, 12, dtype=torch.bool)
    valid[2, -2:] = False  # a padded row, to exercise per_row_topk's validity handling
    index_batches = [
        torch.tensor([0, 1, 2], dtype=torch.int64),
        torch.tensor([7, 1000, 4_885_155], dtype=torch.int64),
        torch.tensor([2, 1, 0], dtype=torch.int64),
    ]
    masks = [
        selection_weights("random", valid=valid, keep_fraction=0.6, seed=data_seed, index=index)
        for index in index_batches
    ]
    return _sha256_of(_canonical_tensor_bytes(torch.stack(masks)))


def test_random_selection_mask_matches_recorded_hash() -> None:
    """Regression pin for the random-control masks' per-instance derivation (A2/E5).

    Random-control and its seed-69 replicate are rerun under the unified
    commit (they are no longer kept from an earlier commit), so this is a
    plain regression guard on ``selection_weights("random", ...)`` and the
    ``per_row_topk`` it calls -- not a cross-commit compatibility claim. It
    exists so an accidental change to the per-instance derivation (the
    ``per_instance_uniform``/``per_instance_tiebreak`` streams in
    ``selection.py``) is caught immediately rather than only showing up as an
    unexplained shift in a real run's logged masks.

    These hashes were recorded by running the exact procedure in
    ``_random_mask_hash`` once and asserting the result reproduces on every
    later run. If this test fails after an intentional change to the random
    method's derivation, re-record the hash *and* re-record
    ``perplexity``/``rho-1``-adjacent numbers that might have shared a
    seed/index convention; don't just paper over the failure.
    """
    assert ARM_SPECS["random-control"].data_seed == 42
    assert ARM_SPECS["random-control-seed69"].data_seed == 69
    assert _random_mask_hash(data_seed=42) == (
        "248e6fb7221b22054884e7ef3a2868cc8572045b50b69f67cec14617e57831cd"
    )
    assert _random_mask_hash(data_seed=69) == (
        "245592b3a2ef1cff08d64af385a6840f45b134e32d32403a7b802f9f4a69d24a"
    )


def test_per_instance_tiebreak_matches_recorded_hash() -> None:
    """Regression pin for the shared tie-break stream every score-based method uses.

    ``per_row_topk``/``per_row_middle`` break exact ties via this stream
    (see ``selection.py``'s docstring for why: sort by score, then by this
    per-instance uniform, via a stable sort over a random permutation).
    BLADE's per-row selection (``_select_mask``) also calls this directly.
    """
    index = torch.tensor([0, 1, 4_885_155], dtype=torch.int64)
    tb = per_instance_tiebreak(index, data_seed=42, length=8, device=torch.device("cpu"))
    assert _sha256_of(_canonical_tensor_bytes(tb)) == (
        "fcce154a8267157d68ee5ccc7914a89365811d19e53186a3910d397504938b6c"
    )


def test_tiebreak_is_index_deterministic_and_row_order_independent() -> None:
    """All-tied scores: the tie-break stream alone decides which half is kept."""
    valid = torch.ones(2, 6, dtype=torch.bool)
    scores = torch.zeros(2, 6)
    index = torch.tensor([3, 9])
    tb = per_instance_tiebreak(index, data_seed=42, length=6, device=torch.device("cpu"))
    mask_a = per_row_topk(scores, 0.5, valid, tiebreak=tb)
    mask_b = per_row_topk(scores, 0.5, valid, tiebreak=tb)
    assert torch.equal(mask_a, mask_b)

    swapped_index = torch.tensor([9, 3])
    tb_swapped = per_instance_tiebreak(
        swapped_index, data_seed=42, length=6, device=torch.device("cpu")
    )
    mask_swapped = per_row_topk(scores, 0.5, valid, tiebreak=tb_swapped)
    # Row 0 of mask_swapped (index 9) matches row 1 of mask_a (index 9), and
    # vice versa: the mask for a given index doesn't depend on which row of
    # the batch it lands in.
    assert torch.equal(mask_a[0], mask_swapped[1])
    assert torch.equal(mask_a[1], mask_swapped[0])

    # No tiebreak (the legacy/default path): still a valid mask of the right
    # size, just implementation-defined tie order.
    untiebroken = per_row_topk(scores, 0.5, valid)
    assert int(untiebroken.sum()) == 6  # round(0.5*6)=3 per row, 2 rows


def test_ema_history_update_sequence_matches_recorded_hash() -> None:
    """Regression pin for ``rel-ema-exp``'s ``EMAHistory`` update sequence and
    ``ema_alpha`` schedule.

    ``rel-ema-exp`` is rerun under the unified commit like every other arm;
    this is a plain regression guard on ``EMAHistory``/``ema_alpha`` (both
    untouched by the unification, the offline-scoring change, or the
    per-instance mask change -- none of which this arm's method uses), not a
    cross-commit compatibility claim.

    This drives ``EMAHistory(model, seed=None)`` (the "zero" seed rel-ema-exp
    actually uses) through 5 ``.update(model, alpha)`` calls with
    ``alpha = ema_alpha(step, tau=300.0, constant=None)`` -- rel-ema-exp's own
    ``ema_tau`` -- over a fixed, documented sequence of model weights, and
    hashes the resulting sequence of shadow snapshots plus bias-correction
    values. A separate hash covers the ``ema_alpha(t, tau=300.0, ...)``
    schedule (``alpha(t) = 1 - exp(-t/300)`` per the README) at several ``t``.
    """
    assert ARM_SPECS["rel-ema-exp"].ema_seed == "zero"
    assert ARM_SPECS["rel-ema-exp"].ema_tau == 300.0

    model = Tiny()
    ema = EMAHistory(model, seed=None)
    shadow_snapshots = []
    correction_snapshots = []
    for step, value in enumerate([1.0, -2.0, 3.5, -0.25, 5.0]):
        model.weight.data.copy_(torch.tensor([value, -value]))
        alpha = ema_alpha(step, tau=300.0, constant=None)
        ema.update(model, alpha)
        shadow_snapshots.append(ema.shadow["weight"].clone())
        correction_snapshots.append(ema.correction)
    weights_history = torch.stack(shadow_snapshots)
    corrections = torch.tensor(correction_snapshots, dtype=torch.float64)
    ema_history_hash = _sha256_of(
        _canonical_tensor_bytes(weights_history), _canonical_tensor_bytes(corrections)
    )
    assert ema_history_hash == (
        "adcfd299d10cebcc98dd77ed95c0ffe2bf8df3d71a476e9cc7680ab64a23bb32"
    )

    steps = (0, 1, 50, 150, 299, 300, 301, 600, 2360)
    schedule = torch.tensor(
        [ema_alpha(t, tau=300.0, constant=None) for t in steps], dtype=torch.float64
    )
    ema_alpha_schedule_hash = _sha256_of(_canonical_tensor_bytes(schedule))
    assert ema_alpha_schedule_hash == (
        "0f68bbc094d2e632910a9b3dc41fe5ba2f94c436736d2bca4e97c9c9a40791fd"
    )


def test_attention_received_matches_causal_definition() -> None:
    query = torch.zeros(1, 3, 1, 2)
    key = torch.zeros_like(query)
    # Uniform causal attention: received mass is 1 + 1/2 + 1/3, 1/2 + 1/3, 1/3.
    scores = attention_received_from_qk(query, key)
    assert scores[0].tolist() == pytest.approx([11 / 6, 5 / 6, 1 / 3])


def test_uniform_attention_normalizer_matches_the_raw_uniform_score() -> None:
    # e_i = sum_{j=i}^{L-1} 1/(j+1); for L=3 this is exactly [11/6, 5/6, 1/3],
    # i.e. the raw score a uniform-attention query would produce (see above).
    normalizer = uniform_attention_normalizer(3, torch.device("cpu"))
    assert normalizer.tolist() == pytest.approx([11 / 6, 5 / 6, 1 / 3])


def test_old_uniform_normalized_attention_scoring_is_gone() -> None:
    # The raw / uniform_attention_normalizer score degenerates to a
    # keep-the-tail heuristic on real (recency-biased) attention; nothing may
    # keep importing it. uniform_attention_normalizer survives only as the
    # cold-start prior.
    assert not hasattr(selection_module, "normalize_and_align_attention_scores")
    assert not hasattr(selection_module, "aligned_normalized_attention_scores")


def test_align_log_attention_mass_shifts_a_spike_to_the_label_it_gates() -> None:
    # A raw score with all its mass on position k=3 must, after the +1 target
    # shift, select the *label* index k-1=2 (label t predicts input t+1).
    raw = torch.full((1, 5), 1e-3)
    raw[0, 3] = 1000.0
    aligned = align_log_attention_mass(raw)
    assert aligned.shape == (1, 5)
    assert int(aligned[0].argmax()) == 2
    assert aligned[0, 2] == pytest.approx(math.log(1000.0))
    # The last column has no label under get_labels' left shift and must never
    # be selectable -- and stays -inf through the baseline's standardization.
    assert aligned[0, -1] == -torch.inf
    assert AttentionPositionBaseline().score(aligned)[0, -1] == -torch.inf
    # An underflowed (exactly zero) column mass is clamped, not -inf/NaN.
    assert torch.isfinite(align_log_attention_mass(torch.zeros(1, 3))[0, :-1]).all()


def test_cold_start_prior_gives_flat_scores_under_uniform_attention() -> None:
    # Before any statistics exist, the baseline is the uniform-attention
    # prior: a model attending exactly uniformly scores 0 everywhere.
    query = torch.zeros(1, 6, 1, 2)
    raw = attention_received_from_qk(query, torch.zeros_like(query))
    baseline = AttentionPositionBaseline()
    assert not baseline.has_history
    scores = baseline.score(align_log_attention_mass(raw))
    assert scores[0, :-1].tolist() == pytest.approx([0.0] * 5, abs=1e-5)
    assert scores[0, -1] == -torch.inf
    prior = uniform_prior_log_mass(6, torch.device("cpu"))
    assert prior[:-1].tolist() == pytest.approx(
        uniform_attention_normalizer(6, torch.device("cpu"), torch.float64)[1:].log().tolist()
    )


def test_cold_start_prior_ranks_exactly_like_the_uniform_normalized_score() -> None:
    # On the first step of a fresh run (no history) the new score orders every
    # row's tokens exactly as the previous raw / uniform_attention_normalizer
    # score did: log is monotone and the prior has unit std.
    generator = torch.Generator().manual_seed(0)
    raw = torch.rand(3, 16, generator=generator) + 0.05
    valid = torch.ones(3, 16, dtype=torch.bool)
    valid[:, -1] = False
    normalizer = uniform_attention_normalizer(16, raw.device, raw.dtype)
    legacy = torch.nn.functional.pad((raw / normalizer)[:, 1:], (0, 1), value=-torch.inf)
    cold = AttentionPositionBaseline().score(align_log_attention_mass(raw))
    assert torch.equal(
        per_row_topk(legacy, 0.6, valid), per_row_topk(cold, 0.6, valid)
    )
    assert torch.equal(legacy[:, :-1].argsort(-1), cold[:, :-1].argsort(-1))


def _moments_reference(values: torch.Tensor, valid: torch.Tensor):
    """Per-position (count, mean, population std) over rows, float64, by brute force."""
    values, valid = values.double(), valid & torch.isfinite(values)
    count = valid.double().sum(0)
    mean = torch.where(valid, values, 0.0).sum(0) / count
    var = torch.where(valid, (values - mean) ** 2, 0.0).sum(0) / count
    return count, mean, var.sqrt()


def test_attention_baseline_matches_pooled_statistics_then_replaces_unblended() -> None:
    generator = torch.Generator().manual_seed(1)
    length = 6
    step_a = torch.randn(10, length, generator=generator, dtype=torch.float64).float()
    valid_a = torch.ones(10, length, dtype=torch.bool)
    valid_a[:, -1] = False  # the label-less column
    valid_a[0, 2] = False  # a masked label must not count
    step_a[1, 3] = -torch.inf  # a non-finite value must not count either
    valid_seen = valid_a & torch.isfinite(step_a)

    baseline = AttentionPositionBaseline()
    # Two microbatches of one step accumulate into the same pending moments.
    baseline.observe(step_a[:4], valid_a[:4])
    baseline.observe(step_a[4:], valid_a[4:])
    assert not baseline.has_history  # nothing is used before commit
    baseline.commit()
    assert baseline.has_history and baseline.pending is None
    mean, std = baseline.position_statistics(length, torch.device("cpu"))
    _, ref_mean, ref_std = _moments_reference(step_a[:, :-1], valid_seen[:, :-1])
    assert mean[:-1].tolist() == pytest.approx(ref_mean.tolist())
    assert std[:-1].tolist() == pytest.approx(ref_std.tolist())
    # The never-observed last column falls back to the pooled statistics.
    pooled = step_a[valid_seen].double()
    assert float(mean[-1]) == pytest.approx(float(pooled.mean()))
    assert float(std[-1]) == pytest.approx(float(pooled.std(unbiased=False)))

    # A second step: the baseline is replaced wholesale by the new step's own
    # statistics, not blended with the first step's -- a resumed run scores
    # its next step purely against what it just observed.
    step_b = torch.randn(10, length, generator=generator, dtype=torch.float64).float() + 3.0
    valid_b = torch.ones(10, length, dtype=torch.bool)
    valid_b[:, -1] = False
    baseline.observe(step_b, valid_b)
    baseline.commit()
    mean, std = baseline.position_statistics(length, torch.device("cpu"))
    _, ref_mean_b, ref_std_b = _moments_reference(step_b[:, :-1], valid_b[:, :-1])
    assert mean[:-1].tolist() == pytest.approx(ref_mean_b.tolist())
    assert std[:-1].tolist() == pytest.approx(ref_std_b.tolist())


def test_attention_baseline_scores_are_z_scores_per_position() -> None:
    generator = torch.Generator().manual_seed(2)
    length = 32
    # Strongly position-dependent location *and* spread.
    position = torch.arange(length, dtype=torch.float32)
    values = torch.randn(400, length, generator=generator) * (0.1 + position / 8) - position
    valid = torch.ones(400, length, dtype=torch.bool)
    baseline = AttentionPositionBaseline()
    baseline.observe(values, valid)
    baseline.commit()
    scores = baseline.score(values).double()
    assert scores.mean(0).abs().max() < 1e-6
    assert (scores.std(0, unbiased=False) - 1.0).abs().max() < 1e-6


def test_attention_baseline_resume_is_bit_exact() -> None:
    import io

    generator = torch.Generator().manual_seed(3)
    steps = [torch.randn(2, 4, 8, generator=generator) for _ in range(5)]
    valid = torch.ones(4, 8, dtype=torch.bool)
    probe = torch.randn(3, 8, generator=generator)

    def run(baseline, microbatches):
        for step in microbatches:
            for micro in step:
                baseline.observe(micro, valid)
            baseline.commit()

    uninterrupted = AttentionPositionBaseline()
    run(uninterrupted, steps)

    first = AttentionPositionBaseline()
    run(first, steps[:2])
    first.observe(steps[2][0], valid)  # per-step scratch is never saved
    buffer = io.BytesIO()
    torch.save(first.state_dict(), buffer)
    buffer.seek(0)
    resumed = AttentionPositionBaseline()
    resumed.load_state_dict(torch.load(buffer))
    assert resumed.pending is None
    run(resumed, steps[2:])
    assert torch.equal(resumed.moments, uninterrupted.moments)
    assert torch.equal(resumed.score(probe), uninterrupted.score(probe))

    # A step-0 checkpoint (no history yet) round-trips too.
    empty = AttentionPositionBaseline()
    empty.load_state_dict(AttentionPositionBaseline().state_dict())
    assert not empty.has_history

    state = uninterrupted.state_dict()
    with pytest.raises(ValueError, match="unsupported"):
        AttentionPositionBaseline().load_state_dict({**state, "version": 0})
    with pytest.raises(ValueError, match="length"):
        AttentionPositionBaseline().load_state_dict({**state, "moments": torch.zeros(2, 8)})
    with pytest.raises(ValueError, match="positions"):
        uninterrupted.score(torch.zeros(1, 9))


def test_attention_baseline_all_reduces_pending_over_the_given_group(monkeypatch) -> None:
    calls = []

    def fake_all_reduce(tensor, group=None):
        calls.append(group)
        tensor.mul_(2)  # two ranks that happened to see identical data

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "all_reduce", fake_all_reduce)
    baseline = AttentionPositionBaseline()
    values = torch.tensor([[1.0, 2.0, 3.0]])
    baseline.observe(values, torch.ones(1, 3, dtype=torch.bool))
    baseline.all_reduce_pending("dp-group")
    assert calls == ["dp-group"]
    assert baseline.pending[0].tolist() == [2.0, 2.0, 2.0]
    baseline.commit()
    mean, _ = baseline.position_statistics(3, torch.device("cpu"))
    assert mean.tolist() == pytest.approx([1.0, 2.0, 3.0])  # reduction preserves the mean

    with pytest.raises(RuntimeError, match="no observations"):
        AttentionPositionBaseline().all_reduce_pending()


def _recency_biased_attention(rows: int, length: int, *, seed: int):
    """Synthetic last-layer q/k with trained-model-like recency bias plus content.

    A 2-D rotary-style pair gives every query a logit ``8 cos(pi (j - i) / L)``
    on key ``i`` -- decreasing with distance, so each query concentrates on
    nearby keys, as real trained causal attention does -- and a third
    dimension adds a per-token "salience" ``g_i`` every query attends to. The
    salience is the genuine per-token signal a position-corrected score must
    recover.
    """
    generator = torch.Generator().manual_seed(seed)
    angle = math.pi / length * torch.arange(length, dtype=torch.float32)
    salience = torch.randn(rows, length, generator=generator)
    query = torch.zeros(rows, length, 2, 3)
    key = torch.zeros(rows, length, 2, 3)
    query[..., 0] = (8.0 * torch.cos(angle))[:, None]
    query[..., 1] = (8.0 * torch.sin(angle))[:, None]
    query[..., 2] = 1.0
    key[..., 0] = torch.cos(angle)[:, None]
    key[..., 1] = torch.sin(angle)[:, None]
    key[..., 2] = salience[..., None]
    return attention_received_from_qk(query, key, scale=1.0), salience


def test_empirical_baseline_removes_recency_bias_the_uniform_normalizer_amplifies() -> None:
    """Synthetic replica of farmshare/attention_diagnostic.py's failure and fix.

    Same fixed rule as the diagnostic: 16 position bins, every bin's keep rate
    in [0.3, 0.9]. Under recency-biased attention the old raw /
    uniform-normalizer score keeps the tail and drops the head (the real
    checkpoint's failure, reproduced), while the empirical baseline --
    calibrated on rows disjoint from the scored ones, as the diagnostic and
    training both do -- keeps every bin near keep_fraction *and* selects by
    the per-token salience, not by position.
    """
    length, bins, keep = 256, 16, ARM_SPECS["attention"].keep_fraction
    raw, salience = _recency_biased_attention(32, length, seed=0)
    calibration, _ = _recency_biased_attention(64, length, seed=1)
    valid = torch.ones(32, length, dtype=torch.bool)
    valid[:, -1] = False  # get_labels' label-less last column

    def bin_rates(mask):
        width = length // bins
        return [
            float(mask[:, b * width : (b + 1) * width].sum() / valid[:, b * width : (b + 1) * width].sum())
            for b in range(bins)
        ]

    normalizer = uniform_attention_normalizer(length, raw.device, raw.dtype)
    legacy = torch.nn.functional.pad((raw / normalizer)[:, 1:], (0, 1), value=-torch.inf)
    legacy_rates = bin_rates(per_row_topk(legacy, keep, valid))
    assert legacy_rates[0] < 0.3 and legacy_rates[-1] > 0.9
    assert legacy_rates[-1] - legacy_rates[0] > 0.5

    baseline = AttentionPositionBaseline()
    calibration_valid = torch.ones(64, length, dtype=torch.bool)
    calibration_valid[:, -1] = False
    for half in (slice(0, 32), slice(32, 64)):
        baseline.observe(align_log_attention_mass(calibration[half]), calibration_valid[half])
    baseline.commit()
    scores = baseline.score(align_log_attention_mass(raw))
    mask = per_row_topk(scores, keep, valid)
    rates = bin_rates(mask)
    assert all(0.45 <= rate <= 0.75 for rate in rates), rates

    # Per-token, not positional: the score recovers the target token's own
    # salience (label t is gated by input token t+1).
    target_salience = torch.nn.functional.pad(salience[:, 1:], (0, 1))
    correlation = torch.corrcoef(
        torch.stack([scores[valid].double(), target_salience[valid].double()])
    )[0, 1]
    assert correlation > 0.9
    assert target_salience[mask].mean() - target_salience[valid & ~mask].mean() > 1.0


def test_token_selection_state_persists_attention_baseline() -> None:
    config = TokenSelectionConfig(method="attention_topk", keep_fraction=0.6, total_steps=10)
    state = TokenSelectionState(config, Tiny())
    assert isinstance(state.attention_baseline, AttentionPositionBaseline)
    assert state.ema is None
    values = torch.randn(2, 4)
    state.attention_baseline.observe(values, torch.ones(2, 4, dtype=torch.bool))
    state.after_optimizer_step(Tiny())
    assert state.attention_baseline.has_history and state.attention_baseline.pending is None
    assert state.completed_steps == 1

    saved = state.state_dict()
    assert saved["version"] == 1 and saved["ema"] is None
    restored = TokenSelectionState(config, Tiny())
    restored.load_state_dict(saved)
    assert restored.completed_steps == 1
    assert torch.equal(restored.attention_baseline.moments, state.attention_baseline.moments)

    # An attention arm fails closed on a state without its baseline (e.g. one
    # written before the baseline existed) instead of silently restarting it.
    legacy = {"version": 1, "completed_steps": 5, "ema": None}
    with pytest.raises(ValueError, match="missing its position baseline"):
        TokenSelectionState(config, Tiny()).load_state_dict(legacy)

    # Every other arm: no baseline, still loads a state written before the key
    # existed, and rejects a stray baseline.
    full = TokenSelectionState(
        TokenSelectionConfig(method="full", keep_fraction=1.0, total_steps=10), Tiny()
    )
    assert full.attention_baseline is None
    assert full.state_dict()["attention_baseline"] is None
    full.load_state_dict(legacy)
    assert full.completed_steps == 5
    with pytest.raises(ValueError, match="non-attention arm"):
        full.load_state_dict(saved)


def _function_node(tree: ast.AST, name: str) -> ast.FunctionDef:
    return next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _calls_named(node: ast.AST, attribute: str) -> list[ast.Call]:
    return [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, (ast.Attribute, ast.Name))
        and (call.func.attr if isinstance(call.func, ast.Attribute) else call.func.id) == attribute
    ]


def test_train_batch_scores_before_observing_and_reduces_once_per_step() -> None:
    source = (EDULLM_ROOT / "token_selection_370m" / "train_module.py").read_text(encoding="utf-8")
    assert "normalize_and_align" not in source
    assert "aligned_normalized_attention_scores" not in source
    assert "uniform_attention_normalizer" not in source
    tree = ast.parse(source)
    train_batch = _function_node(tree, "train_batch")
    microbatch_loop = next(
        node
        for node in ast.walk(train_batch)
        if isinstance(node, ast.For) and "enumerate(prepared)" in ast.unparse(node.iter)
    )

    # Inside the microbatch loop: log mass from the capture, scored against
    # the already-committed baseline, and only then observed -- and never in
    # a dry run.
    (mass_call,) = _calls_named(microbatch_loop, "aligned_log_attention_mass")
    assert ast.unparse(mass_call) == "aligned_log_attention_mass(captured)"
    (score_call,) = _calls_named(microbatch_loop, "score")
    assert ast.unparse(score_call) == "baseline.score(log_mass)"
    (observe_call,) = _calls_named(microbatch_loop, "observe")
    assert ast.unparse(observe_call) == "baseline.observe(log_mass, valid)"
    assert score_call.lineno < observe_call.lineno
    guards = [
        node
        for node in ast.walk(microbatch_loop)
        if isinstance(node, ast.If) and observe_call in list(ast.walk(node))
    ]
    assert any(ast.unparse(guard.test) == "not dry_run" for guard in guards)

    # Exactly one all-reduce per step, after (outside) the microbatch loop,
    # over the data-parallel group, and never in a dry run.
    assert not _calls_named(microbatch_loop, "all_reduce_pending")
    (reduce_call,) = _calls_named(train_batch, "all_reduce_pending")
    assert ast.unparse(reduce_call) == (
        "state.attention_baseline.all_reduce_pending(self.dp_process_group)"
    )
    assert reduce_call.lineno > microbatch_loop.end_lineno
    reduce_guard = next(
        node
        for node in ast.walk(train_batch)
        if isinstance(node, ast.If) and reduce_call in list(ast.walk(node))
    )
    assert ast.unparse(reduce_guard.test).startswith("not dry_run")

    # The commit happens at the optimizer-step boundary the checkpointer sees.
    after_step = _function_node(tree, "after_optimizer_step")
    assert [ast.unparse(call) for call in _calls_named(after_step, "commit")] == [
        "self.attention_baseline.commit()"
    ]


def test_attention_diagnostic_mirrors_the_train_time_scoring() -> None:
    source = (EDULLM_ROOT / "farmshare" / "attention_diagnostic.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert "normalize_and_align" not in source
    assert "aligned_normalized_attention_scores" not in source
    calls = {ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
    assert {
        "AttentionPositionBaseline",
        "aligned_log_attention_mass",
        "baseline.observe",
        "baseline.commit",
        "baseline.score",
        "per_row_topk",
    } <= calls
    # The acceptance rule is unchanged and still not a CLI knob.
    assert "KEEP_RATE_LOW = 0.3" in source and "KEEP_RATE_HIGH = 0.9" in source
    assert "NUM_BINS = 16" in source
    assert "--keep-rate" not in source
    # Calibration rows are disjoint from the scored rows.
    assert "calibration_batches = all_batches[: args.calibration_batches]" in source
    assert "batches = all_batches[args.calibration_batches :]" in source


class FakeStream:
    def __init__(self, cursor: int):
        self.cursor = cursor

    def state_dict(self):
        return {"cursor": self.cursor}

    def load_state_dict(self, state):
        self.cursor = state["cursor"]


class FakeBatchStream:
    """A ResumableBatchStream stand-in that hands out distinct fake batches."""

    def __init__(self, cursor: int = 0) -> None:
        self.cursor = cursor

    def next(self) -> dict:
        batch = {
            "input_ids": torch.full((1, 2), self.cursor, dtype=torch.long),
            "index": torch.tensor([self.cursor], dtype=torch.int64),
        }
        self.cursor += 1
        return batch

    def state_dict(self):
        return {"cursor": self.cursor}

    def load_state_dict(self, state):
        self.cursor = state["cursor"]


def _blade_callback(train_cursor=3, hq_cursor=7) -> BladeCallback:
    return BladeCallback(
        total_steps=2360,
        reference_factory=Tiny,
        reference_train_stream=FakeBatchStream(train_cursor),  # type: ignore[arg-type]
        refhq_stream=FakeBatchStream(hq_cursor),  # type: ignore[arg-type]
    )


def test_blade_proxy_sync_materializes_full_state_on_every_rank(monkeypatch) -> None:
    import torch.distributed.checkpoint.state_dict as dist_cp_state

    captured = {}

    def get_model_state_dict(model, *, options):
        captured["options"] = options
        return model.state_dict()

    monkeypatch.setattr(dist_cp_state, "get_model_state_dict", get_model_state_dict)
    state = _full_proxy_state(Tiny())

    assert set(state) == {"weight"}
    assert captured["options"].full_state_dict is True
    assert captured["options"].cpu_offload is False


def test_blade_new_reference_uses_given_lr_and_zero_decay_on_embeddings() -> None:
    class WithEmbeddings(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embeddings = nn.Module()
            self.embeddings.weight = nn.Parameter(torch.zeros(2))
            self.weight = nn.Parameter(torch.zeros(2))

    callback = BladeCallback(
        total_steps=2360,
        reference_factory=WithEmbeddings,
        reference_train_stream=FakeStream(0),  # type: ignore[arg-type]
        refhq_stream=FakeStream(0),  # type: ignore[arg-type]
    )
    callback._new_reference(lr=1e-3)
    assert callback.reference_optim is not None
    weight_decays = sorted(g["weight_decay"] for g in callback.reference_optim.param_groups)
    assert weight_decays == [0.0, 0.1]
    assert all(g["lr"] == 1e-3 for g in callback.reference_optim.param_groups)

    # A second call replaces both the module and the optimizer (no momentum
    # carried across syncs): the old optimizer's state is gone.
    first_reference = callback.reference
    callback._new_reference(lr=5e-4)
    assert callback.reference is not first_reference
    assert all(g["lr"] == 5e-4 for g in callback.reference_optim.param_groups)


def test_blade_reference_lr_is_floored_at_post_warmup() -> None:
    """E1a: the step-0 sync must not train at LR 0.

    A real ``CosWithWarmup`` schedule returns exactly 0.0 at step 0 (by
    construction, during warmup). Reading the proxy's own scheduled LR
    literally at the sync step -- the paper's "eta_w = eta_u" -- would leave
    all 75 step-0 K-updates as no-ops, since AdamW at LR 0 changes nothing.
    ``_reference_lr`` floors the queried step at ``reference_warmup_steps``,
    so the step-0 sync reads the scheduler as if it were already past
    warmup, while every later sync (already past warmup) is unaffected.
    """
    from olmo_core.optim import CosWithWarmup

    warmup_steps = 24
    scheduler = CosWithWarmup(warmup=warmup_steps, alpha_f=0.1)
    peak_lr = 4e-4
    total_steps = 2360

    # Sanity: the raw schedule really does return 0 at step 0 and something
    # positive once warmup has actually elapsed, confirming the bug this
    # fix addresses is real.
    assert scheduler.get_lr(peak_lr, 0, total_steps) == 0.0
    assert scheduler.get_lr(peak_lr, warmup_steps, total_steps) == pytest.approx(peak_lr)

    callback = BladeCallback(
        total_steps=total_steps,
        reference_factory=Tiny,
        reference_train_stream=FakeStream(0),  # type: ignore[arg-type]
        refhq_stream=FakeStream(0),  # type: ignore[arg-type]
        reference_scheduler=scheduler,
        reference_initial_lr=peak_lr,
        reference_warmup_steps=warmup_steps,
    )

    # Step 0: floored at the post-warmup LR, not the raw (zero) schedule value.
    assert callback._reference_lr(0) == pytest.approx(peak_lr)
    # A later sync (already well past warmup) is unaffected by the floor.
    assert callback._reference_lr(400) == scheduler.get_lr(peak_lr, 400, total_steps)
    assert callback._reference_lr(400) < peak_lr  # cosine has decayed by then


def test_blade_scores_reference_under_bf16_params_without_mutating_it() -> None:
    """E1d: the scoring forward borrows bf16-cast weights via functional_call;
    the reference's own (real, fp32) parameters, which the K-updates' AdamW
    trains, must never be mutated by this."""
    observed_dtypes = []

    class Recorder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(2))

        def forward(self, ids, **_kwargs):
            observed_dtypes.append(self.weight.dtype)
            return types.SimpleNamespace(ce_loss=ids.float())

    callback = _blade_callback()
    callback.reference = Recorder()
    assert callback.reference.weight.dtype == torch.float32

    ids = torch.zeros(1, 2, dtype=torch.long)
    labels = torch.zeros(1, 2, dtype=torch.long)
    callback._score_reference_bf16(ids, labels, {})

    assert observed_dtypes == [torch.bfloat16]
    assert callback.reference.weight.dtype == torch.float32


def test_blade_sync_parity_flags_a_real_divergence(monkeypatch) -> None:
    """E1d's post-sync check: if the two paths disagree, log a clear warning
    rather than silently accepting a corrupted selection signal."""
    callback = _blade_callback()
    callback.trainer = types.SimpleNamespace(
        train_module=types.SimpleNamespace(label_ignore_index=-100)
    )
    labels = torch.tensor([[1, 2, 3]])
    monkeypatch.setattr(
        callback,
        "_proxy_and_reference_ce",
        lambda batch: (labels, torch.tensor([[1.0, 1.0, 1.0]]), torch.tensor([[1.0, 1.0, 5.0]])),
    )
    warnings: list[str] = []
    import token_selection_370m.blade as blade_module

    monkeypatch.setattr(
        blade_module.log, "warning", lambda msg, *args: warnings.append(msg % args)
    )

    callback._log_sync_parity(0, {"input_ids": labels})

    assert any("below the 99.9%" in message for message in warnings)


def test_blade_backward_mean_ce_masks_the_train_term_and_leaves_val_unmasked(monkeypatch) -> None:
    callback = _blade_callback()
    callback._new_reference(lr=4e-4)
    callback.reference_microbatch_tokens = 4
    calls = []
    batch = {"input_ids": torch.arange(8, dtype=torch.long).reshape(4, 2)}

    def mean_ce_with_weight(model, micro_batch, weight, divisor):
        calls.append((micro_batch["input_ids"].clone(), weight.clone(), float(divisor)))
        return model.weight.sum() * micro_batch["input_ids"].float().sum() * weight.sum() / divisor

    monkeypatch.setattr(callback, "_mean_ce_with_weight", mean_ce_with_weight)
    assert callback.reference is not None

    # Unmasked (mask=None): weight equals validity exactly (get_labels pads
    # the last column of this 2-token-per-row fixture to -100), matching the
    # RefHQ/Instruct L_val term.
    from token_selection_370m.blade import get_labels

    valid = get_labels(batch) != -100
    callback._backward_mean_ce(callback.reference, batch, weight=1.0, mask=None)
    assert len(calls) == 2
    assert torch.equal(torch.cat([call[1] for call in calls]), valid)
    assert all(call[2] == float(valid.sum()) for call in calls)

    calls.clear()
    # Masked: only positions that are both selected and valid count, and the
    # divisor is that (mask & valid) total, not the full valid count -- this
    # is the alpha-weighted L_train term.
    mask = torch.tensor([[True, False], [False, True], [True, True], [False, False]])
    weighted = mask & valid
    callback._backward_mean_ce(callback.reference, batch, weight=0.6, mask=mask)
    assert len(calls) == 2
    assert torch.equal(torch.cat([call[1] for call in calls]), weighted)
    assert all(call[2] == float(weighted.sum()) for call in calls)


def test_blade_select_mask_is_per_row_with_tiebreak() -> None:
    """E1c: BLADE's selection is per_row_topk, the same unit and round()
    formula every other arm uses -- not the paper's pooled ceil(gamma|B|)
    over the whole batch. Ties are broken by the shared per-instance stream,
    keyed on this arm's data_seed and the batch's global index."""
    callback = _blade_callback()
    callback.trainer = types.SimpleNamespace(
        train_module=types.SimpleNamespace(label_ignore_index=-100)
    )
    labels = torch.tensor([[10, 11, 12, 13, 14]])
    proxy_ce = torch.tensor([[5.0, 5.0, 5.0, 5.0, 1.0]])
    reference_ce = torch.zeros_like(proxy_ce)  # every excess score ties at 5.0 except one
    index = torch.tensor([0])

    mask = callback._select_mask(labels, proxy_ce, reference_ce, index)

    # round(0.6*5)=3 per row, exactly per_row_topk's formula.
    assert int(mask.sum()) == 3
    # Same index -> same tie-break -> same mask, deterministically.
    assert torch.equal(mask, callback._select_mask(labels, proxy_ce, reference_ce, index))


def test_blade_selection_scoring_microbatches_full_rank_batch(monkeypatch) -> None:
    callback = _blade_callback()
    callback.selection_microbatch_tokens = 4
    batch = {"input_ids": torch.arange(8, dtype=torch.long).reshape(4, 2)}
    calls = []

    def score_microbatch(micro_batch):
        ids = micro_batch["input_ids"]
        calls.append(ids.clone())
        labels = ids.clone()
        return labels, ids.float() + 1, ids.float() + 3

    monkeypatch.setattr(callback, "_proxy_and_reference_ce_microbatch", score_microbatch)
    labels, proxy_ce, reference_ce = callback._proxy_and_reference_ce(batch)

    assert len(calls) == 2
    assert all(call.shape == (2, 2) for call in calls)
    assert torch.equal(labels, batch["input_ids"])
    assert torch.equal(proxy_ce, batch["input_ids"].float() + 1)
    assert torch.equal(reference_ce, batch["input_ids"].float() + 3)


def test_blade_pre_step_writes_token_weight_not_labels(monkeypatch) -> None:
    callback = _blade_callback()
    callback._new_reference(lr=4e-4)
    callback.last_sync = 400
    callback.trainer = types.SimpleNamespace(
        global_step=450,
        train_module=types.SimpleNamespace(label_ignore_index=-100),
    )
    labels = torch.tensor([[10, 11, 12, 13]])
    proxy_ce = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    reference_ce = torch.full_like(proxy_ce, 3.0)
    monkeypatch.setattr(
        callback,
        "_proxy_and_reference_ce",
        lambda batch: (labels, proxy_ce, reference_ce),
    )
    batch = {"input_ids": labels.clone(), "index": torch.tensor([0])}

    callback.pre_step(batch)

    # round(0.6*4) = 2 tokens (per_row_topk's formula, per row), by
    # proxy-reference (4-3=1, 3-3=0, 2-3=-1, 1-3=-2): positions 0,1 beat
    # positions 2,3. Written as a weight, never touching "labels".
    assert "labels" not in batch
    assert batch["token_weight"].bool().tolist() == [[True, True, False, False]]


def test_blade_pre_step_raises_without_a_reference() -> None:
    callback = _blade_callback()
    callback.last_sync = 400
    callback.trainer = types.SimpleNamespace(
        global_step=450,
        train_module=types.SimpleNamespace(label_ignore_index=-100),
    )
    with pytest.raises(RuntimeError, match="no dynamic reference"):
        callback.pre_step({"input_ids": torch.tensor([[10, 11]])})


def test_blade_perform_sync_first_sync_has_no_outgoing_reference_to_score(monkeypatch) -> None:
    callback = _blade_callback()
    callback.trainer = types.SimpleNamespace(global_step=0)
    scored = []
    sync_calls = []
    monkeypatch.setattr(
        callback, "_score_regmix_mask", lambda batch: scored.append(batch) or torch.tensor([True])
    )
    monkeypatch.setattr(
        callback,
        "_sync_from_proxy",
        lambda *, lr: (sync_calls.append(lr), callback._new_reference(lr=lr))[-1],
    )
    monkeypatch.setattr(callback, "_run_k_updates", lambda *a, **k: None)
    monkeypatch.setattr(callback, "_log_sync_parity", lambda *a, **k: None)

    callback._perform_sync(0)

    # No outgoing reference exists yet, so no batch is scored: every K-update
    # uses alpha=1 (every RegMix token counts), matching the paper's
    # from-scratch-with-no-warmup first episode.
    assert scored == []
    # E1a: the reference LR is floored at its post-warmup value, so even a
    # step-0 sync (where the proxy's own scheduler is still in warmup) trains
    # at a real LR, not 0. The default test schedule/warmup (_ConstantSchedule,
    # reference_warmup_steps=0) already returns the constant 4e-4 either way;
    # see test_blade_reference_lr_is_floored_at_post_warmup for the case that
    # actually exercises the floor against a real warmup schedule.
    assert sync_calls == [4e-4]
    assert callback.last_sync == 0
    assert callback.reference is not None


def test_blade_perform_sync_scores_before_overwriting_the_outgoing_reference(monkeypatch) -> None:
    callback = _blade_callback()
    callback._new_reference(lr=4e-4)  # an existing ("outgoing") reference
    outgoing = callback.reference
    callback.trainer = types.SimpleNamespace(global_step=399)
    events = []

    def score(batch):
        events.append(("score", callback.reference is outgoing))
        return torch.tensor([True])

    def sync_from_proxy(*, lr):
        events.append(("sync", callback.reference is outgoing))
        callback.reference = Tiny()  # simulate the overwrite

    monkeypatch.setattr(callback, "_score_regmix_mask", score)
    monkeypatch.setattr(callback, "_sync_from_proxy", sync_from_proxy)
    monkeypatch.setattr(callback, "_run_k_updates", lambda *a, **k: None)
    monkeypatch.setattr(callback, "_log_sync_parity", lambda *a, **k: None)

    callback._perform_sync(400)

    # Every score happened while self.reference was still the outgoing one,
    # and strictly before the sync that replaces it.
    assert events[: callback.schedule.k_steps] == [("score", True)] * callback.schedule.k_steps
    assert events[callback.schedule.k_steps] == ("sync", True)
    assert callback.last_sync == 400


def test_blade_sync_boundary_saves_before_and_after_sync(monkeypatch) -> None:
    callback = _blade_callback()
    callback.trainer = types.SimpleNamespace(global_step=399)
    events = []
    monkeypatch.setattr(
        callback, "_perform_sync", lambda sync_step: events.append(f"sync-{sync_step}")
    )
    monkeypatch.setattr(
        callback,
        "_save_sync_checkpoint",
        lambda *, step, phase: events.append(f"save-{phase}-{step}"),
    )

    callback.post_train_batch()

    assert events == ["save-pre-400", "sync-400", "save-post-400"]
    assert callback.completed_step == 399


def test_blade_pre_train_runs_the_step_zero_sync_exactly_once(monkeypatch) -> None:
    callback = _blade_callback()
    events = []
    monkeypatch.setattr(callback, "_perform_sync", lambda sync_step: events.append(sync_step))
    monkeypatch.setattr(
        callback,
        "_save_sync_checkpoint",
        lambda *, step, phase: events.append(f"save-{phase}-{step}"),
    )

    callback.pre_train()
    assert events == ["save-pre-0", 0, "save-post-0"]

    # A resume past the first sync (last_sync already set) must not re-run it.
    events.clear()
    callback.last_sync = 0
    callback.reference = Tiny()
    callback.pre_train()
    assert events == []


def test_blade_pre_step_fallback_reruns_a_pending_sync_on_resume(monkeypatch) -> None:
    # Simulates resuming from a "pre" checkpoint: last_sync is still at the
    # *previous* sync, but completed_step+1 is the next sync step.
    callback = _blade_callback()
    callback._new_reference(lr=4e-4)
    callback.completed_step = 399
    callback.last_sync = 0
    callback.trainer = types.SimpleNamespace(
        global_step=400,
        train_module=types.SimpleNamespace(label_ignore_index=-100),
    )
    events = []
    monkeypatch.setattr(callback, "_perform_sync", lambda sync_step: events.append(sync_step))
    monkeypatch.setattr(
        callback,
        "_proxy_and_reference_ce",
        lambda batch: (
            torch.tensor([[1]]),
            torch.tensor([[1.0]]),
            torch.tensor([[0.0]]),
        ),
    )

    callback.pre_step({"input_ids": torch.tensor([[1]]), "index": torch.tensor([0])})

    assert events == [400]


def test_resume_prefers_post_sync_boundary_until_normal_step_catches_up(tmp_path) -> None:
    from token_selection_entrypoint import _latest_resume_checkpoint

    def materialize(path: Path) -> None:
        (path / "model_and_optim").mkdir(parents=True)
        (path / "model_and_optim" / ".metadata").touch()

    normal = tmp_path / "step375"
    pre = tmp_path / "sync_checkpoints" / "step400-pre"
    post = tmp_path / "sync_checkpoints" / "step400-post"
    for path in (normal, pre, post):
        materialize(path)

    assert _latest_resume_checkpoint(tmp_path) == (post, True)

    caught_up = tmp_path / "step400"
    materialize(caught_up)
    assert _latest_resume_checkpoint(tmp_path) == (caught_up, False)


def test_blade_locked_schedule_and_full_resume_state() -> None:
    assert BLADE_SYNC_STEPS == (0, 400, 800, 1200, 1600, 2000)
    assert BLADE_REFERENCE_MICROBATCH_TOKENS == 8_192
    assert BLADE_SELECTION_MICROBATCH_TOKENS == 32_768
    callback = _blade_callback()
    callback._new_reference(lr=4e-4)
    assert callback.reference is not None and callback.reference_optim is not None
    callback.reference.weight.data.fill_(9)
    callback.completed_step = 1200
    callback.last_sync = 1200
    state = callback.state_dict()
    assert state["checkpoint_format"] == BLADE_CHECKPOINT_FORMAT
    assert state["version"] == 3
    assert state["dynamic_reference_optim"] is not None
    assert state["reference_train_stream"] == {"cursor": 3}
    assert state["refhq_stream"] == {"cursor": 7}

    restored = _blade_callback(0, 0)
    restored._restore(state)
    assert restored.completed_step == 1200
    assert restored.last_sync == 1200
    assert restored.reference is not None
    assert torch.equal(restored.reference.weight, torch.tensor([9.0, 9.0]))
    assert restored.reference_train_stream.cursor == 3
    assert restored.refhq_stream.cursor == 7
    assert next(step for step in BLADE_SYNC_STEPS if step > restored.completed_step) == 1600

    boundary = _blade_callback()
    boundary._new_reference(lr=4e-4)
    boundary.completed_step = 399
    boundary.last_sync = 400
    restored_boundary = _blade_callback(0, 0)
    restored_boundary._restore(boundary.state_dict())
    assert restored_boundary.completed_step == 399
    assert restored_boundary.last_sync == 400

    # Resuming from a step0-pre checkpoint: no sync has ever completed, and
    # this must be accepted (pre_train's fresh-run guard relies on it).
    fresh = _blade_callback(0, 0)
    fresh_state = fresh.state_dict()
    assert fresh_state["dynamic_reference"] is None
    restored_fresh = _blade_callback(0, 0)
    restored_fresh._restore(fresh_state)
    assert restored_fresh.completed_step == 0
    assert restored_fresh.last_sync is None
    assert restored_fresh.reference is None


def test_blade_rejects_schedule_drift_and_missing_post_first_sync_reference() -> None:
    with pytest.raises(ValueError, match="locked"):
        BladeSchedule(k_steps=74).validate(2360)
    state = _blade_callback().state_dict()
    state["completed_step"] = 400
    state["last_sync"] = 400
    with pytest.raises(ValueError, match="missing"):
        _blade_callback()._restore(state)
    source = _blade_callback()
    source._new_reference(lr=4e-4)
    inconsistent = source.state_dict()
    inconsistent["completed_step"] = 1200
    inconsistent["last_sync"] = 800
    with pytest.raises(ValueError, match="last sync"):
        _blade_callback()._restore(inconsistent)


class FakeResumableLoader:
    def __init__(self, cursor: int = 0) -> None:
        self.cursor = cursor
        self.epoch = 0

    def reshuffle(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        return self

    def __next__(self):
        value = self.cursor
        self.cursor += 1
        return {"cursor": value}

    def state_dict(self):
        return {"cursor": self.cursor, "epoch": self.epoch}

    def load_state_dict(self, state):
        self.cursor = state["cursor"]
        self.epoch = state["epoch"]


def test_blade_secondary_stream_resume_is_next_batch_exact() -> None:
    original = ResumableBatchStream(FakeResumableLoader())
    assert original.next() == {"cursor": 0}
    state = original.state_dict()
    expected = original.next()
    resumed = ResumableBatchStream(FakeResumableLoader())
    resumed.load_state_dict(state)
    assert resumed.next() == expected


def test_identity_pins_reference_provenance(tmp_path: Path) -> None:
    arm = ARM_SPECS["rho-1"]
    reference = tmp_path / "instruct-step940.pt"
    reference.write_bytes(b"immutable reference")
    corpus = types.SimpleNamespace(
        version="v1",
        paths=("/scratch/users/nzhao2/agent-runs/regmix-10b-20260725-124810/tokenized/dclm/dclm.npy",),
        dtype="uint32",
        rows=9_900_000_000,
    )
    binding = immutable_corpus_binding(arm.dataset_id, corpus)
    manifest = {"algorithm": "reference-token-ce-v1", "complete": True, "reference_sha256": "a" * 64}
    identity = scientific_identity(
        arm,
        dataset_binding=binding,
        refhq_binding=None,
        max_tokens=9_900_000_000,
        reference_path=str(reference),
        reference_scores_manifest=manifest,
    )
    assert identity["reference_contract"] == arm.reference_contract
    assert identity["reference_contract"] == INSTRUCT_REFERENCE_CONTRACT
    assert len(identity["reference_sha256"]) == 64
    assert identity["dataset_binding"] == binding
    assert len(identity["dataset_binding"]["paths_sha256"]) == 64
    assert identity["wandb_project"] == "token-selection"
    assert identity["init_seed"] == arm.init_seed
    assert identity["data_seed"] == arm.data_seed
    assert identity["git_commit"] is None
    # A1: the offline reference-scores table is pinned by contract name and by
    # a manifest digest, so a resume refuses a changed table.
    assert identity["reference_scores_contract"] == arm.reference_scores_contract
    assert identity["reference_scores_contract"] == REFERENCE_SCORES_CONTRACT
    assert len(identity["reference_scores_sha256"]) == 64

    with pytest.raises(ValueError, match="reference-scores manifest"):
        scientific_identity(
            arm,
            dataset_binding=binding,
            refhq_binding=None,
            max_tokens=9_900_000_000,
            reference_path=str(reference),
            reference_scores_manifest=None,
        )


def test_immutable_bindings_fail_closed_for_latest_and_missing_blade_refhq() -> None:
    unresolved = types.SimpleNamespace(
        version="latest",
        paths=("/scratch/example/train.bin",),
        dtype="uint32",
        rows=1,
    )
    with pytest.raises(ValueError, match="immutable version"):
        immutable_corpus_binding("pretrain/regmix-10b", unresolved)

    resolved = types.SimpleNamespace(
        version="v1",
        paths=("/scratch/example/train.bin",),
        dtype="uint32",
        rows=1,
    )
    with pytest.raises(ValueError, match="RefHQ binding"):
        scientific_identity(
            ARM_SPECS["blade"],
            dataset_binding=immutable_corpus_binding("pretrain/regmix-10b", resolved),
            refhq_binding=None,
            max_tokens=9_900_000_000,
            reference_path=None,
        )


def test_production_recipe_statically_assembles_public_olmo_apis() -> None:
    source = (EDULLM_ROOT / "token_selection_370m" / "recipe.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = {ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
    assert {
        "TransformerConfig.olmo2_370M",
        "NumpyFSLDatasetConfig",
        "NumpyDataLoaderConfig",
        "TrainerConfig",
        "CheckpointerCallback",
        "GPUMemoryMonitorCallback",
        "ConfigSaverCallback",
        "WandBCallback",
        "TaskLossEvalCallback",
        "trainer_config.build",
    } <= calls
    assert "DataParallelType.hsdp" in source
    assert "LoadStrategy.if_available if resume else LoadStrategy.never" in source
    assert 'checkpoint_kwargs["pre_train_checkpoint"] = not resume' in source
    # Every arm, including "full" and "blade", now routes through the same
    # custom module; there is no separate stock-module branch left.
    assert "_custom_module(model, module_config, selection_config)" in source
    assert "module_config.build(model)" not in source
    assert 'checkpoint_kwargs["fixed_steps"]' in source
    assert "task_loss_nproc=PRODUCTION_WORLD_SIZE if production else None" in source
    assert 'os.environ.get("EDULLM_NUM_WORKERS", "8")' in source
    assert 'os.environ.get("EDULLM_NUM_THREADS", "8")' in source
    assert 'os.environ.get("EDULLM_PREFETCH_FACTOR", "4")' in source
    assert PRODUCTION_WORLD_SIZE == 4
    assert "CUSTOM_LOSS_METHODS" not in source
    # "no RunPod path" is descriptive prose about the *absence* of one; there
    # must be no functional pointer to a runpod/ path or module.
    assert "runpod/" not in source and "runpod." not in source and "import runpod" not in source
    assert "s3://" not in source
    # E2: exact step count, not ceil(tokens/batch), with a no-wrap guard.
    assert "max_duration=Duration.steps(steps)" in source
    assert "steps > main_loader.total_batches" in source
    # E7: every ladder checkpoint stays on disk for these runs.
    assert "keep_all_checkpoints=True" in source
    # E1a: BLADE's reference LR floor needs the proxy's own warmup length.
    assert "reference_warmup_steps=WARMUP_STEPS" in source


def test_packaged_evaluator_labels_match_production_contract() -> None:
    evaluator = EDULLM_ROOT / "eval_task_loss_olmo_core.py"
    evaluator_source = evaluator.read_text(encoding="utf-8")
    tree = ast.parse(evaluator_source)
    labels = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "TASK_LABELS" for target in node.targets
        ):
            labels = ast.literal_eval(node.value)
            break
    from production_contract.task_loss import TASK_LOSS_RAW_LABELS

    assert labels == TASK_LOSS_RAW_LABELS
    assert '"--device-eval-batch-size", type=int, default=1' in evaluator_source
    # E6: both sides build the same class, so any missing/unexpected key is a
    # real bug -- no more tolerating up to 5% missing keys.
    assert 'model.load_state_dict(payload["model"], strict=True)' in evaluator_source
    assert "strict=False" not in evaluator_source
    # There is no RunPod image or Dockerfile anymore: every run's requirements
    # are installed by farmshare/setup_venv.sh directly.
    requirements = (EDULLM_ROOT / "requirements-token-selection-eval.txt").read_text(
        encoding="utf-8"
    )
    assert "transformers==4.57.6" in requirements


def test_write_identity_keeps_environment_out_of_the_resume_fingerprint(tmp_path: Path) -> None:
    """E8: torch/CUDA/driver/GPU/pip facts are recorded for the record, but
    never fold into the resume-blocking fingerprint -- a driver or dependency
    bump between a run and its resume shouldn't refuse the resume the way a
    real scientific-identity change does."""
    from production_contract.checkpoint import assert_resume_fingerprint

    save_folder = tmp_path / "save"
    progress_dir = tmp_path / "progress"
    save_folder.mkdir()
    identity = {"arm": "rho-1", "seed": 42}
    environment = {"torch_version": "2.9.0", "gpu_name": "L40S"}

    write_identity(save_folder, progress_dir, identity, environment=environment)

    fingerprint_text = (save_folder / "run_fingerprint.json").read_text(encoding="utf-8")
    assert "torch_version" not in fingerprint_text
    assert "L40S" not in fingerprint_text

    record = json.loads((progress_dir / "run_identity.json").read_text(encoding="utf-8"))
    assert record["environment"] == environment

    # A resume checks only the fingerprint; a changed environment must never
    # be able to make this raise.
    assert_resume_fingerprint(save_folder, identity)


def test_runtime_environment_reports_facts_and_tolerates_probe_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """E8: torch/CUDA/python facts are read directly; nvidia-smi and pip
    freeze are best-effort (missing on a laptop, or on a node without a GPU)
    and must never raise out of this function."""
    import subprocess

    import token_selection_entrypoint as entrypoint

    def failing_check_output(*_args, **_kwargs):
        raise FileNotFoundError("nvidia-smi not found")

    monkeypatch.setattr(subprocess, "check_output", failing_check_output)

    info = entrypoint.runtime_environment()

    assert set(info) == {
        "torch_version",
        "cuda_version",
        "python_version",
        "gpu_name",
        "nvidia_driver_version",
        "pip_freeze_sha256",
    }
    assert info["nvidia_driver_version"] is None
    assert info["pip_freeze_sha256"] is None
    assert isinstance(info["torch_version"], str)


def test_resolve_reference_scores_fails_closed_on_missing_or_incomplete_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A1: the entrypoint refuses to launch rho-1/perplexity against a table
    that isn't there, or that a crashed/killed scoring job never finished."""
    import token_selection_entrypoint as entrypoint

    incomplete_root = tmp_path / "reference-scores" / "incomplete"
    incomplete_root.mkdir(parents=True)
    reference_scores_module.write_manifest(
        incomplete_root,
        {
            "schema_version": reference_scores_module.MANIFEST_SCHEMA_VERSION,
            "algorithm": reference_scores_module.ALGORITHM,
            "complete": False,
            "reference_sha256": "deadbeef",
            "corpus_binding": {},
            "sequence_length": 4,
            "rows": 1,
            "dtype": "float32",
        },
    )
    manifest_path = tmp_path / "ready.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "family": "token-selection",
                "corpora": {},
                "reference_scores": {"present-but-incomplete": str(incomplete_root)},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("EDULLM_INPUT_MANIFEST", str(manifest_path))

    # An arm with no reference-scores contract (e.g. "attention") never
    # touches the manifest at all.
    assert entrypoint.resolve_reference_scores(None) == (None, None)

    with pytest.raises(RuntimeError, match="not complete"):
        entrypoint.resolve_reference_scores("present-but-incomplete")

    with pytest.raises(RuntimeError, match="no materialized reference-score table"):
        entrypoint.resolve_reference_scores("missing-entirely")


def test_entrypoint_builds_and_fits_production_trainer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import token_selection_entrypoint as entrypoint

    task_script = tmp_path / "task-loss.py"
    task_script.write_text("# fixture\n", encoding="utf-8")
    events: list[str] = []

    class Corpus:
        version = "immutable-v1"
        rows = 9_900_000_000
        paths = ("/scratch/users/nzhao2/agent-runs/regmix-10b-20260725-124810/tokenized/dclm/dclm.npy",)
        dtype = "uint32"

    class Trainer:
        def fit(self) -> None:
            events.append("fit")

    def fake_build(*args, **kwargs):
        assert args[0] is ARM_SPECS["attention"]
        assert kwargs["production"] is True
        assert kwargs["resume"] is False
        events.append("build")
        return Trainer()

    fake_olmo = types.ModuleType("olmo_core")
    fake_train = types.ModuleType("olmo_core.train")
    fake_utils = types.ModuleType("olmo_core.utils")
    fake_train.prepare_training_environment = lambda **kwargs: events.append("prepare")
    fake_train.teardown_training_environment = lambda: events.append("teardown")
    fake_utils.seed_all = lambda seed: events.append(f"seed:{seed}")
    monkeypatch.setitem(sys.modules, "olmo_core", fake_olmo)
    monkeypatch.setitem(sys.modules, "olmo_core.train", fake_train)
    monkeypatch.setitem(sys.modules, "olmo_core.utils", fake_utils)
    monkeypatch.setattr(entrypoint, "resolve_corpus", lambda **kwargs: Corpus())
    monkeypatch.setattr(entrypoint, "build_trainer", fake_build)
    monkeypatch.setattr(
        entrypoint, "write_identity", lambda *args, **kwargs: events.append("identity")
    )
    # E8: a real environment probe (nvidia-smi, pip freeze) is exercised by
    # runtime_environment's own unit coverage, not by this wiring test.
    monkeypatch.setattr(entrypoint, "runtime_environment", lambda: {"fake": "env"})
    monkeypatch.setattr(
        entrypoint,
        "assert_production_runtime",
        lambda expected_world_size: events.append(f"world:{expected_world_size}"),
    )
    monkeypatch.delenv("EDULLM_DATASET_VERSION", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "token_selection_entrypoint.py",
            "--arm",
            "attention",
            "--save-folder",
            str(tmp_path / "save"),
            "--work-dir",
            str(tmp_path / "work"),
            "--progress-dir",
            str(tmp_path / "progress"),
            "--task-loss-script",
            str(task_script),
        ],
    )

    entrypoint.main()

    assert events == [
        "prepare",
        "world:4",
        "seed:6198",
        "build",
        "identity",
        "fit",
        "teardown",
    ]
