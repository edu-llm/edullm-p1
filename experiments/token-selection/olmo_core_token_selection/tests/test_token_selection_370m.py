from __future__ import annotations

import ast
import contextlib
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
from token_selection_370m.arms import (  # noqa: E402
    ARM_SPECS,
    HQ_REFERENCE_CONTRACT,
    INSTRUCT_REFERENCE_CONTRACT,
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
    scientific_identity,
    total_steps,
)
from token_selection_370m.selection import (  # noqa: E402
    EMAHistory,
    WeightShadow,
    attention_received_from_qk,
    capture_last_attention,
    ema_alpha,
    normalize_and_align_attention_scores,
    selection_weights,
    uniform_attention_normalizer,
)


def test_exact_approved_arm_family_and_wandb_routing() -> None:
    assert tuple(
        (name, spec.method, spec.dataset_id, spec.keep_fraction) for name, spec in ARM_SPECS.items()
    ) == (
        ("hq-reference", "full", "pretrain/refhq-regmix-5p5b", 1.0),
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
    assert ARM_SPECS["perplexity"].reference_contract == HQ_REFERENCE_CONTRACT
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
    # The two reference arms use their whole corpus as the budget (Corpus.rows,
    # via arm.max_tokens=None), not the fixed 9.9e9 the other arms share.
    assert ARM_SPECS["hq-reference"].max_tokens is None
    assert ARM_SPECS["instruct-reference"].max_tokens is None
    assert total_steps(5_517_296_144) == 1315
    assert total_steps(3_942_810_012) == 940


def test_custom_module_backpropagates_differentiable_total_loss() -> None:
    source = (EDULLM_ROOT / "token_selection_370m" / "train_module.py").read_text(encoding="utf-8")
    assert 'token_loss = self._loss_tensor(output, micro_labels, "loss")' in source
    assert "loss = (token_loss.float() * weights).sum() / divisor" in source


def test_weight_swaps_reshard_fsdp_before_restoring_parameters() -> None:
    source = (EDULLM_ROOT / "token_selection_370m" / "selection.py").read_text(encoding="utf-8")
    assert source.count("_reshard(model)") == 2
    assert "owner.unshard()" in source
    assert "owner.reshard()" in source


def test_weight_shadow_matches_reference_outputs_and_rho_masks() -> None:
    training = Tiny()
    training.weight.data.copy_(torch.tensor([1.0, -2.0]))
    reference = Tiny()
    reference_state = {"weight": torch.tensor([3.0, 5.0], dtype=torch.float64)}
    reference.load_state_dict(reference_state)
    original = training.weight.detach().clone()
    inputs = torch.tensor([[2.0, -1.0]])
    targets = torch.tensor([[4.0, -2.0]])

    expected_reference_output = reference(inputs)
    expected_reference_loss = (expected_reference_output - targets).square()
    current_loss = (training(inputs) - targets).square()
    expected_mask = selection_weights(
        "rho_excess",
        valid=torch.ones_like(current_loss, dtype=torch.bool),
        keep_fraction=0.5,
        step=0,
        seed=42,
        current=current_loss,
        reference=expected_reference_loss,
    )

    shadow = WeightShadow.from_state_dict(training, reference_state)
    assert torch.equal(training.weight, original)
    assert shadow.weights["weight"].shape == training.weight.shape
    assert shadow.weights["weight"].device == training.weight.device
    assert shadow.weights["weight"].dtype == training.weight.dtype
    with torch.no_grad(), shadow.swap_to(training):
        actual_reference_output = training(inputs)
    actual_reference_loss = (actual_reference_output - targets).square()
    actual_mask = selection_weights(
        "rho_excess",
        valid=torch.ones_like(current_loss, dtype=torch.bool),
        keep_fraction=0.5,
        step=0,
        seed=42,
        current=current_loss,
        reference=actual_reference_loss,
    )

    assert torch.equal(actual_reference_output, expected_reference_output)
    assert torch.equal(actual_reference_loss, expected_reference_loss)
    assert torch.equal(actual_mask, expected_mask)
    assert torch.equal(training.weight, original)


def test_weight_shadow_repeatedly_restores_training_parameters_and_gradients() -> None:
    training = Tiny()
    training.weight.data.copy_(torch.tensor([1.25, -2.5]))
    original = training.weight.detach().clone()
    shadow = WeightShadow.from_state_dict(
        training,
        {"weight": torch.tensor([3.0, 5.0])},
    )

    for _ in range(5):
        with torch.no_grad(), shadow.swap_to(training):
            assert torch.equal(training.weight, torch.tensor([3.0, 5.0]))
        assert torch.equal(training.weight, original)

    with pytest.raises(RuntimeError, match="scoring failed"):
        with shadow.swap_to(training):
            raise RuntimeError("scoring failed")
    assert torch.equal(training.weight, original)

    training(torch.tensor([[2.0, -4.0]])).sum().backward()
    assert torch.equal(training.weight.grad, torch.tensor([2.0, -4.0]))
    assert all(
        not weight.requires_grad and weight.grad is None for weight in shadow.weights.values()
    )
    assert torch.equal(training.weight, original)


def test_weight_shadow_hot_path_uses_only_local_copies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training = Tiny()
    training.weight.data.copy_(torch.tensor([1.0, 2.0]))
    writes = 0
    original_write = selection_module._write

    def count_write(parameter, value):
        nonlocal writes
        writes += 1
        original_write(parameter, value)

    monkeypatch.setattr(selection_module, "_write", count_write)
    shadow = WeightShadow.from_state_dict(
        training,
        {"weight": torch.tensor([3.0, 5.0])},
    )
    assert writes == 1

    def forbidden(*_args, **_kwargs):
        raise AssertionError("hot path attempted CPU or full-tensor materialization")

    monkeypatch.setattr(selection_module, "_write", forbidden)
    monkeypatch.setattr(selection_module, "_snapshot", forbidden)
    reference_storage = shadow.weights["weight"].data_ptr()
    for _ in range(3):
        with shadow.swap_to(training):
            assert torch.equal(training.weight, torch.tensor([3.0, 5.0]))
        assert torch.equal(training.weight, torch.tensor([1.0, 2.0]))
        assert shadow.weights["weight"].data_ptr() == reference_storage


def test_reference_microbatches_share_one_weight_swap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training = Tiny()
    training.weight.data.copy_(torch.tensor([1.0, 2.0]))
    shadow = WeightShadow.from_state_dict(
        training,
        {"weight": torch.tensor([3.0, 5.0])},
    )
    original_swap = shadow.swap_to
    swaps = 0

    @contextlib.contextmanager
    def counted_swap(model):
        nonlocal swaps
        swaps += 1
        with original_swap(model) as swapped:
            yield swapped

    monkeypatch.setattr(shadow, "swap_to", counted_swap)
    inputs = [
        torch.tensor([[1.0, 2.0]]),
        torch.tensor([[4.0, 7.0]]),
        torch.tensor([[3.0, 6.0]]),
    ]
    with torch.no_grad(), shadow.swap_to(training):
        scores = [training(value) for value in inputs]

    assert swaps == 1
    assert torch.equal(scores[0], torch.tensor([[3.0, 10.0]]))
    assert torch.equal(scores[1], torch.tensor([[12.0, 35.0]]))
    assert torch.equal(scores[2], torch.tensor([[9.0, 30.0]]))
    assert torch.equal(training.weight, torch.tensor([1.0, 2.0]))
    assert training.training


def test_train_module_batches_reference_scoring_structurally() -> None:
    source = (EDULLM_ROOT / "token_selection_370m" / "train_module.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    score_many = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_score_many"
    )
    swaps = [
        node
        for node in ast.walk(score_many)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "swap_to"
    ]
    assert len(swaps) == 1
    assert any(
        isinstance(child, ast.For)
        for with_node in ast.walk(score_many)
        if isinstance(with_node, ast.With)
        for statement in with_node.body
        for child in ast.walk(statement)
    )
    assert "self._score_many(state.reference, scoring_batches)" in source
    assert "self._score(state.reference" not in source


def test_weight_accounting_synchronizes_once_per_batch() -> None:
    source = (EDULLM_ROOT / "token_selection_370m" / "train_module.py").read_text(encoding="utf-8")
    assert "observed_weight += weights.sum()" in source
    assert source.count("observed_weight.item()") == 1


def test_weight_shadow_restores_local_fsdp_parameter_after_forward(tmp_path: Path) -> None:
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
        shadow = WeightShadow.from_state_dict(training, reference_state)
        with torch.no_grad(), shadow.swap_to(training):
            actual = training(inputs)

        assert torch.equal(actual, expected)
        assert torch.equal(training.weight.to_local(), original_shard)
        assert shadow.weights["weight"].device == training.weight.to_local().device
        assert shadow.weights["weight"].shape == training.weight.to_local().shape

        ema = EMAHistory(training, seed=reference_state)
        with torch.no_grad(), ema.swap_to(training):
            actual_ema = training(inputs)
        assert torch.equal(actual_ema, expected)
        assert torch.equal(training.weight.to_local(), original_shard)
    finally:
        dist.destroy_process_group()


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
    current = torch.tensor([[4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0]])
    reference = torch.ones_like(current)
    rho = selection_weights(
        "rho_excess",
        valid=valid,
        keep_fraction=0.5,
        step=0,
        seed=42,
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
        step=0,
        seed=42,
        current=current,
        history=reference * 3,
    )
    assert rel.sum(dim=1).tolist() == [2, 2]
    # "blade" is deliberately not a selection_weights method: BLADE computes
    # its own weights in BladeCallback and supplies them via
    # batch["token_weight"], read before selection_weights() is ever called.
    with pytest.raises(ValueError, match="unsupported"):
        selection_weights(
            "blade", valid=valid, keep_fraction=0.5, step=500, seed=42, current=current,
            reference=reference * 3,
        )


def test_middle_ppl_drops_easy_and_hard_and_random_is_resumable() -> None:
    valid = torch.ones(1, 10, dtype=torch.bool)
    middle = selection_weights(
        "middle_ppl",
        valid=valid,
        keep_fraction=0.6,
        step=0,
        seed=42,
        reference=torch.arange(10.0).unsqueeze(0),
    )
    assert middle.bool().tolist() == [
        [False, False, True, True, True, True, True, True, False, False]
    ]
    first = selection_weights("random", valid=valid, keep_fraction=0.6, step=125, seed=42)
    resumed = selection_weights("random", valid=valid, keep_fraction=0.6, step=125, seed=42)
    assert torch.equal(first, resumed)


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


def test_normalize_and_align_gives_all_ones_under_uniform_attention() -> None:
    query = torch.zeros(1, 5, 1, 2)
    key = torch.zeros_like(query)
    raw = attention_received_from_qk(query, key)
    normalized = raw / uniform_attention_normalizer(5, raw.device, raw.dtype)
    assert normalized[0].tolist() == pytest.approx([1.0] * 5)


def test_normalize_and_align_shifts_a_spike_to_the_label_it_gates() -> None:
    # A raw score with all its mass on position k=3 must, after the +1 target
    # shift, select the *label* index k-1=2 (label t predicts input t+1).
    raw = torch.zeros(1, 5)
    raw[0, 3] = 1000.0
    aligned = normalize_and_align_attention_scores(raw)
    assert aligned.shape == (1, 5)
    assert int(aligned[0].argmax()) == 2
    # The last column has no label under get_labels' left shift and must never
    # be selectable.
    assert aligned[0, -1] == -torch.inf


def test_normalize_and_align_is_position_invariant_up_to_the_shift() -> None:
    # A perfectly uniform raw score, once normalized, is flat; after the
    # alignment shift it stays flat except for the -inf sentinel column.
    length = 8
    normalizer = uniform_attention_normalizer(length, torch.device("cpu"))
    raw = normalizer.unsqueeze(0).clone()  # a raw score that is exactly uniform
    aligned = normalize_and_align_attention_scores(raw)
    assert aligned[0, :-1].tolist() == pytest.approx([1.0] * (length - 1))
    assert aligned[0, -1] == -torch.inf


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
        batch = {"input_ids": torch.full((1, 2), self.cursor, dtype=torch.long)}
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


def test_blade_select_mask_is_exact_top_k_even_with_ties() -> None:
    callback = _blade_callback()
    callback.trainer = types.SimpleNamespace(
        train_module=types.SimpleNamespace(label_ignore_index=-100)
    )
    labels = torch.tensor([[10, 11, 12, 13, 14]])
    proxy_ce = torch.tensor([[5.0, 5.0, 5.0, 5.0, 1.0]])
    reference_ce = torch.zeros_like(proxy_ce)  # every excess score ties at 5.0 except one

    mask = callback._select_mask(labels, proxy_ce, reference_ce)

    # gamma=0.6 of 5 valid tokens -> ceil(3.0) = 3, exactly, despite the tie.
    assert int(mask.sum()) == 3


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
    batch = {"input_ids": labels.clone()}

    callback.pre_step(batch)

    # ceil(0.6*4) = 3 tokens, by proxy-reference (4-3=1, 3-3=0, 2-3=-1, 1-3=-2):
    # positions 0,1,2 beat position 3. Written as a weight, never touching
    # "labels".
    assert "labels" not in batch
    assert batch["token_weight"].bool().tolist() == [[True, True, True, False]]


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

    callback._perform_sync(0)

    # No outgoing reference exists yet, so no batch is scored: every K-update
    # uses alpha=1 (every RegMix token counts), matching the paper's
    # from-scratch-with-no-warmup first episode.
    assert scored == []
    assert sync_calls == [4e-4]  # the default (constant) test schedule
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

    callback.pre_step({"input_ids": torch.tensor([[1]])})

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
    identity = scientific_identity(
        arm,
        dataset_binding=binding,
        refhq_binding=None,
        max_tokens=9_900_000_000,
        reference_path=str(reference),
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
    # There is no RunPod image or Dockerfile anymore: every run's requirements
    # are installed by farmshare/setup_venv.sh directly.
    requirements = (EDULLM_ROOT / "requirements-token-selection-eval.txt").read_text(
        encoding="utf-8"
    )
    assert "transformers==4.57.6" in requirements


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
    monkeypatch.setattr(entrypoint, "write_identity", lambda *args: events.append("identity"))
    monkeypatch.setattr(
        entrypoint,
        "assert_production_runtime",
        lambda expected_world_size: events.append(f"world:{expected_world_size}"),
    )
    monkeypatch.setenv("EDULLM_DATASET_VERSION", "v1")
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
