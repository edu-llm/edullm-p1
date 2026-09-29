"""Small OLMo train-module extension for token weights and passive excess loss."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import torch
import torch.distributed as dist
from torch import Tensor

from olmo_core.data.utils import get_labels, split_batch
from olmo_core.distributed.utils import get_local_tensor, is_distributed
from olmo_core.nn.lm_head import LMOutputWithLoss
from olmo_core.optim import SkipStepOptimizer
from olmo_core.train import ReduceType
from olmo_core.train.callbacks import Callback
from olmo_core.train.train_module import TransformerTrainModule

from .reference_scores import ReferenceScoreTable
from .selection import (
    AttentionPositionBaseline,
    EMAHistory,
    aligned_log_attention_mass,
    capture_last_attention,
    ema_alpha,
    selection_weights,
)

# Methods that read a frozen reference's per-token loss (from the offline
# table, not a resident model -- see reference_scores.py).
_REFERENCE_SCORE_METHODS = frozenset({"rho_excess", "middle_ppl"})


@dataclass(frozen=True)
class TokenSelectionConfig:
    method: str
    keep_fraction: float
    total_steps: int
    seed: int = 42
    reference_scores_path: Optional[str] = None
    reference_scores_reference_sha256: Optional[str] = None
    reference_scores_corpus_binding: Optional[Mapping[str, Any]] = None
    ema_seed: Optional[str] = None
    ema_alpha: Optional[float] = None
    ema_tau: Optional[float] = None


class TokenSelectionState:
    def __init__(self, config: TokenSelectionConfig, model) -> None:
        self.config = config
        self.completed_steps = 0
        self.reference_table: Optional[ReferenceScoreTable] = None
        if config.method in _REFERENCE_SCORE_METHODS:
            if not config.reference_scores_path:
                raise ValueError(f"{config.method} requires a reference-scores table")
            self.reference_table = ReferenceScoreTable(
                config.reference_scores_path,
                expected_reference_sha256=config.reference_scores_reference_sha256,
                expected_corpus_binding=config.reference_scores_corpus_binding or {},
            )
        self.ema: Optional[EMAHistory] = None
        if config.method == "rel_ema":
            if config.ema_seed != "zero":
                raise ValueError("relative EMA must explicitly select zero initialization")
            self.ema = EMAHistory(model, seed=None)
        # attention_topk scores each token against an empirical per-position
        # baseline of the model's own attention, carried across steps (and
        # resumes) like the EMA weights above -- see AttentionPositionBaseline.
        self.attention_baseline: Optional[AttentionPositionBaseline] = None
        if config.method == "attention_topk":
            self.attention_baseline = AttentionPositionBaseline()

    def alpha(self) -> float:
        return ema_alpha(
            self.completed_steps,
            tau=self.config.ema_tau,
            constant=self.config.ema_alpha,
        )

    def after_optimizer_step(self, model) -> None:
        if self.ema is not None:
            self.ema.update(model, self.alpha())
        if self.attention_baseline is not None:
            self.attention_baseline.commit()
        self.completed_steps += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "completed_steps": self.completed_steps,
            "ema": self.ema.state_dict() if self.ema is not None else None,
            "attention_baseline": (
                self.attention_baseline.state_dict()
                if self.attention_baseline is not None
                else None
            ),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("version") != 1:
            raise ValueError("unsupported token-selection checkpoint state")
        self.completed_steps = int(state["completed_steps"])
        if self.ema is not None:
            if not isinstance(state.get("ema"), Mapping):
                raise ValueError("relative-EMA resume state is missing EMA weights")
            self.ema.load_state_dict(state["ema"])
        elif state.get("ema") is not None:
            raise ValueError("non-EMA arm received EMA resume state")
        # "attention_baseline" is an additive key: a state written before it
        # existed still loads for every arm that has no baseline, while an
        # attention arm fails closed rather than silently restarting its
        # baseline from the cold-start prior mid-run.
        baseline_state = state.get("attention_baseline")
        if self.attention_baseline is not None:
            if not isinstance(baseline_state, Mapping):
                raise ValueError("attention resume state is missing its position baseline")
            self.attention_baseline.load_state_dict(baseline_state)
        elif baseline_state is not None:
            raise ValueError("non-attention arm received attention-baseline resume state")


class TokenWeightedTrainModule(TransformerTrainModule):
    """Uses OLMo's per-token CE/Z outputs, changing only their final reduction."""

    def __init__(self, *args, selection_config: TokenSelectionConfig, **kwargs):
        super().__init__(*args, **kwargs)
        if self.tp_enabled or self.cp_enabled:
            raise ValueError(
                "token selection requires full per-token losses; TP/CP are unsupported"
            )
        self.selection_config = selection_config
        self.selection_state = TokenSelectionState(selection_config, self.model)

    @staticmethod
    def _valid(labels: Tensor) -> Tensor:
        return labels != -100

    @staticmethod
    def _loss_tensor(output: LMOutputWithLoss, labels: Tensor, name: str) -> Tensor:
        value = getattr(output, name)
        if value is None:
            raise RuntimeError(f"OLMo output did not provide required {name}")
        return get_local_tensor(value).reshape_as(labels)

    @contextlib.contextmanager
    def _score_mode(self):
        was_training = self.model.training
        old_mode = self._model_mode
        self.model.eval()
        try:
            yield
        finally:
            self.model.train(was_training)
            self._model_mode = old_mode

    def _score_many(
        self,
        shadow: EMAHistory,
        batches: Sequence[tuple[Tensor, Tensor, dict[str, Any]]],
    ) -> list[Tensor]:
        """Score every microbatch under a single EMA-shadow weight swap."""
        losses: list[Tensor] = []
        with self._score_mode(), torch.no_grad(), shadow.swap_to(self.model):
            for ids, labels, kwargs in batches:
                output = self.model_forward(
                    ids,
                    labels=labels,
                    ignore_index=self.label_ignore_index,
                    loss_reduction="none",
                    return_logits=False,
                    **kwargs,
                )
                if not isinstance(output, LMOutputWithLoss):
                    raise RuntimeError("OLMo did not return per-token scoring loss")
                losses.append(self._loss_tensor(output, labels, "ce_loss"))
                self.model.reset_auxiliary_metrics()
        return losses

    def _planned_weight(self, labels: Tensor, batch: Mapping[str, Any], *, dry_run: bool = False) -> float:
        valid = self._valid(labels)
        provided = batch.get("token_weight")
        if provided is not None:
            weight = provided.to(device=labels.device, dtype=torch.float32)
            if weight.shape != labels.shape or (weight < 0).any():
                raise ValueError("token_weight must be non-negative and match labels")
            return float((weight * valid).sum().item())
        if self.selection_config.method == "full" or (dry_run and self.selection_config.method == "blade"):
            return float(valid.sum().item())
        count = valid.sum(-1)
        keep = torch.minimum(
            torch.clamp(
                (count.float() * self.selection_config.keep_fraction).round().long(), min=1
            ),
            count,
        )
        return float(keep.sum().item())

    @staticmethod
    def _batch_index(micro: Mapping[str, Any], *, rows: int, dry_run: bool) -> Tensor:
        index = micro.get("index")
        if index is not None:
            return index
        if not dry_run:
            raise RuntimeError(
                "token-selection batch is missing its global instance index ('index')"
            )
        return torch.zeros(rows, dtype=torch.int64)

    def train_batch(self, batch: dict[str, Any], dry_run: bool = False):
        self._set_model_mode("train")
        if "labels" not in batch:
            batch["labels"] = get_labels(batch, label_ignore_index=self.label_ignore_index)
        labels = batch["labels"]
        divisor = self._planned_weight(labels, batch, dry_run=dry_run)
        if divisor <= 0:
            raise RuntimeError("batch contains no weighted target tokens")
        sequence_length = batch["input_ids"].shape[1]
        micro_batches = split_batch(batch, self.rank_microbatch_size // sequence_length)
        prepared = []
        for micro in micro_batches:
            ids, micro_labels, model_kwargs = self._prepare_batch(dict(micro))
            assert micro_labels is not None
            prepared.append(
                (
                    micro,
                    ids,
                    micro_labels,
                    model_kwargs,
                    self._valid(micro_labels).to(self.device, non_blocking=True),
                )
            )

        state = self.selection_state
        config = self.selection_config
        scoring_batches = [
            (ids, micro_labels, model_kwargs)
            for _, ids, micro_labels, model_kwargs, _ in prepared
        ]
        history_scores = (
            self._score_many(state.ema, scoring_batches) if state.ema is not None else None
        )
        reference_scores: Optional[list[Tensor]] = None
        if config.method in _REFERENCE_SCORE_METHODS:
            if state.reference_table is None:
                raise RuntimeError(f"{config.method} is missing its frozen reference-score table")
            reference_scores = [
                state.reference_table.gather(
                    self._batch_index(micro, rows=int(micro["input_ids"].shape[0]), dry_run=dry_run),
                    device=self.device,
                )
                for micro in micro_batches
            ]

        ce_batch = torch.zeros((), device=self.device)
        all_token_ce_batch = torch.zeros((), device=self.device)
        z_batch = (
            torch.zeros((), device=self.device) if self.z_loss_multiplier is not None else None
        )
        observed_weight = torch.zeros((), device=self.device)

        total_valid_tokens = max(float(self._valid(labels).sum().item()), 1.0)

        for micro_index, (micro, ids, micro_labels, model_kwargs, valid) in enumerate(prepared):
            with self._train_microbatch_context(micro_index, len(micro_batches)):
                current = attention = None
                history = history_scores[micro_index] if history_scores is not None else None
                reference = (
                    reference_scores[micro_index] if reference_scores is not None else None
                )

                capture = (
                    capture_last_attention(self.model)
                    if config.method == "attention_topk"
                    else contextlib.nullcontext()
                )
                with capture as captured:
                    output = self.model_forward(
                        ids,
                        labels=micro_labels,
                        ignore_index=self.label_ignore_index,
                        loss_reduction="none",
                        z_loss_multiplier=self.z_loss_multiplier,
                        return_logits=False,
                        **model_kwargs,
                    )
                if not isinstance(output, LMOutputWithLoss):
                    raise RuntimeError("OLMo did not return per-token train losses")
                token_loss = self._loss_tensor(output, micro_labels, "loss")
                token_ce = self._loss_tensor(output, micro_labels, "ce_loss")
                if config.method in {"rho_excess", "rel_ema"}:
                    current = token_ce.detach()
                if config.method == "attention_topk":
                    baseline = state.attention_baseline
                    if baseline is None:
                        raise RuntimeError("attention_topk is missing its position baseline")
                    log_mass = aligned_log_attention_mass(captured)
                    # Score against the baseline committed through the
                    # previous step, then record this microbatch for the
                    # next one. A dry run never feeds the baseline.
                    attention = baseline.score(log_mass)
                    if not dry_run:
                        baseline.observe(log_mass, valid)

                supplied = micro.get("token_weight")
                if supplied is not None:
                    weights = supplied.to(self.device, dtype=torch.float32) * valid
                elif dry_run and config.method == "blade":
                    # BLADE always supplies token_weight on a real step (pre_step
                    # sets it via the dynamic reference); Trainer.fit's own
                    # dry-run mock batch never calls pre_step, so nothing sets it
                    # here. selection_weights has no "blade" case (by design --
                    # see its docstring), so fall back to the same weights a
                    # "full" arm would use; the dry run never backprops anyway.
                    weights = valid.float()
                else:
                    weights = selection_weights(
                        config.method,
                        valid=valid,
                        keep_fraction=config.keep_fraction,
                        seed=config.seed,
                        index=self._batch_index(
                            micro, rows=int(ids.shape[0]), dry_run=dry_run
                        ),
                        current=current,
                        history=history,
                        reference=reference,
                        attention=attention,
                    )
                observed_weight += weights.sum()
                ce_loss = (token_ce.float() * weights).sum() / divisor
                loss = (token_loss.float() * weights).sum() / divisor
                all_token_ce_batch += (
                    (token_ce.float() * valid.float()).sum() / total_valid_tokens
                ).detach()
                if z_batch is not None:
                    token_z = self._loss_tensor(output, micro_labels, "z_loss")
                    z_loss = (token_z.float() * weights).sum() / divisor
                    z_batch += z_loss.detach()
                ce_batch += ce_loss.detach()
                loss.backward()

        observed_weight_value = float(observed_weight.item())
        if abs(observed_weight_value - divisor) > 1e-4:
            raise RuntimeError(
                f"token-weight divisor mismatch: planned {divisor}, observed {observed_weight_value}"
            )
        if not dry_run and state.attention_baseline is not None:
            # Once per step on every rank: the baseline committed in
            # after_optimizer_step is the global batch's, identical on all ranks.
            state.attention_baseline.all_reduce_pending(self.dp_process_group)
        self.model.post_batch(dry_run=dry_run)
        if dry_run:
            self.model.reset_auxiliary_metrics()
            return
        if isinstance(self.optim, SkipStepOptimizer):
            if is_distributed():
                ce_batch.div_(self._reduce_divide_factor)
                dist.all_reduce(ce_batch)
                ce_batch.div_(self.world_size)
                ce_batch.mul_(self._reduce_divide_factor)
                all_token_ce_batch.div_(self._reduce_divide_factor)
                dist.all_reduce(all_token_ce_batch)
                all_token_ce_batch.div_(self.world_size)
                all_token_ce_batch.mul_(self._reduce_divide_factor)
            self.record_ce_loss(ce_batch)
            # SkipStepAdamW's spike detector reads this every step; feed it the
            # all-token CE, the same statistic stock OLMo training uses, not the
            # kept-token CE -- a discontinuity in the kept set alone (BLADE's
            # reference resync, REL-EMA's growing alpha) shouldn't be able to
            # trigger a skip streak that doesn't reflect an actual optimization
            # problem.
            self.optim.latest_loss = all_token_ce_batch
        else:
            self.record_ce_loss(ce_batch, ReduceType.mean)
        self.record_metric(
            "CE loss (all tokens)", all_token_ce_batch, ReduceType.mean, namespace="train"
        )
        self.record_metric(
            "selected token fraction",
            observed_weight / max(float(labels.numel()), 1.0),
            ReduceType.mean,
            namespace="train",
        )
        if z_batch is not None:
            self.record_metric("Z loss", z_batch, ReduceType.mean, namespace="train")
        for name, (value, reduction) in self.model.compute_auxiliary_metrics(reset=True).items():
            self.record_metric(name, value, reduction, namespace="train")


class TokenSelectionStateCallback(Callback):
    """Advances EMA before the priority-1 checkpointer and persists resume state."""

    priority = 3

    def __init__(self) -> None:
        self._last_step: Optional[int] = None
        self._pending: Optional[Mapping[str, Any]] = None

    def post_attach(self) -> None:
        if self._pending is not None:
            self.trainer.train_module.selection_state.load_state_dict(self._pending)
            self._pending = None

    def post_train_batch(self) -> None:
        step = int(self.trainer.global_step)
        if step == self._last_step:
            return
        self.trainer.train_module.selection_state.after_optimizer_step(
            self.trainer.train_module.model
        )
        self._last_step = step

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "last_step": self._last_step,
            "selection": self.trainer.train_module.selection_state.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("version") != 1:
            raise ValueError("unsupported token-selection callback state")
        self._last_step = state.get("last_step")
        if hasattr(self, "trainer"):
            self.trainer.train_module.selection_state.load_state_dict(state["selection"])
        else:
            self._pending = state["selection"]
