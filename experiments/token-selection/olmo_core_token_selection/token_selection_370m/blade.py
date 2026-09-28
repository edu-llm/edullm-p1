"""BLADE dynamic-reference controller integrated through public callback APIs.

Follows Wang et al. 2026 ("BLADE: Scalable Bi-Level Adaptive Data Selection for
LLM Training"), Sec. 2.2 / Algorithm 1 / App. A.1, C:

- The reference is periodically reset from the proxy (``_sync_from_proxy``) and
  then trained for K steps on H_lambda(alpha, w) = L_val(w) + lambda * L_train(alpha, w),
  where L_val is the unmasked mean CE on one RefHQ/Instruct batch and L_train is
  the alpha-weighted mean CE (selected tokens only) on one RegMix batch.
- alpha for each K-update's RegMix batch is the top-gamma selection by
  (proxy - outgoing reference) excess loss, scored *before* the reference is
  overwritten (``_score_regmix_mask``), exactly mirroring the per-step selection
  the proxy itself trains on (``pre_step`` -> ``batch["token_weight"]``).
- The very first sync (step 0, before any training batch) has no outgoing
  reference to score against, so alpha=1 (every RegMix token counts): this
  matches the paper's "warm up the proxy, then sync at t=0" schedule collapsed
  to a from-scratch run with no warmup.
- A fresh AdamW is created at every sync (no momentum carried across episodes),
  at the proxy's own scheduled LR for that step, held constant through K.
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

log = logging.getLogger(__name__)

try:  # OLMo's optional runtime dependencies are not available in pure unit-test hosts.
    from olmo_core.data.utils import get_labels, split_batch
    from olmo_core.distributed.utils import get_rank, get_world_size, is_distributed
    from olmo_core.nn.lm_head import LMOutputWithLoss
    from olmo_core.train.callbacks import Callback
except ImportError:  # pragma: no cover - production images take the branch above.
    Callback = object  # type: ignore[assignment,misc]

    class LMOutputWithLoss:  # type: ignore[no-redef]
        pass

    def get_rank() -> int:
        return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

    def get_world_size() -> int:
        return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

    def is_distributed() -> bool:
        return dist.is_available() and dist.is_initialized()

    def get_labels(batch: dict[str, Any], label_ignore_index: int = -100) -> Tensor:
        labels = batch["input_ids"].clone()
        if batch.get("label_mask") is not None:
            labels.masked_fill_(~batch["label_mask"], label_ignore_index)
        return F.pad(labels[..., 1:], (0, 1), value=label_ignore_index)

    def split_batch(batch: dict[str, Any], num_microbatch_instances: int) -> list[dict[str, Any]]:
        batch_size = batch["input_ids"].shape[0]
        return [
            {
                key: value[start : start + num_microbatch_instances]
                for key, value in batch.items()
            }
            for start in range(0, batch_size, num_microbatch_instances)
        ]


# Selection runs from step 0 (no full-loss warmup): every arm shares the same
# 60%-keep, from-step-0 budget. Six equal episodes of length tau=400 cover the
# full 2360-step budget's ladder-relevant span; K=75 matches every other arm's
# reference-update depth.
BLADE_SYNC_STEPS = (0, 400, 800, 1200, 1600, 2000)
BLADE_TAU = 400
BLADE_K = 75
BLADE_GAMMA = 0.6
BLADE_LAMBDA = 1.0
BLADE_REFERENCE_MICROBATCH_TOKENS = 8_192
BLADE_SELECTION_MICROBATCH_TOKENS = 32_768
BLADE_CHECKPOINT_FORMAT = "blade_selection_weighted_v3"


@dataclass(frozen=True)
class BladeSchedule:
    sync_steps: tuple[int, ...] = BLADE_SYNC_STEPS
    tau: int = BLADE_TAU
    k_steps: int = BLADE_K
    gamma: float = BLADE_GAMMA
    lambda_penalty: float = BLADE_LAMBDA

    def validate(self, total_steps: int) -> None:
        if self != BladeSchedule():
            raise ValueError("the approved BLADE schedule is locked; use a new run identity")
        if any(step > total_steps for step in self.sync_steps):
            raise ValueError("BLADE sync schedule exceeds the run duration")


class _ConstantSchedule:
    """Fallback LR schedule for unit tests that don't wire up the real one."""

    def get_lr(self, initial_lr: float, current: int, t_max: int) -> float:
        del current, t_max
        return initial_lr


_DEFAULT_SCHEDULE = _ConstantSchedule()


class ResumableBatchStream:
    """Infinite stream whose loader position and epoch survive callback checkpoints."""

    def __init__(self, loader, *, epoch: int = 1):
        self.loader = loader
        self.epoch = int(epoch)
        self._iterator = None

    def next(self) -> dict[str, Any]:
        while True:
            if self._iterator is None:
                self.loader.reshuffle(self.epoch)
                self._iterator = iter(self.loader)
            try:
                return next(self._iterator)
            except StopIteration:
                self.epoch += 1
                self._iterator = None

    def state_dict(self) -> dict[str, Any]:
        return {"version": 1, "epoch": self.epoch, "loader": self.loader.state_dict()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("version") != 1:
            raise ValueError("unsupported BLADE stream state")
        self.epoch = int(state["epoch"])
        self.loader.load_state_dict(state["loader"])
        self._iterator = None


def _output_ce(output: Any, labels: Tensor) -> Tensor:
    if not isinstance(output, LMOutputWithLoss):
        raise RuntimeError("BLADE requires OLMo per-token LM losses")
    return output.ce_loss.reshape_as(labels)


@contextlib.contextmanager
def _autocast(device: torch.device):
    if device.type == "cuda":
        with torch.autocast("cuda", dtype=torch.bfloat16):
            yield
    else:
        yield


def _full_proxy_state(model: nn.Module) -> dict[str, Tensor]:
    try:
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

        # This state is loaded into a replicated reference model on every rank.
        # Combining full_state_dict=True with cpu_offload=True only materializes
        # the state on rank 0 and returns an empty dict elsewhere.
        return get_model_state_dict(
            model, options=StateDictOptions(full_state_dict=True, cpu_offload=False)
        )
    except ImportError:
        return {name: value.detach().cpu() for name, value in model.state_dict().items()}


class BladeCallback(Callback):
    """Runs sync/K updates in ``post_train_batch``/``pre_train`` and checkpoints all non-proxy state.

    The proxy model and optimizer remain ordinary OLMo checkpointer state. This callback
    contributes the dynamic reference, its optimizer, both K-update streams, and schedule
    cursor to the trainer state saved in the same checkpoint.
    """

    priority = 4

    def __init__(
        self,
        *,
        total_steps: int,
        reference_factory: Callable[[], nn.Module],
        reference_train_stream: ResumableBatchStream,
        refhq_stream: ResumableBatchStream,
        schedule: BladeSchedule = BladeSchedule(),
        reference_scheduler: Any = _DEFAULT_SCHEDULE,
        reference_initial_lr: float = 4e-4,
        max_grad_norm: float = 1.0,
        reference_microbatch_tokens: int = BLADE_REFERENCE_MICROBATCH_TOKENS,
        selection_microbatch_tokens: int = BLADE_SELECTION_MICROBATCH_TOKENS,
    ) -> None:
        schedule.validate(total_steps)
        self.total_steps = int(total_steps)
        self.reference_factory = reference_factory
        self.reference_train_stream = reference_train_stream
        self.refhq_stream = refhq_stream
        self.schedule = schedule
        self.reference_scheduler = reference_scheduler
        self.reference_initial_lr = float(reference_initial_lr)
        self.max_grad_norm = float(max_grad_norm)
        self.reference_microbatch_tokens = int(reference_microbatch_tokens)
        if self.reference_microbatch_tokens <= 0:
            raise ValueError("BLADE reference microbatch tokens must be positive")
        self.selection_microbatch_tokens = int(selection_microbatch_tokens)
        if self.selection_microbatch_tokens <= 0:
            raise ValueError("BLADE selection microbatch tokens must be positive")
        self.reference: Optional[nn.Module] = None
        self.reference_optim: Optional[torch.optim.Optimizer] = None
        self.completed_step = 0
        self.last_sync: Optional[int] = None
        self._pending: Optional[Mapping[str, Any]] = None

    def _k_progress_path(self) -> Path:
        return self.trainer.work_dir / "blade_k_progress.json"

    def _write_k_progress(self, *, trainer_step: int, k_step: int) -> None:
        if get_rank() != 0:
            return
        payload = {
            "trainer_step": int(trainer_step),
            "k_step": int(k_step),
            "k_total": int(self.schedule.k_steps),
            "updated_at": time.time(),
        }
        path = self._k_progress_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        log.info(
            "BLADE reference K-update %d/%d at trainer step %d",
            k_step,
            self.schedule.k_steps,
            trainer_step,
        )

    def _clear_k_progress(self) -> None:
        if get_rank() != 0:
            return
        with contextlib.suppress(FileNotFoundError):
            self._k_progress_path().unlink()

    def _new_reference(self, *, lr: float) -> None:
        """Build a fresh reference module and a fresh AdamW at ``lr``.

        Always called at sync time (never reused across episodes), so the
        reference's optimizer state never carries momentum from a previous
        episode's K-updates. Weight-decay groups mirror the proxy's own
        (0.1 everywhere except 0.0 on ``embeddings.weight``).
        """
        self.reference = self.reference_factory()
        decay, no_decay = [], []
        for name, parameter in self.reference.named_parameters():
            (no_decay if name == "embeddings.weight" else decay).append(parameter)
        groups = [
            {"params": decay, "weight_decay": 0.1},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        self.reference_optim = torch.optim.AdamW(
            groups, lr=lr, betas=(0.9, 0.95), foreach=False
        )

    def _sync_from_proxy(self, *, lr: float) -> None:
        self._new_reference(lr=lr)
        assert self.reference is not None
        state = _full_proxy_state(self.trainer.train_module.model)
        self.reference.load_state_dict(state, strict=True)

    def _backward_mean_ce(
        self,
        model: nn.Module,
        batch: dict[str, Any],
        *,
        weight: float,
        mask: Optional[Tensor] = None,
    ) -> None:
        """Backward the mean CE over ``mask & valid`` tokens (or just ``valid`` if ``mask`` is None).

        ``mask=None`` is the unmasked RefHQ/Instruct term (L_val); a mask is the
        selection-weighted RegMix term (L_train, alpha-weighted per Wang et al.).
        """
        sequence_length = int(batch["input_ids"].shape[1])
        if self.reference_microbatch_tokens < sequence_length:
            raise RuntimeError(
                "BLADE reference microbatch token limit is smaller than sequence length"
            )
        labels = get_labels(batch)
        valid = labels != -100
        weight_tensor = (mask.to(valid.device) & valid) if mask is not None else valid
        divisor = weight_tensor.sum().clamp(min=1)
        micro_size = self.reference_microbatch_tokens // sequence_length
        micro_batches = split_batch(batch, micro_size)
        offset = 0
        for micro_batch in micro_batches:
            rows = int(micro_batch["input_ids"].shape[0])
            micro_weight = weight_tensor[offset : offset + rows]
            loss = self._mean_ce_with_weight(model, micro_batch, micro_weight, divisor)
            (float(weight) * loss).backward()
            offset += rows
            del loss

    def _mean_ce_with_weight(
        self,
        model: nn.Module,
        batch: dict[str, Any],
        weight: Tensor,
        divisor: Tensor,
    ) -> Tensor:
        ids = batch["input_ids"].to(next(model.parameters()).device)
        labels = get_labels({"input_ids": ids})
        weight = weight.to(ids.device)
        kwargs = {
            key: value.to(ids.device) if isinstance(value, Tensor) else value
            for key, value in batch.items()
            if key != "input_ids"
        }
        with _autocast(ids.device):
            output = model(
                ids,
                labels=labels,
                ignore_index=-100,
                loss_reduction="none",
                return_logits=False,
                **kwargs,
            )
        # output.ce_loss is detached by OLMo-core (logging only, see LMOutputWithLoss);
        # backward needs the live output.loss instead. z_loss_multiplier is never in
        # kwargs here, so output.loss is pure (undetached) CE, same value as ce_loss.
        ce = output.loss.reshape_as(labels)
        return (ce.float() * weight.float()).sum() / divisor.to(ce.device)

    def _run_k_updates(
        self,
        regmix_batches: list[dict[str, Any]],
        masks: list[Optional[Tensor]],
        *,
        trainer_step: int,
    ) -> None:
        assert self.reference is not None and self.reference_optim is not None
        self.reference.train()
        for parameter in self.reference.parameters():
            parameter.requires_grad_(True)
        try:
            for k_idx, (regmix_batch, mask) in enumerate(zip(regmix_batches, masks)):
                self._write_k_progress(trainer_step=trainer_step, k_step=k_idx + 1)
                self.reference_optim.zero_grad(set_to_none=True)
                self._backward_mean_ce(
                    self.reference,
                    regmix_batch,
                    weight=self.schedule.lambda_penalty,
                    mask=mask,
                )
                self._backward_mean_ce(
                    self.reference,
                    self.refhq_stream.next(),
                    weight=1.0,
                    mask=None,
                )
                if is_distributed() and get_world_size() > 1:
                    for parameter in self.reference.parameters():
                        if parameter.grad is not None:
                            dist.all_reduce(parameter.grad, op=dist.ReduceOp.AVG)
                torch.nn.utils.clip_grad_norm_(self.reference.parameters(), self.max_grad_norm)
                self.reference_optim.step()
        finally:
            self._clear_k_progress()
        self.reference.eval()
        for parameter in self.reference.parameters():
            parameter.requires_grad_(False)

    def _proxy_and_reference_ce_microbatch(
        self, batch: dict[str, Any]
    ) -> tuple[Tensor, Tensor, Tensor]:
        module = self.trainer.train_module
        ids = batch["input_ids"].to(module.device)
        labels = get_labels(batch, label_ignore_index=module.label_ignore_index).to(module.device)
        model_kwargs = {
            key: value.to(module.device) if isinstance(value, Tensor) else value
            for key, value in batch.items()
            if key not in {"input_ids", "labels", "label_mask", "instance_mask", "attention_mask"}
        }
        was_training = module.model.training
        module.model.eval()
        assert self.reference is not None
        self.reference.eval()
        with torch.no_grad(), _autocast(module.device):
            proxy = module.model_forward(
                ids,
                labels=labels,
                ignore_index=module.label_ignore_index,
                loss_reduction="none",
                return_logits=False,
                **model_kwargs,
            )
            reference = self.reference(
                ids,
                labels=labels,
                ignore_index=module.label_ignore_index,
                loss_reduction="none",
                return_logits=False,
                **model_kwargs,
            )
        module.model.train(was_training)
        module._model_mode = "train" if was_training else "eval"
        return labels, _output_ce(proxy, labels), _output_ce(reference, labels)

    def _proxy_and_reference_ce(self, batch: dict[str, Any]) -> tuple[Tensor, Tensor, Tensor]:
        sequence_length = int(batch["input_ids"].shape[1])
        if self.selection_microbatch_tokens < sequence_length:
            raise RuntimeError(
                "BLADE selection microbatch token limit is smaller than sequence length"
            )
        labels, proxy_ce, reference_ce = [], [], []
        for micro_batch in split_batch(
            batch, self.selection_microbatch_tokens // sequence_length
        ):
            micro_labels, micro_proxy_ce, micro_reference_ce = (
                self._proxy_and_reference_ce_microbatch(micro_batch)
            )
            labels.append(micro_labels)
            proxy_ce.append(micro_proxy_ce)
            reference_ce.append(micro_reference_ce)
        return (
            torch.cat(labels, dim=0),
            torch.cat(proxy_ce, dim=0),
            torch.cat(reference_ce, dim=0),
        )

    def _select_mask(self, labels: Tensor, proxy_ce: Tensor, reference_ce: Tensor) -> Tensor:
        """Exact top-``gamma`` mask by (proxy - reference) excess loss.

        Uses ``topk`` indices rather than a ``>=`` threshold, so ties never
        push the kept count above the exact budget.
        """
        valid = labels != self.trainer.train_module.label_ignore_index
        selection_scores = proxy_ce - reference_ce
        flat_scores = selection_scores[valid]
        keep = max(1, int(torch.ceil(torch.tensor(self.schedule.gamma * flat_scores.numel()))))
        keep = min(keep, flat_scores.numel())
        _, top_indices = torch.topk(flat_scores, keep)
        flat_mask = torch.zeros_like(flat_scores, dtype=torch.bool)
        flat_mask[top_indices] = True
        mask = torch.zeros_like(valid)
        mask[valid] = flat_mask
        return mask

    def _score_regmix_mask(self, batch: dict[str, Any]) -> Tensor:
        """Score one RegMix batch against the *current* (outgoing) reference."""
        labels, proxy_ce, reference_ce = self._proxy_and_reference_ce(batch)
        return self._select_mask(labels, proxy_ce, reference_ce)

    def _perform_sync(self, sync_step: int) -> None:
        """Pre-score (unless this is the first-ever sync), sync, and run K updates.

        Pre-scoring the K RegMix batches against the *outgoing* reference,
        before it is overwritten, is what makes the reference's own training
        term selection-weighted like the proxy's. At the very first sync there
        is no outgoing reference, so alpha=1 (every token counts).
        """
        first_sync = self.reference is None
        regmix_batches = [
            self.reference_train_stream.next() for _ in range(self.schedule.k_steps)
        ]
        masks: list[Optional[Tensor]] = (
            [None] * self.schedule.k_steps
            if first_sync
            else [self._score_regmix_mask(batch) for batch in regmix_batches]
        )
        lr = float(
            self.reference_scheduler.get_lr(self.reference_initial_lr, sync_step, self.total_steps)
        )
        self._sync_from_proxy(lr=lr)
        self._run_k_updates(regmix_batches, masks, trainer_step=sync_step)
        self.last_sync = sync_step

    def _sync_checkpoint_path(self, step: int, phase: str) -> str:
        return str(
            Path(self.trainer.save_folder)
            / "sync_checkpoints"
            / f"step{int(step)}-{phase}"
        )

    def _save_sync_checkpoint(self, *, step: int, phase: str) -> None:
        path = self._sync_checkpoint_path(step, phase)
        if self.trainer.checkpointer.dir_is_checkpoint(path):
            log.info("BLADE %s-sync checkpoint already exists at '%s'", phase, path)
            return
        log.info(
            "Saving BLADE %s-sync checkpoint for step %d to '%s'...",
            phase,
            step,
            path,
        )
        self.trainer._log_metrics()
        self.trainer._join_bookkeeping_ops()
        self.trainer.checkpointer.save(
            path,
            self.trainer.train_module,
            self.trainer.state_dict(),
            ephemeral=False,
        )
        for callback in self.trainer.callbacks.values():
            callback.post_checkpoint_saved(path)
        log.info("BLADE %s-sync checkpoint saved", phase)

    def pre_train(self) -> None:
        """Run the step-0 sync before the first training batch.

        ``0`` is a sync step, but it is not reachable from ``post_train_batch``
        (there is no "batch 0" whose completion could trigger it) or from
        ``pre_step`` (``trainer.global_step`` is never 0 there -- it is
        incremented to 1 before the first batch's ``pre_step`` runs). This
        fires exactly once, whether the run is fresh or is resuming from
        before the first sync ever completed (``last_sync is None and
        reference is None`` is true in both cases; a resume past that point
        restores a non-None ``last_sync`` or ``reference`` and this no-ops).
        """
        if 0 in self.schedule.sync_steps and self.last_sync is None and self.reference is None:
            self._save_sync_checkpoint(step=0, phase="pre")
            self._perform_sync(0)
            self._save_sync_checkpoint(step=0, phase="post")

    def pre_step(self, batch: dict[str, Any]) -> None:
        step = int(self.trainer.global_step)
        if step in self.schedule.sync_steps and self.last_sync != step:
            # Resume-from-pre-sync-checkpoint fallback: we crashed after saving
            # the pre-sync checkpoint (which still has last_sync at its
            # *previous* value) but before finishing this sync's K updates.
            # self.reference is already restored to the outgoing reference and
            # the streams to their pre-draw position, so redoing the sync here
            # reproduces exactly what would have happened without the crash.
            log.warning("Resuming into pending BLADE sync %d in pre_step fallback", step)
            self._perform_sync(step)
        if self.reference is None:
            raise RuntimeError(
                "BLADE selection has no dynamic reference; resume state is incomplete"
            )
        labels, proxy_ce, ref_ce = self._proxy_and_reference_ce(batch)
        mask = self._select_mask(labels, proxy_ce, ref_ce)
        batch["token_weight"] = mask.float()

    def post_train_batch(self) -> None:
        self.completed_step = int(self.trainer.global_step)
        sync_step = self.completed_step + 1
        if sync_step not in self.schedule.sync_steps or self.last_sync == sync_step:
            return
        self._save_sync_checkpoint(step=sync_step, phase="pre")
        self._perform_sync(sync_step)
        self._save_sync_checkpoint(step=sync_step, phase="post")

    def post_attach(self) -> None:
        if self._pending is not None:
            pending, self._pending = self._pending, None
            self._restore(pending)

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 3,
            "checkpoint_format": BLADE_CHECKPOINT_FORMAT,
            "completed_step": self.completed_step,
            "last_sync": self.last_sync,
            "schedule": self.schedule.__dict__,
            "dynamic_reference": (
                {name: value.detach().cpu() for name, value in self.reference.state_dict().items()}
                if self.reference is not None
                else None
            ),
            "dynamic_reference_optim": (
                self.reference_optim.state_dict() if self.reference_optim is not None else None
            ),
            "reference_train_stream": self.reference_train_stream.state_dict(),
            "refhq_stream": self.refhq_stream.state_dict(),
        }

    def _restore(self, state: Mapping[str, Any]) -> None:
        if state.get("version") != 3 or state.get("checkpoint_format") != BLADE_CHECKPOINT_FORMAT:
            raise ValueError("unsupported or incomplete BLADE checkpoint state")
        if dict(state["schedule"]) != self.schedule.__dict__:
            raise ValueError("BLADE resume schedule differs from checkpoint")
        self.completed_step = int(state["completed_step"])
        self.last_sync = state.get("last_sync")
        if not 0 <= self.completed_step <= self.total_steps:
            raise ValueError("BLADE completed step is outside the locked run")
        expected_completed_sync = next(
            (step for step in reversed(self.schedule.sync_steps) if step <= self.completed_step),
            None,
        )
        boundary_sync = (
            self.completed_step + 1
            if self.completed_step + 1 in self.schedule.sync_steps
            else None
        )
        if self.last_sync not in {expected_completed_sync, boundary_sync}:
            raise ValueError("BLADE last sync is inconsistent with the completed checkpoint step")
        reference_state = state.get("dynamic_reference")
        if reference_state is not None:
            if state.get("dynamic_reference_optim") is None:
                raise ValueError("BLADE dynamic reference is missing optimizer state")
            self._new_reference(lr=self.reference_initial_lr)
            assert self.reference is not None and self.reference_optim is not None
            self.reference.load_state_dict(reference_state, strict=True)
            self.reference_optim.load_state_dict(state["dynamic_reference_optim"])
            self.reference.eval()
            for parameter in self.reference.parameters():
                parameter.requires_grad_(False)
        elif self.completed_step > 0 or self.last_sync is not None:
            raise ValueError("post-first-sync BLADE checkpoint is missing its dynamic reference")
        self.reference_train_stream.load_state_dict(state["reference_train_stream"])
        self.refhq_stream.load_state_dict(state["refhq_stream"])

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if hasattr(self, "trainer"):
            self._restore(state)
        else:
            self._pending = state
