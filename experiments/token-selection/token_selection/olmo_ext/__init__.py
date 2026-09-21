"""Partial vendor of the token-selection OLMo-core extensions.

Only the four modules the experiments in this repository actually import are
vendored here: the checkpoint ladder, the task-loss eval hook, the durability
marker, and the W&B task-loss logging. The upstream package also carries the
token-scoring machinery (``attention_score``, ``ema``, ``frozen_ref``,
``scorers``, ``metrics``, ``train_module``, ``refhq_materialize``), which the
token-selection experiment itself needs but nothing here does.

The copy of this file that came across with the vendor was the full upstream
one, so it re-exported all eight modules and raised ImportError on the six
that were never vendored -- which is what broke
``experiments/skill-dag/mixlaw/tests/test_mixlaw_hardening.py`` on a clean
checkout. It now re-exports only what is present.

Every import site in this repository uses the submodule form
(``from token_selection.olmo_ext.wandb_logging import ...``), so these
re-exports are for convenience only; adding a module back to the vendor does
not require touching its callers.
"""

from .checkpoint_ladder import (
    DEFAULT_CHECKPOINT_INTERVAL,
    checkpointer_kwargs_for_ladder,
    is_permanent_checkpoint_step,
    permanent_checkpoint_steps,
)
from .durability import LAST_DURABLE_STEP_FILENAME, write_last_durable_step
from .task_loss_hook import resolve_eval_script, trigger_task_loss_eval
from .wandb_logging import (
    TASK_LOSS_RAW_LABELS,
    task_loss_metrics,
    task_loss_payload_complete,
)

__all__ = [
    "DEFAULT_CHECKPOINT_INTERVAL",
    "LAST_DURABLE_STEP_FILENAME",
    "TASK_LOSS_RAW_LABELS",
    "checkpointer_kwargs_for_ladder",
    "is_permanent_checkpoint_step",
    "permanent_checkpoint_steps",
    "resolve_eval_script",
    "task_loss_metrics",
    "task_loss_payload_complete",
    "trigger_task_loss_eval",
    "write_last_durable_step",
]
