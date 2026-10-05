#!/usr/bin/env python3
"""Per-item task loss of finished probe checkpoints, for subset-sensitivity analysis.

``mixlaw/eval_task_loss.py`` reports only the mean bits-per-byte over a label's
items. This script runs the same evaluators (same ladder config, same batch
order) over the *whole* validation set of each curve label and keeps every
scored (doc, continuation) row with the batch it arrived in, so any subset of
items, such as the first 16 batches of 8 that the in-run evals scored, can be
re-aggregated afterwards without another forward pass.

Writes one JSON per probe: ``{label: [[doc_id, cont_id, bpb, batch_idx], ...]}``.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import timedelta
from pathlib import Path

os.environ["WANDB_DISABLED"] = "1"
os.environ["WANDB_MODE"] = "disabled"

import torch
import torch.distributed as dist
from olmo.config import EvaluatorConfig, EvaluatorType
from olmo.eval import build_evaluator
from olmo.model import OLMo
from olmo.tokenizer import Tokenizer
from olmo.torch_util import get_local_rank
from olmo.util import add_cached_path_clients, prepare_cli_environment

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "mixlaw")]
from eval_task_loss import load_compatible_train_config  # noqa: E402
from mixlaw_common import CURVE_TASK_LOSS_LABELS, patch_torch_load_for_olmo_checkpoints  # noqa: E402

log = logging.getLogger("eval_item_losses")


def item_rows(model, cfg, tokenizer, device, label: str, batch_size: int) -> list[list[float]]:
    evaluator = build_evaluator(
        cfg,
        EvaluatorConfig(
            label=label,
            type=EvaluatorType.downstream,
            device_eval_batch_size=batch_size,
            subset_num_batches=None,
        ),
        tokenizer,
        device,
    )
    evaluator.reset_metrics()
    metric = evaluator.eval_metric
    rows: list[list[float]] = []
    for batch_idx, batch in enumerate(evaluator.eval_loader):
        batch = {
            k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        before = len(metric.loglikelihoods)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            logits = model(input_ids=batch["input_ids"]).logits
        metric.update(batch, logits)
        for t in metric.loglikelihoods[before:]:
            doc_id, cont_id, bpb = (float(x) for x in t.tolist())
            rows.append([int(doc_id), int(cont_id), bpb, batch_idx])
    return rows


def main() -> None:
    patch_torch_load_for_olmo_checkpoints()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--device-eval-batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    torch.cuda.set_device(f"cuda:{get_local_rank()}")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    prepare_cli_environment()
    add_cached_path_clients()
    device = torch.device("cuda")

    cfg = load_compatible_train_config(args.checkpoint, None)
    cfg.data.num_workers = args.num_workers
    cfg.device_eval_batch_size = args.device_eval_batch_size
    cfg.evaluators = []

    model = OLMo.from_checkpoint(str(args.checkpoint), device="cuda").eval()
    tokenizer = Tokenizer.from_train_config(cfg)

    out: dict[str, object] = {"run_name": args.run_name, "checkpoint": str(args.checkpoint),
                              "device_eval_batch_size": args.device_eval_batch_size, "labels": {}}
    for label in CURVE_TASK_LOSS_LABELS:
        rows = item_rows(model, cfg, tokenizer, device, label, args.device_eval_batch_size)
        out["labels"][label] = rows
        docs = {r[0] for r in rows}
        log.info("%s %s: %d rows, %d docs", args.run_name, label, len(rows), len(docs))
        print(f"{label}\t{len(rows)} rows\t{len(docs)} docs", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
