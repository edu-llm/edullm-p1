#!/usr/bin/env python3
"""Train one DataDecide-60M domain-mixture run on one GPU, scoring every test item.

One entrypoint for all 31 runs (24 MixLaw pilots + 7 one-hot Skill-It probes) listed in
``runs.json``::

    python train_60m.py --run mix07 --data-root <.../tokens> --out-dir <dir>

Design, all of it fixed by ``runs.json`` so every run uses the same code path:

* **Model.** DataDecide-60M (d_model 384, 16 layers, 12 heads, MLP ratio 8, seq 2048, global
  batch 96, LR 5.8e-3, untied head) with the embedding widened to the dolma2 vocabulary.
* **Length.** 1440 optimizer steps (283,115,520 tokens).
* **Schedule.** Linear warmup over the first 10% of steps, constant peak, cosine to 10% of
  peak over the last 10% (144 / 1152 / 144 steps).
* **Data.** The published ``pretrain/olmo-127b`` v1 train shards, all of them, straight from
  ``--data-root``. Each run's sequences are fixed up front: exact largest-remainder domain
  counts for the mixture weights, chunks drawn *without replacement* uniformly over each
  domain, order shuffled. The plan is saved next to the run.
* **Evals.** The six ARC/MMLU test labels, every item, every 120 steps (12 points, the last at
  the final step). The curve goes to ``task_loss.jsonl``; the final unsharded checkpoint is
  kept so any later evaluation can be run from it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import socket
import sys
from dataclasses import dataclass
from math import cos, pi
from pathlib import Path
from typing import Any, Optional

import numpy as np

HERE = Path(__file__).resolve().parent
RUNS_JSON = HERE / "runs.json"
MANIFEST = HERE.parents[2] / "datasets" / "manifests" / "olmo-127b-v1" / "outputs.json"

# --- DataDecide 60M (allenai/DataDecide-dolma1_7-60M config + Appendix Table 2) ---------
D_MODEL = 384
N_HEADS = 12
N_LAYERS = 16
MLP_RATIO = 8
SEQ_LEN = 2048
GLOBAL_BATCH_SEQS = 96
LEARNING_RATE = 5.8e-3
TOKENS_PER_STEP = GLOBAL_BATCH_SEQS * SEQ_LEN
WARMUP_FRACTION = 0.10
DECAY_FRACTION = 0.10
ALPHA_F = 0.1  # final LR as a fraction of peak

# dolma2 vocabulary; the corpus is already tokenized with it.
TOKENIZER_ID = "allenai/dolma2-tokenizer"
VOCAB_SIZE = 100_278
EMBEDDING_SIZE = 100_352
EOS_TOKEN_ID = 100_257
PAD_TOKEN_ID = 100_277

BYTES_PER_TOKEN = 4
log = logging.getLogger("train_60m")


# --- run registry ----------------------------------------------------------------------
def load_registry(path: Path = RUNS_JSON) -> dict:
    reg = json.loads(path.read_text(encoding="utf-8"))
    domains = reg["domains"]
    for run in reg["runs"]:
        w = run["weights"]
        if len(w) != len(domains) or min(w) < 0 or abs(sum(w) - 1.0) > 1e-3:
            raise SystemExit(f"run {run['name']}: weights must be a simplex point over {domains}")
    names = [r["name"] for r in reg["runs"]]
    if len(set(names)) != len(names):
        raise SystemExit("duplicate run names in runs.json")
    return reg


def get_run(reg: dict, name: str) -> dict:
    for run in reg["runs"]:
        if run["name"] == name:
            return run
    raise SystemExit(f"unknown run {name!r}; runs.json has {[r['name'] for r in reg['runs']]}")


# --- schedule --------------------------------------------------------------------------
def schedule_steps(total_steps: int) -> tuple[int, int]:
    """(warmup steps, decay steps) as fixed fractions of the run."""
    return max(1, round(WARMUP_FRACTION * total_steps)), max(1, round(DECAY_FRACTION * total_steps))


def lr_at(step: int, total_steps: int, peak: float = LEARNING_RATE) -> float:
    """Warmup -> constant peak -> cosine over the last ``decay`` steps to ``ALPHA_F * peak``."""
    warmup, decay = schedule_steps(total_steps)
    eta_min = peak * ALPHA_F
    if step < warmup:
        return peak * step / warmup
    decay_start = max(warmup, total_steps - decay)
    if step < decay_start:
        return peak
    if step >= total_steps or total_steps <= decay_start:
        return eta_min
    progress = (step - decay_start) / (total_steps - decay_start)
    return eta_min + (peak - eta_min) * (1 + cos(pi * progress)) / 2


# --- data ------------------------------------------------------------------------------
def shard_table(data_root: Path, manifest: dict, domains: list[str]) -> dict[str, list[tuple[str, int]]]:
    """Per domain: [(shard path, tokens)] for every published train shard, checked on disk.

    The shard list and byte sizes come from the corpus manifest; a missing or truncated
    shard stops the run before any compute is spent.
    """
    expected: dict[str, list[dict]] = {d: [] for d in domains}
    for obj in manifest["objects"]:
        name = obj["name"]  # tokens/<source>/train-00000.u32le.bin
        if "/train-" in name and obj["source"] in expected:
            expected[obj["source"]].append(obj)
    table: dict[str, list[tuple[str, int]]] = {}
    for d in domains:
        objs = sorted(expected[d], key=lambda o: o["name"])
        if not objs:
            raise SystemExit(f"manifest lists no train shards for {d}")
        shards = []
        for obj in objs:
            path = data_root / d / Path(obj["name"]).name
            if not path.is_file() or path.stat().st_size != obj["bytes"]:
                raise SystemExit(f"{path}: missing or size != manifest {obj['bytes']}")
            # the last shard of a domain is not a whole number of chunks; its tail is unused
            shards.append((str(path), int(obj["tokens"])))
        table[d] = shards
    return table


def allocate_sequences(weights: list[float], total_seqs: int) -> list[int]:
    """Largest-remainder split of ``total_seqs``; a zero-weight domain always gets 0."""
    raw = [w * total_seqs for w in weights]
    counts = [int(r) for r in raw]
    eligible = [i for i, w in enumerate(weights) if w > 0]
    order = sorted(eligible, key=lambda i: (raw[i] - counts[i], raw[i]), reverse=True)
    for k in range(total_seqs - sum(counts)):
        counts[order[k % len(order)]] += 1
    return counts


def build_plan(chunks_per_domain: list[int], weights: list[float], total_seqs: int, seed: int) -> np.ndarray:
    """``[total_seqs, 2]`` of (domain index, chunk index within the domain).

    Chunks are drawn without replacement, uniformly over each domain's whole stream, so no
    sequence repeats within a run and no part of a domain is favored. Row order is shuffled.
    """
    counts = allocate_sequences(weights, total_seqs)
    rows = []
    for di, (n_chunks, c) in enumerate(zip(chunks_per_domain, counts)):
        if c == 0:
            continue
        if c > n_chunks:
            raise SystemExit(f"domain {di} needs {c} chunks but has {n_chunks}")
        rng = np.random.default_rng([seed, di])
        chunk_ids = rng.choice(n_chunks, size=c, replace=False)
        rows.append(np.stack([np.full(c, di, dtype=np.int64), chunk_ids.astype(np.int64)], axis=1))
    plan = np.concatenate(rows, axis=0)
    order = np.random.default_rng([seed, 10_000]).permutation(len(plan))
    return plan[order]


class PlannedDataset:
    """OLMo-compatible dataset: item ``i`` is the planned (domain, chunk) sequence."""

    def __init__(self, plan: np.ndarray, domains: list[str], shards: dict[str, list[tuple[str, int]]]) -> None:
        self.plan = plan
        self.domains = domains
        self.shards = shards
        self._cum = {d: np.cumsum([t // SEQ_LEN for _, t in shards[d]]) for d in domains}
        self._mm: dict[str, np.memmap] = {}

    def __len__(self) -> int:
        return len(self.plan)

    def locate(self, domain: str, chunk: int) -> tuple[str, int]:
        cum = self._cum[domain]
        s = int(np.searchsorted(cum, chunk, side="right"))
        local = chunk - (int(cum[s - 1]) if s > 0 else 0)
        return self.shards[domain][s][0], local * SEQ_LEN

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        di, chunk = self.plan[int(index)]
        path, start = self.locate(self.domains[int(di)], int(chunk))
        mm = self._mm.get(path)
        if mm is None:
            mm = self._mm[path] = np.memmap(path, mode="r", dtype="<u4")
        return {"input_ids": torch.from_numpy(np.asarray(mm[start : start + SEQ_LEN], dtype=np.int64))}

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state["_mm"] = {}  # memmaps are reopened in each dataloader worker
        return state


# --- trainer log -> task_loss.jsonl -----------------------------------------------------
_STEP_RE = re.compile(r"\bstep=(\d+)")


def parse_trainer_log_record(
    msg: str, labels: set[str], last_step: Optional[int] = None
) -> tuple[Optional[int], Optional[float], dict[str, float]]:
    """(step, train loss, {label: task-loss bpb}) from one trainer log record.

    ai2-olmo 0.6.0 writes the step (``[step=N/T,epoch=E]``) and each evaluator's metric
    (``eval/downstream_bpb/<label>_bpb=<v>``) as separate records, so a record without
    ``step=`` takes ``last_step``: the eval for step N runs right after the step-N record.
    """
    match = _STEP_RE.search(msg)
    step = int(match.group(1)) if match else last_step
    train_loss: Optional[float] = None
    task_losses: dict[str, float] = {}
    for token in msg.replace(",", " ").split():
        key, sep, raw = token.partition("=")
        if not sep:
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        if key in ("train/CrossEntropyLoss", "loss"):
            train_loss = value
        elif key.startswith("eval/"):
            tail = key.rsplit("/", 1)[-1]
            # every label already ends in ``_bpb`` and the metric type adds another
            for cand in (tail.removesuffix("_bpb"), tail):
                if cand in labels:
                    task_losses[cand] = value
                    break
    return step, train_loss, task_losses


class CurveHandler(logging.Handler):
    """Append in-run task losses to ``task_loss.jsonl`` and mirror them to W&B."""

    def __init__(self, out_dir: Path, labels: list[str], total_steps: int, wb_run: Any = None) -> None:
        super().__init__()
        self.labels = set(labels)
        self.total_steps = total_steps
        self.curve_path = out_dir / "task_loss.jsonl"
        self.progress_path = out_dir / "progress.json"
        self.wb_run = wb_run
        self._last_step: Optional[int] = None
        self._curve: dict[int, dict[str, float]] = {}
        if self.curve_path.exists():
            for line in self.curve_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self._curve.setdefault(int(row["step"]), {}).update(row["task_loss_bpb"])

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
            if "step=" not in msg and "eval/" not in msg:
                return
            step, train_loss, task = parse_trainer_log_record(msg, self.labels, self._last_step)
            if step is None:
                return
            self._last_step = step
            if task:
                self._curve.setdefault(step, {}).update(task)
                self.curve_path.write_text(
                    "".join(
                        json.dumps({"step": s, "task_loss_bpb": row}) + "\n"
                        for s, row in sorted(self._curve.items())
                    ),
                    encoding="utf-8",
                )
                if self.wb_run is not None:
                    row = {f"eval/{k.removesuffix('_rc_5shot_bpb')}": v for k, v in task.items()}
                    if len(self._curve[step]) == len(self.labels):
                        row["eval/macro_bpb"] = sum(self._curve[step].values()) / len(self.labels)
                    self.wb_run.log(row, step=step)
            if train_loss is not None and self.wb_run is not None:
                self.wb_run.log({"train/loss": train_loss}, step=step)
            self.progress_path.write_text(
                json.dumps({"step": step, "total_steps": self.total_steps, "train_loss": train_loss}) + "\n",
                encoding="utf-8",
            )
        except Exception:  # a logging handler must never break training
            return


# --- OLMo config -----------------------------------------------------------------------
def build_train_config(args: argparse.Namespace, reg: dict, run: dict, out_dir: Path):
    from olmo import (
        ActivationType,
        DDPConfig,
        InitFnType,
        LayerNormType,
        ModelConfig,
        OptimizerConfig,
        OptimizerType,
        SchedulerConfig,
        SchedulerType,
        TokenizerConfig,
        TrainConfig,
    )
    from olmo.config import (
        DataConfig,
        DistributedStrategy,
        EvaluatorConfig,
        EvaluatorType,
        InstanceFilterConfig,
        ShardedCheckpointerType,
        SpeedMonitorConfig,
    )
    from olmo.util import find_latest_checkpoint

    st = reg["settings"]
    total_steps = int(st["steps"])
    warmup, _ = schedule_steps(total_steps)
    if GLOBAL_BATCH_SEQS % args.microbatch:
        raise SystemExit(f"global batch {GLOBAL_BATCH_SEQS} not divisible by microbatch {args.microbatch}")
    never_subset = 10**9  # ``None`` breaks olmo.train.eval(); this never binds

    model = ModelConfig(
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, mlp_ratio=MLP_RATIO, weight_tying=False,
        alibi=False, alibi_bias_max=8.0, rope=True, rope_full_precision=True, rope_theta=10_000,
        flash_attention=os.environ.get("OLMO_FLASH_ATTENTION", "0") == "1",
        attention_dropout=0.0, attention_layer_norm=False, attention_layer_norm_with_affine=False,
        include_bias=False, layer_norm_type=LayerNormType.rms, layer_norm_with_affine=True,
        layer_norm_eps=1e-6, bias_for_layer_norm=False, activation_type=ActivationType.swiglu,
        residual_dropout=0.0, embedding_dropout=0.0, max_sequence_length=SEQ_LEN,
        vocab_size=VOCAB_SIZE, embedding_size=EMBEDDING_SIZE, eos_token_id=EOS_TOKEN_ID,
        pad_token_id=PAD_TOKEN_ID, init_device="cuda", init_fn=InitFnType.normal, init_std=0.02,
        init_cutoff_factor=3, norm_after=False, precision="amp_bf16",
    )
    save_folder = out_dir / "checkpoints"
    return TrainConfig(
        run_name=run["name"],
        seed=int(st["model_seed"]),
        wandb=None,
        model=model,
        ddp=DDPConfig(),
        optimizer=OptimizerConfig(
            name=OptimizerType.adamw, learning_rate=LEARNING_RATE, weight_decay=0.1, eps=1e-8,
            decay_norm_and_bias=True, decay_embeddings=False, betas=(0.9, 0.95), metrics_log_interval=10,
        ),
        scheduler=SchedulerConfig(
            name=SchedulerType.cosine_with_warmup, alpha_f=ALPHA_F, warmup_min_lr=0.0,
            t_warmup=warmup, t_max=total_steps,
        ),
        max_duration=f"{total_steps * TOKENS_PER_STEP}T",
        global_train_batch_size=GLOBAL_BATCH_SEQS,
        tokenizer=TokenizerConfig(identifier=TOKENIZER_ID),
        save_folder=str(save_folder),
        save_overwrite=True,
        save_interval_unsharded=total_steps,
        save_num_unsharded_checkpoints_to_keep=1,
        save_interval=None,
        load_path=find_latest_checkpoint(str(save_folder)),
        try_load_latest_save=True,
        eval_on_load=False,
        sharded_checkpointer=ShardedCheckpointerType.olmo_core,
        device_train_microbatch_size=args.microbatch,
        precision="amp_bf16",
        distributed_strategy=DistributedStrategy.ddp,  # one GPU: DDP is a no-op wrapper
        fused_loss=os.environ.get("OLMO_FUSED_LOSS", "0") == "1",
        gen1_gc_interval=2,
        max_grad_norm=1.0,
        speed_monitor=SpeedMonitorConfig(window_size=1),
        eval_interval=int(st["eval_interval"]),
        device_eval_batch_size=args.eval_batch_size,
        eval_subset_num_batches=never_subset,
        evaluators=[
            EvaluatorConfig(label=lb, type=EvaluatorType.downstream, subset_num_batches=never_subset)
            for lb in st["eval_labels"]
        ],
        data=DataConfig(
            num_workers=args.num_workers, drop_last=True, pin_memory=True, prefetch_factor=4,
            persistent_workers=True, instance_filter=InstanceFilterConfig(),
            paths=["<planned_dataset>"], memmap_dtype="uint32", seed=int(st["data_seed"]),
        ),
        save_data_indices=False,
        no_pre_train_checkpoint=True,
        auxiliary_loss_multiplier=1e-5,
        softmax_auxiliary_loss=True,
    )


def install_dataset(dataset: PlannedDataset) -> None:
    """Make OLMo's train dataloader read ``dataset``."""

    def _build(_cfg: Any, _data_cfg: Any, **_kwargs: Any) -> PlannedDataset:  # 0.6.0 passes extras
        return dataset

    import olmo.data as od

    od.build_memmap_dataset = _build  # type: ignore[assignment]
    try:
        import olmo.data.memmap_dataset as mmd

        mmd.build_memmap_dataset = _build  # type: ignore[assignment]
    except Exception:
        pass


def make_scheduler(cfg):
    from olmo.optim import Scheduler

    @dataclass
    class WarmupConstantCosine(Scheduler):
        warmup_steps: int = 0
        decay_steps: int = 0

        def get_lr(self, initial_lr: float, step: int, max_steps: int) -> float:
            return lr_at(step, max_steps, initial_lr)

    sc = cfg.scheduler
    return WarmupConstantCosine(
        grad_clip_warmup_steps=None if sc.grad_clip_warmup_steps is None else int(sc.grad_clip_warmup_steps),
        grad_clip_warmup_factor=sc.grad_clip_warmup_factor,
        warmup_min_lr=sc.warmup_min_lr,
        warmup_steps=int(sc.t_warmup),
        decay_steps=schedule_steps(int(sc.t_max))[1],
    )


def init_wandb(args: argparse.Namespace, meta: dict, out_dir: Path):
    """Rank-0 W&B run in project ``domain-probes``; a failure here never stops training."""
    if args.wandb_mode == "disabled" or not os.environ.get("WANDB_API_KEY"):
        return None
    try:
        import secrets

        import wandb

        id_file = out_dir / "wandb_run_id.txt"
        # wandb.util.generate_id was removed from newer wandb releases, so make the id here.
        run_id = id_file.read_text().strip() if id_file.exists() else secrets.token_hex(4)
        id_file.write_text(run_id)
        (out_dir / "wandb").mkdir(exist_ok=True)
        return wandb.init(
            project=args.wandb_project, entity=os.environ.get("WANDB_ENTITY", "eduLLM"),
            group=meta["kind"], name=meta["run"], id=run_id, resume="allow",
            dir=str(out_dir / "wandb"), config=meta, mode=args.wandb_mode,
            tags=["domain-probes", "datadecide-60m", meta["kind"]],
        )
    except Exception as exc:  # pragma: no cover
        log.warning("W&B disabled: %s", exc)
        return None


# --- main ------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run name from runs.json, e.g. mix07 or onehot_dclm")
    ap.add_argument("--data-root", type=Path, required=True, help="dir holding <domain>/train-*.u32le.bin")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--microbatch", type=int, default=4, help="must divide the global batch of 96")
    ap.add_argument("--eval-batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--steps", type=int, default=None, help="override runs.json steps (smoke tests only)")
    ap.add_argument("--eval-interval", type=int, default=None, help="override runs.json eval interval (smoke only)")
    ap.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    ap.add_argument("--wandb-project", default="domain-probes")
    args = ap.parse_args()

    reg = load_registry()
    if args.steps is not None:
        reg["settings"]["steps"] = int(args.steps)
    if args.eval_interval is not None:
        reg["settings"]["eval_interval"] = int(args.eval_interval)
    run = get_run(reg, args.run)
    st = reg["settings"]
    domains = reg["domains"]
    total_steps = int(st["steps"])
    total_seqs = total_steps * GLOBAL_BATCH_SEQS
    out_dir = args.out_dir / run["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    shards = shard_table(args.data_root, manifest, domains)
    chunks = [sum(t // SEQ_LEN for _, t in shards[d]) for d in domains]
    plan = build_plan(chunks, run["weights"], total_seqs, int(st["data_seed"]))
    plan_path = out_dir / "data_plan.npy"
    np.save(plan_path, plan)
    realized = np.bincount(plan[:, 0], minlength=len(domains)) / len(plan)

    import torch

    meta = {
        "run": run["name"], "kind": run["kind"], "weights": run["weights"], "domains": domains,
        "realized_weights": realized.tolist(), "total_steps": total_steps, "total_seqs": total_seqs,
        "tokens": total_steps * TOKENS_PER_STEP, "schedule": dict(zip(("warmup", "decay"), schedule_steps(total_steps))),
        "lr": LEARNING_RATE, "microbatch": args.microbatch, "eval_labels": st["eval_labels"],
        "eval_interval": st["eval_interval"], "model_seed": st["model_seed"], "data_seed": st["data_seed"],
        "dataset_id": reg["dataset"]["id"], "dataset_version": reg["dataset"]["version"],
        "train_shards": {d: len(shards[d]) for d in domains}, "domain_chunks": dict(zip(domains, chunks)),
        "data_plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "code_commit": os.environ.get("CODE_COMMIT", "unknown"), "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    from datetime import timedelta

    import torch.distributed as dist
    import torch.multiprocessing as mp
    from olmo.data import build_train_dataloader
    from olmo.eval import build_evaluators
    from olmo.model import OLMo
    from olmo.optim import build_optimizer
    from olmo.torch_util import barrier, get_global_rank, get_local_rank, get_world_size, seed_all
    from olmo.train import Trainer
    from olmo.util import add_cached_path_clients, prepare_cli_environment
    from torch.nn.parallel import DistributedDataParallel as DDP

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    torch.cuda.set_device(f"cuda:{get_local_rank()}")
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    prepare_cli_environment()
    add_cached_path_clients()

    install_dataset(PlannedDataset(plan, domains, shards))
    cfg = build_train_config(args, reg, run, out_dir)
    cfg.model.precision = cfg.precision
    cfg.device_train_batch_size = cfg.global_train_batch_size // get_world_size()
    cfg.device_train_grad_accum = cfg.device_train_batch_size // cfg.device_train_microbatch_size
    device = torch.device("cuda")
    Path(cfg.save_folder).mkdir(parents=True, exist_ok=True)
    cfg.save(Path(cfg.save_folder) / "config.yaml")

    wb_run = init_wandb(args, meta, out_dir) if get_global_rank() == 0 else None
    handler = CurveHandler(out_dir, st["eval_labels"], total_steps, wb_run)
    for name in ("", "trainer", "train", "olmo.train"):
        logging.getLogger(name).addHandler(handler)

    seed_all(cfg.seed)
    train_loader = build_train_dataloader(cfg)
    evaluators = build_evaluators(cfg, device)
    items = {ev.label: len(ev.eval_loader.dataset) for ev in evaluators}
    meta["eval_items"] = items
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    expected = st.get("expected_eval_items")
    if expected and items != expected:
        raise SystemExit(f"eval item counts {items} != expected {expected}")
    log.info("eval items per label: %s", items)
    barrier()

    olmo_model = OLMo(cfg.model)
    log.info("non-embedding parameters: %s", f"{olmo_model.num_params(include_embedding=False):,d}")
    olmo_model.set_activation_checkpointing(cfg.activation_checkpointing)
    dist_model = DDP(olmo_model.to(device), find_unused_parameters=cfg.ddp.find_unused_params)
    olmo_model.reset_parameters()
    optim = build_optimizer(cfg, dist_model)
    scheduler = make_scheduler(cfg)

    with Trainer(
        cfg=cfg, epoch=cfg.epoch, model=olmo_model, dist_model=dist_model, optim=optim,
        scheduler=scheduler, train_loader=train_loader, device=device, evaluators=evaluators,
    ) as trainer:
        if cfg.load_path is not None:
            log.info("restoring %s", cfg.load_path)
            trainer.restore_checkpoint(cfg.load_path, load_optimizer_state=True, load_trainer_state=True)
        trainer.fit()
    if wb_run is not None:
        wb_run.finish()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
