# domain_probes: the 31 DataDecide-60M domain runs

One trainer ([`train_60m.py`](train_60m.py)), one launcher ([`launch.sbatch`](launch.sbatch)) and
one list of runs ([`runs.json`](runs.json)) produce every 60M run in the skill-dag experiments:

- **24 pilots** `mix01`–`mix24`, the designed mixtures of [`../mixlaw/`](../mixlaw/README.md#mixture-sampling)
  (weights in [`../mixlaw/mixtures.json`](../mixlaw/mixtures.json)). The mixing-law and LightGBM surrogates are fit on them.
- **7 one-hot probes** `onehot_<domain>`, 100% of one domain each. The Skill-It probe matrix
  ([`../skillit/`](../skillit/README.md)) compares them with a reference run.

All 31 runs share the same code, the same hardware class, the same seeds and the same evaluation, so
they can be compared with each other and nothing else differs between a pilot and a probe except the
mixture weights.

## Setup

| | |
|---|---|
| Model | DataDecide-60M: $d_{\text{model}}$ 384, 16 layers, 12 heads, MLP ratio 8, sequence 2048, untied head, dolma2 vocabulary (100,352 embedding rows). 76,296,576 non-embedding parameters |
| Batch / LR | 96 sequences (196,608 tokens) per step; peak LR $5.8\times10^{-3}$ |
| Length | **1440 steps** = 283,115,520 tokens (4.96 tokens/param against DataDecide's 57,078,144), so the last eval lands on the final step |
| LR schedule | 144 steps linear warmup, 1152 steps constant, 144 steps cosine decay to 10% of peak |
| Data | The published `pretrain/olmo-127b` v1 train shards, **all** of them, read in place (no staged copy). Each run's sequences are fixed up front: exact largest-remainder domain counts for the weights, 2048-token chunks drawn **without replacement** uniformly over the whole domain, order shuffled. The plan is saved as `data_plan.npy` on scratch and its SHA-256 is in `run_meta.json`. The shard counts and token counts are checked against [`datasets/manifests/olmo-127b-v1/outputs.json`](../../../datasets/manifests/olmo-127b-v1/outputs.json) before the run starts |
| Seeds | model initialization 6198, data order 6198, for every run |
| Evaluation | the six ARC/MMLU **test** labels, **every item** (1172 / 2376 / 3018 / 4705 / 3077 / 3242 items for `arc_challenge`, `arc_easy`, `mmlu_stem`, `mmlu_humanities`, `mmlu_social_sciences`, `mmlu_other`, 17,590 in all), every 120 steps: 12 points, steps 120…1440. The trainer fails if a label scores a different number of items |
| Hardware | one NVIDIA L40S per run on FarmShare, micro-batch 4 (peak 23.7 GB; micro-batch 8 peaks at 44 GB, 16 does not fit), PyTorch 2.9.0+cu128, OLMo ladder model code from the `ai2-olmo` package |
| Cost | each run took 1 h 57 m 49 s to 1 h 59 m 13 s of Slurm wall-clock, about 61 GPU-hours for all 31 |

Why every item and not a subset: the earlier pilots and probes scored only the first 128 items of each
validation split, and that prefix is not representative (for the reference probe, MMLU humanities
z = +4.6, social sciences z = −4.2 against random 128-item subsets), which flips probe rankings on
humanities. Scoring every item of the test split costs about 2 minutes per run.

## Outputs

Committed, per run, in [`runs/<name>/`](runs/):

- `task_loss.jsonl`: one row per eval, `{"step": ..., "task_loss_bpb": {label: bits-per-byte}}`
- `run_meta.json`: the weights actually realized, schedule, dataset id/version, shard and chunk counts per
  domain, data-plan hash, code commit, host, Slurm job and array task, GPU, torch version, eval item counts

Not committed (on FarmShare scratch under `/scratch/users/nzhao2/agent-runs/domain-probes-20261006/`): the
final unsharded checkpoint `step1440-unsharded`, the data plan and the Slurm logs.

## Provenance

- Code: commit `5727e248923235b0e8c29abedd5f5d71cba51774` for all 31 runs (recorded in every `run_meta.json`).
- Smoke job 1792275 (60 steps at micro-batch 4, 8, 16; picked 4) and array job 1792276, one task per run, serial.
  Runs started on `oat-01` to `oat-06`. Tasks 28 to 30 (`mix22`, `mix23`, `mix24`) were moved to the `gpu` QOS or
  resubmitted (`mix24` as job 1826231) to get a GPU sooner; this changes where they ran, not what they
  computed. Each run's job id and node are in its `run_meta.json`.
- **W&B logging did not run.** The trainer logs to W&B project `domain-probes`, but its run-id helper
  (`wandb.util.generate_id`) does not exist in the installed wandb, so W&B was disabled with a warning at the start of
  every run and the runs are not in W&B. The training itself was not affected, and `task_loss.jsonl` and
  `run_meta.json` above are the record. The helper was replaced afterwards, which does not change any
  number.

## Reproduce

```bash
# smoke test (60 steps at three micro-batch sizes, checks the full eval) then the array
MODE=smoke  sbatch --job-name=domsmoke ... launch.sbatch
MODE=array  sbatch --job-name=domprobe --array=0-30%1 --dependency=afterok:<smoke job> ... launch.sbatch
```

`launch.sbatch` documents the environment it needs (`RUN_ROOT`, `DATA_ROOT`, `VENV`, optional `WANDB_ENV`).
A task whose `task_loss.jsonl` already ends at step 1440 is skipped, so the array can be resubmitted safely.

Tests: `py -m pytest experiments/skill-dag/domain_probes/tests`.
