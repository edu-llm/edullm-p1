# Skill-It on FarmShare (4 × L40S)

Runs 4 data-parallel ranks, not RunPod's production 8 — FarmShare's per-user
GPU quota caps at 4 concurrent GPUs. Both launch paths below always pass
`--allow-local-only`, so these are non-production runs by this repo's own
contract (skips the exactly-8-rank assertion and the forced
`--wandb-mode=online` requirement). Training, the Skill-It controller,
checkpoints, and the full task-loss eval suite are unaffected — see
`SKILLIT.md` for the exact distinction. `TRAIN_TIME=20:00:00` below is a
~16h expected wall-clock run with headroom already applied; treat it as
provisional until a real run confirms it, the same way RunPod's
`maximum_runtime_hours` is set from a measured benchmark rather than assumed.

## Launch path (`launch_no_aws.sh` / `train_no_aws.sbatch`)

`launch_no_aws.sh` reads training data straight from an already-existing
local `ready.json` manifest (`EDULLM_RUNPOD_INPUT_MANIFEST`);
`setup_venv_no_aws.sh` skips the `boto3`/`edullm-data` installs entirely,
since `.edullm/runpod/entrypoint.py`'s `resolve_local_datasets` never imports
them. No cloud session, no staging job, no external object-storage access
anywhere in this path. See `SKILLIT.md`'s "FarmShare deployment" section for
where the real local copy of `pretrain/olmo-127b/v1` currently lives and how
its permissions are scoped.

To hand this off to a second student on their own FarmShare account, use
`HANDOFF.md` plus the packaged bundle (code + this manifest + these scripts,
no dataset — the manifest points at absolute paths on the first student's
scratch, readable cluster-wide without a transfer). Submission on the second
account is direct `sbatch` against `train_no_aws.sbatch` from their own
socket — see `HANDOFF.md` for exact commands.

## Resource defaults

Override before submit:

```bash
export TRAIN_GPUS=4 TRAIN_CPUS=32 TRAIN_MEM=192G TRAIN_TIME=20:00:00
export STAGE_CPUS=8 STAGE_MEM=32G STAGE_TIME=06:00:00
```

Jobs always exclude `wheat-01`.

## Recovery

Re-submit with the same `RUN_DIR` and `RECOVERY_MODE=resume` (or `retry-start`
before the first checkpoint). Restaging never needs a fresh credentials file:
this path always reads training data from the local manifest.

## W&B policy

Every permanent checkpoint uploads full 20-task eval metrics to W&B. Only the
final checkpoint is uploaded as a model artifact (branch code).
