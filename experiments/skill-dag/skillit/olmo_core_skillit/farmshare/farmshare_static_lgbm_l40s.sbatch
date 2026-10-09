#!/usr/bin/env bash
# 370M fixed static LightGBM mixture run on one 4×L40S node.
#
# Stage a dedicated RUN_DIR before submission. It must contain a patched,
# loader-compatible OLMo-core snapshot at OLMo-core/.edullm with arm index 2
# defined as static-lgbm-min1pct, plus a symlink or copy of manifest/ready.json.
#
# Required exported variables:
#   RUN_DIR, TRAIN_VENV, WANDB_ENV_FILE, HF_ENV_FILE
#
# The input manifest is read-only: this job never stages data or touches
# external object storage.
# The resource request is based on FarmShare job 1719708 (same 370M 4×L40S
# contract: 32 CPUs, 16:58:51 elapsed, 63.9 GiB MaxRSS).
#SBATCH --job-name=static-lgbm-min1pct
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=72G
#SBATCH --time=18:30:00
#SBATCH --constraint=GPU_SKU:L40S
#SBATCH --output=logs/static-lgbm-%j.out
#SBATCH --error=logs/static-lgbm-%j.err

set -Eeuo pipefail

: "${RUN_DIR:?RUN_DIR is required}"
: "${TRAIN_VENV:?TRAIN_VENV is required}"
: "${WANDB_ENV_FILE:?WANDB_ENV_FILE is required}"
: "${HF_ENV_FILE:?HF_ENV_FILE is required}"

REPO_DIR="${RUN_DIR}/OLMo-core"
EDULLM_DIR="${REPO_DIR}/.edullm"
MANIFEST="${RUN_DIR}/manifest/ready.json"
ARM_ROOT="${RUN_DIR}/runs/static-lgbm-min1pct"

[[ -x "${TRAIN_VENV}/bin/python" ]] || { echo "missing TRAIN_VENV=${TRAIN_VENV}" >&2; exit 2; }
[[ -f "${MANIFEST}" ]] || { echo "missing staged input manifest=${MANIFEST}" >&2; exit 2; }
[[ -r "${WANDB_ENV_FILE}" ]] || { echo "missing W&B session file=${WANDB_ENV_FILE}" >&2; exit 2; }
[[ -r "${HF_ENV_FILE}" ]] || { echo "missing HF session file=${HF_ENV_FILE}" >&2; exit 2; }
[[ -f "${EDULLM_DIR}/runpod/entrypoint.py" ]] || { echo "missing staged code=${EDULLM_DIR}" >&2; exit 2; }

source /etc/profile.d/z00_lmod.sh 2>/dev/null || true
module load python/3.12.3 2>/dev/null || true
module load cuda/12.4.0 2>/dev/null || module load cuda/12.9.0 2>/dev/null || true
source "${TRAIN_VENV}/bin/activate"

# FarmShare's flash-attention wheel is incompatible with the L40S runtime.
export OLMO_FLASH_ATTENTION=0
export OLMO_ATTN_BACKEND=torch
export OLMO_FUSED_LOSS=0
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="${RUN_DIR}/hf-cache"
export TRANSFORMERS_CACHE="${HF_HOME}"
export WANDB_DIR="${ARM_ROOT}/wandb"
export EDULLM_RUNPOD_INPUT_MANIFEST="${MANIFEST}"
export PYTHONPATH="${REPO_DIR}/src:${EDULLM_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# The static run is intentionally a local-manifest-only path over the
# verified pre-staged manifest. Do not permit inherited cloud credential
# variables to reach the loader.
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_PROFILE \
  AWS_DEFAULT_PROFILE AWS_SHARED_CREDENTIALS_FILE AWS_CONFIG_FILE \
  AWS_WEB_IDENTITY_TOKEN_FILE AWS_ROLE_ARN \
  AWS_CONTAINER_CREDENTIALS_RELATIVE_URI AWS_CONTAINER_CREDENTIALS_FULL_URI || true
source "${WANDB_ENV_FILE}"
source "${HF_ENV_FILE}"
export HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN}}"
[[ -n "${WANDB_API_KEY:-}" ]] || { echo "WANDB session lacks WANDB_API_KEY" >&2; exit 2; }
[[ -n "${HF_TOKEN}" ]] || { echo "HF session lacks HF_TOKEN/HUGGING_FACE_HUB_TOKEN" >&2; exit 2; }

mkdir -p \
  "${RUN_DIR}/logs" \
  "${ARM_ROOT}/work" \
  "${ARM_ROOT}/checkpoints" \
  "${ARM_ROOT}/progress" \
  "${ARM_ROOT}/task_loss"
umask 077
run_name="static-lgbm-min1pct-farmshare-${SLURM_JOB_ID}"
printf "run_name=%s\narm=static-lgbm-min1pct\nseed=42\nupdates=disabled\n" "${run_name}" \
  > "${ARM_ROOT}/run_identity.txt"

echo "[$(date -Is)] host=$(hostname) job=${SLURM_JOB_ID} arm=static-lgbm-min1pct"
nvidia-smi -L

exec "${TRAIN_VENV}/bin/python" -m torch.distributed.run --standalone --nproc-per-node=4 \
  "${EDULLM_DIR}/runpod/entrypoint.py" \
  --arm-index 2 \
  --run-name "${run_name}" \
  --work-dir "${ARM_ROOT}/work" \
  --save-folder "${ARM_ROOT}/checkpoints" \
  --progress-dir "${ARM_ROOT}/progress" \
  --task-loss-dir "${ARM_ROOT}/task_loss" \
  --task-loss-evaluator "${EDULLM_DIR}/eval_task_loss_olmo_core.py" \
  --wandb-mode online \
  --allow-local-only
