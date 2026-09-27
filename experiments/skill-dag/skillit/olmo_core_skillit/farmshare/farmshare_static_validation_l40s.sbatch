#!/usr/bin/env bash
# 370M fixed-mixture static validation arm (arm index 3-6) on one 4×L40S node.
#
# Stage a dedicated RUN_DIR before submission. It must contain an OLMo-core
# snapshot whose .edullm was patched by patch_static_validation_arms.py (arms
# 0-6), plus a symlink or copy of manifest/ready.json. One RUN_DIR serves all
# four arms; each arm trains into its own runs/<arm_id>/ folder.
#
# Required exported variables:
#   RUN_DIR, TRAIN_VENV, WANDB_ENV_FILE, HF_ENV_FILE, ARM_INDEX, ARM_ID, DATA_SEED
#
# The input manifest is read-only: this job never stages data or touches
# external object storage.
# Resources match farmshare_static_lgbm_l40s.sbatch (job 1744338, 17:11:45
# elapsed); the time limit leaves headroom for a slower node.
#SBATCH --job-name=static-validation
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=72G
#SBATCH --time=20:00:00
#SBATCH --constraint=GPU_SKU:L40S
#SBATCH --output=logs/static-validation-%j.out
#SBATCH --error=logs/static-validation-%j.err

set -Eeuo pipefail

: "${RUN_DIR:?RUN_DIR is required}"
: "${TRAIN_VENV:?TRAIN_VENV is required}"
: "${WANDB_ENV_FILE:?WANDB_ENV_FILE is required}"
: "${HF_ENV_FILE:?HF_ENV_FILE is required}"
: "${ARM_INDEX:?ARM_INDEX is required}"
: "${ARM_ID:?ARM_ID is required}"
: "${DATA_SEED:?DATA_SEED is required}"

REPO_DIR="${RUN_DIR}/OLMo-core"
EDULLM_DIR="${REPO_DIR}/.edullm"
MANIFEST="${RUN_DIR}/manifest/ready.json"
ARM_ROOT="${RUN_DIR}/runs/${ARM_ID}"

[[ "${ARM_INDEX}" =~ ^[3-6]$ ]] || { echo "ARM_INDEX must be 3-6, got ${ARM_INDEX}" >&2; exit 2; }
[[ "${ARM_ID}" =~ ^static-[a-z0-9.-]+$ ]] || { echo "invalid ARM_ID=${ARM_ID}" >&2; exit 2; }
[[ "${DATA_SEED}" =~ ^[0-9]+$ ]] || { echo "invalid DATA_SEED=${DATA_SEED}" >&2; exit 2; }
[[ -x "${TRAIN_VENV}/bin/python" ]] || { echo "missing TRAIN_VENV=${TRAIN_VENV}" >&2; exit 2; }
[[ -f "${MANIFEST}" ]] || { echo "missing staged input manifest=${MANIFEST}" >&2; exit 2; }
[[ -r "${WANDB_ENV_FILE}" ]] || { echo "missing W&B session file=${WANDB_ENV_FILE}" >&2; exit 2; }
[[ -r "${HF_ENV_FILE}" ]] || { echo "missing HF session file=${HF_ENV_FILE}" >&2; exit 2; }
[[ -f "${EDULLM_DIR}/runpod/entrypoint.py" ]] || { echo "missing staged code=${EDULLM_DIR}" >&2; exit 2; }
# Trainer.fit silently resumes from any checkpoint in the save folder, so a
# fresh arm must start from an empty ARM_ROOT.
if [[ -e "${ARM_ROOT}" ]] && [[ -n "$(ls -A "${ARM_ROOT}")" ]]; then
  echo "refusing non-empty ARM_ROOT=${ARM_ROOT}" >&2
  exit 2
fi

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

# The static runs are intentionally a local-manifest-only path over the
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

# The exported arm must be the one the patched recipe defines at ARM_INDEX.
"${TRAIN_VENV}/bin/python" - "${ARM_INDEX}" "${ARM_ID}" "${DATA_SEED}" <<'PY'
import sys

import skillit_math

index, arm_id, seed = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
arm = skillit_math.arm_by_index(index)
if arm.arm_id != arm_id or skillit_math.data_seed(index) != seed or arm.a_mode != "static":
    raise SystemExit(
        f"exported arm {index}/{arm_id}/seed {seed} does not match the recipe: "
        f"{arm.arm_id}/{arm.a_mode}/seed {skillit_math.data_seed(index)}"
    )
print(f"recipe arm {index}: {arm.arm_id} data_seed={seed} wandb_project={arm.wandb_project}")
print(f"initial_weights={skillit_math.initial_weights(index).tolist()}")
PY

mkdir -p \
  "${RUN_DIR}/logs" \
  "${ARM_ROOT}/work" \
  "${ARM_ROOT}/checkpoints" \
  "${ARM_ROOT}/progress" \
  "${ARM_ROOT}/task_loss"
umask 077
run_name="${ARM_ID}-farmshare-${SLURM_JOB_ID}"
printf "run_name=%s\narm_index=%s\narm=%s\ndata_seed=%s\ninit_seed=0\nseed_all=42\nupdates=disabled\n" \
  "${run_name}" "${ARM_INDEX}" "${ARM_ID}" "${DATA_SEED}" > "${ARM_ROOT}/run_identity.txt"

echo "[$(date -Is)] host=$(hostname) job=${SLURM_JOB_ID} arm_index=${ARM_INDEX} arm=${ARM_ID} data_seed=${DATA_SEED}"
nvidia-smi -L
echo "gpu_driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sort -u | paste -sd, -)"
"${TRAIN_VENV}/bin/python" - <<'PY'
from importlib import metadata

import olmo_core
import torch
import wandb

print(f"torch={torch.__version__} cuda={torch.version.cuda} wandb={wandb.__version__}")
print(f"olmo_core={olmo_core.__file__}")
for dist in ("ai2-olmo-core", "ai2-olmo", "numpy"):
    try:
        print(f"{dist}={metadata.version(dist)}")
    except metadata.PackageNotFoundError:
        print(f"{dist}=not installed")
PY

exec "${TRAIN_VENV}/bin/python" -m torch.distributed.run --standalone --nproc-per-node=4 \
  "${EDULLM_DIR}/runpod/entrypoint.py" \
  --arm-index "${ARM_INDEX}" \
  --run-name "${run_name}" \
  --work-dir "${ARM_ROOT}/work" \
  --save-folder "${ARM_ROOT}/checkpoints" \
  --progress-dir "${ARM_ROOT}/progress" \
  --task-loss-dir "${ARM_ROOT}/task_loss" \
  --task-loss-evaluator "${EDULLM_DIR}/eval_task_loss_olmo_core.py" \
  --wandb-mode online \
  --allow-local-only
