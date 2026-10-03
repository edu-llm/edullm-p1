#!/usr/bin/env bash
# 370M Skill-It probe arm (arm index 0) on one 4×L40S node, with the offline matrix
# rebuilt against the LightGBM-start probe.
#
# Stage a dedicated RUN_DIR before submission (stage_probe_rerun.sh): an OLMo-core
# snapshot whose .edullm was patched by patch_static_validation_arms.py (arms 0-6) and
# then by patch_probe_matrix.py, a symlink or copy of manifest/ready.json, and the
# rebuilt A_offline.json as RUN_DIR/A_offline.json.
#
# Required exported variables:
#   RUN_DIR, TRAIN_VENV, WANDB_ENV_FILE, HF_ENV_FILE
#
# The input manifest is read-only: this job never stages data or touches external
# object storage. Resources and time limit match farmshare_static_validation_l40s.sbatch
# (the static arms ran 16-17 h on this hardware).
#SBATCH --job-name=probelgb
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=72G
#SBATCH --time=20:00:00
#SBATCH --constraint=GPU_SKU:L40S
#SBATCH --output=logs/probe-rerun-%j.out
#SBATCH --error=logs/probe-rerun-%j.err

set -Eeuo pipefail

: "${RUN_DIR:?RUN_DIR is required}"
: "${TRAIN_VENV:?TRAIN_VENV is required}"
: "${WANDB_ENV_FILE:?WANDB_ENV_FILE is required}"
: "${HF_ENV_FILE:?HF_ENV_FILE is required}"

ARM_INDEX=0
ARM_ID=probe
REPO_DIR="${RUN_DIR}/OLMo-core"
EDULLM_DIR="${REPO_DIR}/.edullm"
MANIFEST="${RUN_DIR}/manifest/ready.json"
A_OFFLINE_JSON="${RUN_DIR}/A_offline.json"
ARM_ROOT="${RUN_DIR}/runs/${ARM_ID}"

[[ -x "${TRAIN_VENV}/bin/python" ]] || { echo "missing TRAIN_VENV=${TRAIN_VENV}" >&2; exit 2; }
[[ -f "${MANIFEST}" ]] || { echo "missing staged input manifest=${MANIFEST}" >&2; exit 2; }
[[ -f "${A_OFFLINE_JSON}" ]] || { echo "missing staged matrix=${A_OFFLINE_JSON}" >&2; exit 2; }
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

# Local-manifest-only path over the verified pre-staged manifest. Do not permit
# inherited cloud credential variables to reach the loader.
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

# The recipe must define the probe arm at index 0, and the matrix it carries must be
# the rebuilt one staged next to it.
"${TRAIN_VENV}/bin/python" - "${A_OFFLINE_JSON}" <<'PY'
import json
import sys

import numpy as np

import skillit_math

arm = skillit_math.arm_by_index(0)
if (arm.arm_id, arm.a_mode, skillit_math.data_seed(0)) != ("probe", "probe", 42):
    raise SystemExit(f"arm 0 is {arm.arm_id}/{arm.a_mode}/seed {skillit_math.data_seed(0)}, not probe/probe/42")
staged = np.asarray(json.load(open(sys.argv[1], encoding="utf-8"))["A"], dtype=np.float64)
if not np.array_equal(skillit_math.offline_a(), staged):
    raise SystemExit("the recipe's offline A is not the staged A_offline.json")
print(f"recipe arm 0: {arm.arm_id} a_mode={arm.a_mode} wandb_project={arm.wandb_project}")
print(f"offline_a_source_sha256={skillit_math.OFFLINE_A_SOURCE_SHA256}")
print(f"initial_weights={skillit_math.initial_weights(0).tolist()}")
print(f"offline_a nonzero entries={int((staged > 0).sum())} of {staged.size}")
PY

mkdir -p \
  "${RUN_DIR}/logs" \
  "${ARM_ROOT}/work" \
  "${ARM_ROOT}/checkpoints" \
  "${ARM_ROOT}/progress" \
  "${ARM_ROOT}/task_loss"
umask 077
run_name="probe-lgbref-farmshare-${SLURM_JOB_ID}"
printf "run_name=%s\narm_index=%s\narm=%s\ndata_seed=42\ninit_seed=0\nseed_all=42\nupdates=probe (offline A rebuilt against probe_lgb_start)\n" \
  "${run_name}" "${ARM_INDEX}" "${ARM_ID}" > "${ARM_ROOT}/run_identity.txt"

echo "[$(date -Is)] host=$(hostname) job=${SLURM_JOB_ID} arm_index=${ARM_INDEX} arm=${ARM_ID}"
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
