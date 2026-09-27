#!/usr/bin/env bash
# Laptop-side submit: sync code, push the W&B session, stage inputs, launch training.
#
# No AWS session is pushed: staging is local file verification only.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/config.env"

SUNET="${FARMSHARE_SUNET:-nzhao2}"
SOCK="${FARMSHARE_SOCK:-/tmp/farmshare-${SUNET}.sock}"
HOST="${SUNET}@login.farmshare.stanford.edu"
LOCAL_REPO="${LOCAL_REPO:?set LOCAL_REPO to the OLMo-core checkout to sync}"
PUSH_WANDB_SESSION="${PUSH_WANDB_SESSION:-${SCRIPT_DIR}/push_wandb_session_to_farmshare.sh}"

ARM="${ARM:-}"
RECOVERY_MODE="${RECOVERY_MODE:-fresh}"
SKIP_STAGE="${SKIP_STAGE:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
STAGE_MODE="${STAGE_MODE:-corpora}"
TS="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="${RUN_DIR:-/scratch/users/${SUNET}/agent-runs/${EXPERIMENT_SLUG}-${TS}}"

export RUN_DIR LOCAL_REPO SOCK HOST
export ARM RECOVERY_MODE
export STAGE_CPUS STAGE_MEM STAGE_TIME TRAIN_GPUS TRAIN_CPUS TRAIN_MEM TRAIN_TIME

bash "${SCRIPT_DIR}/sync_repo.sh"

if [[ -x "${PUSH_WANDB_SESSION}" ]]; then
  bash "${PUSH_WANDB_SESSION}" "${RUN_DIR}"
else
  echo "no W&B session pusher at ${PUSH_WANDB_SESSION}; push wandb-session.env yourself" >&2
fi

STAGE_EXPORT="RUN_DIR='${RUN_DIR}',SCRIPTS_DIR='${RUN_DIR}/scripts',STAGE_MODE='${STAGE_MODE}'"

STAGE_JOB=""
if [[ "${SKIP_STAGE}" != "1" ]]; then
  STAGE_JOB="$(ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" bash -s <<EOF
set -Eeuo pipefail
JOB=\$(sbatch --parsable --exclude=wheat-01 \
  --partition=normal \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=${STAGE_CPUS} \
  --mem=${STAGE_MEM} \
  --time=${STAGE_TIME} \
  --job-name=${EXPERIMENT_SLUG}-stage \
  --chdir='${RUN_DIR}' \
  --output='${RUN_DIR}/logs/stage-%j.out' \
  --error='${RUN_DIR}/logs/stage-%j.err' \
  --export=ALL,${STAGE_EXPORT} \
  '${RUN_DIR}/scripts/stage_job.sbatch')
echo "\${JOB}"
EOF
)"
  echo "stage_job=${STAGE_JOB}"
fi

if [[ "${SKIP_TRAIN}" == "1" ]]; then
  echo "RUN_DIR=${RUN_DIR}"
  exit 0
fi

DEP_FLAG=""
if [[ -n "${STAGE_JOB}" ]]; then
  DEP_FLAG="--dependency=afterok:${STAGE_JOB}"
fi

TRAIN_EXPORT="RUN_DIR='${RUN_DIR}',SCRIPTS_DIR='${RUN_DIR}/scripts',RECOVERY_MODE='${RECOVERY_MODE}'"
if [[ -n "${ARM}" ]]; then
  TRAIN_EXPORT+=",ARM='${ARM}'"
fi

TRAIN_JOB="$(ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" bash -s <<EOF
set -Eeuo pipefail
JOB=\$(sbatch --parsable --exclude=wheat-01 ${DEP_FLAG} \
  --partition=gpu \
  --qos=gpu \
  --nodes=1 \
  --ntasks=1 \
  --gpus-per-node=${TRAIN_GPUS} \
  --cpus-per-task=${TRAIN_CPUS} \
  --mem=${TRAIN_MEM} \
  --time=${TRAIN_TIME} \
  --constraint=GPU_SKU:L40S \
  --job-name=${EXPERIMENT_SLUG}-train \
  --chdir='${RUN_DIR}' \
  --output='${RUN_DIR}/logs/train-%j.out' \
  --error='${RUN_DIR}/logs/train-%j.err' \
  --export=ALL,${TRAIN_EXPORT} \
  '${RUN_DIR}/scripts/train_job.sbatch')
echo "\${JOB}"
EOF
)"

echo "RUN_DIR=${RUN_DIR}"
echo "train_job=${TRAIN_JOB}"
