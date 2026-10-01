#!/usr/bin/env bash
# Preflight and submit one staged static validation arm (index 3-6) from FarmShare login.
#
# Run only after the dedicated RUN_DIR has been staged and patched by
# patch_static_validation_arms.py. No dataset bytes or credentials are copied.
#
# Required: RUN_DIR, TRAIN_VENV, WANDB_ENV_FILE, HF_ENV_FILE, ARM_INDEX, ARM_ID, DATA_SEED.
# Optional: AFTER_JOB_ID (queue behind that job with --dependency=afterany:<id>);
#           TEST_ONLY=1 (run every check and `sbatch --test-only`, submit nothing).
set -Eeuo pipefail

RUN_DIR="${RUN_DIR:?RUN_DIR is required}"
TRAIN_VENV="${TRAIN_VENV:?TRAIN_VENV is required}"
WANDB_ENV_FILE="${WANDB_ENV_FILE:?WANDB_ENV_FILE is required}"
HF_ENV_FILE="${HF_ENV_FILE:?HF_ENV_FILE is required}"
ARM_INDEX="${ARM_INDEX:?ARM_INDEX is required}"
ARM_ID="${ARM_ID:?ARM_ID is required}"
DATA_SEED="${DATA_SEED:?DATA_SEED is required}"
AFTER_JOB_ID="${AFTER_JOB_ID:-}"
TEST_ONLY="${TEST_ONLY:-0}"
JOB_SCRIPT="${RUN_DIR}/scripts/farmshare_static_validation_l40s.sbatch"
EDULLM_DIR="${RUN_DIR}/OLMo-core/.edullm"
ARM_ROOT="${RUN_DIR}/runs/${ARM_ID}"

[[ "${ARM_INDEX}" =~ ^[3-6]$ ]] || { echo "ARM_INDEX must be 3-6, got ${ARM_INDEX}" >&2; exit 2; }
[[ "${ARM_ID}" =~ ^static-[a-z0-9.-]+$ ]] || { echo "invalid ARM_ID=${ARM_ID}" >&2; exit 2; }
[[ "${DATA_SEED}" =~ ^[0-9]+$ ]] || { echo "invalid DATA_SEED=${DATA_SEED}" >&2; exit 2; }
[[ -z "${AFTER_JOB_ID}" || "${AFTER_JOB_ID}" =~ ^[0-9]+$ ]] || { echo "invalid AFTER_JOB_ID" >&2; exit 2; }
[[ -x "${TRAIN_VENV}/bin/python" ]] || { echo "missing TRAIN_VENV=${TRAIN_VENV}" >&2; exit 2; }
[[ -r "${WANDB_ENV_FILE}" ]] || { echo "missing W&B session file=${WANDB_ENV_FILE}" >&2; exit 2; }
[[ -r "${HF_ENV_FILE}" ]] || { echo "missing HF session file=${HF_ENV_FILE}" >&2; exit 2; }
[[ -f "${RUN_DIR}/manifest/ready.json" ]] || { echo "missing read-only data manifest" >&2; exit 2; }
[[ -f "${EDULLM_DIR}/runpod/entrypoint.py" ]] || { echo "missing patched code" >&2; exit 2; }
[[ -f "${JOB_SCRIPT}" ]] || { echo "missing job script=${JOB_SCRIPT}" >&2; exit 2; }
if [[ -e "${ARM_ROOT}" ]] && [[ -n "$(ls -A "${ARM_ROOT}")" ]]; then
  echo "refusing non-empty ARM_ROOT=${ARM_ROOT}" >&2
  exit 2
fi
(
  source "${WANDB_ENV_FILE}"
  [[ -n "${WANDB_API_KEY:-}" ]]
) || { echo "W&B session lacks WANDB_API_KEY" >&2; exit 2; }
(
  source "${HF_ENV_FILE}"
  [[ -n "${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}" ]]
) || { echo "HF session lacks HF_TOKEN/HUGGING_FACE_HUB_TOKEN" >&2; exit 2; }
bash -n "${JOB_SCRIPT}"

# The exported arm must be the one the patched recipe defines at ARM_INDEX.
PYTHONPATH="${EDULLM_DIR}" "${TRAIN_VENV}/bin/python" - "${ARM_INDEX}" "${ARM_ID}" "${DATA_SEED}" <<'PY'
import sys

import skillit_math

index, arm_id, seed = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
arm = skillit_math.arm_by_index(index)
if arm.arm_id != arm_id or skillit_math.data_seed(index) != seed or arm.a_mode != "static":
    raise SystemExit(f"arm {index}/{arm_id}/seed {seed} does not match the patched recipe")
print(f"recipe arm {index}: {arm.arm_id} data_seed={seed} wandb_project={arm.wandb_project}")
PY

sbatch_args=(
  "--chdir=${RUN_DIR}"
  "--job-name=${ARM_ID}"
  "--export=ALL,RUN_DIR=${RUN_DIR},TRAIN_VENV=${TRAIN_VENV},WANDB_ENV_FILE=${WANDB_ENV_FILE},HF_ENV_FILE=${HF_ENV_FILE},ARM_INDEX=${ARM_INDEX},ARM_ID=${ARM_ID},DATA_SEED=${DATA_SEED}"
)
if [[ -n "${AFTER_JOB_ID}" ]]; then
  sbatch_args+=("--dependency=afterany:${AFTER_JOB_ID}")
fi
sbatch --test-only "${sbatch_args[@]}" "${JOB_SCRIPT}"
if [[ "${TEST_ONLY}" == "1" ]]; then
  echo "TEST_ONLY=1: not submitting ${ARM_ID}"
  exit 0
fi
job_id="$(sbatch --parsable "${sbatch_args[@]}" "${JOB_SCRIPT}")"
printf '%s\n' "${job_id}" | tee "${RUN_DIR}/submitted_job_id_${ARM_ID}.txt"
