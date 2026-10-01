#!/usr/bin/env bash
# Preflight and submit a staged static LightGBM run from FarmShare login.
#
# Run only after the dedicated RUN_DIR has been staged and patched by
# patch_legacy_static_lgbm_arm.py. No dataset bytes or credentials are copied.
set -Eeuo pipefail

RUN_DIR="${RUN_DIR:?RUN_DIR is required}"
TRAIN_VENV="${TRAIN_VENV:?TRAIN_VENV is required}"
WANDB_ENV_FILE="${WANDB_ENV_FILE:?WANDB_ENV_FILE is required}"
HF_ENV_FILE="${HF_ENV_FILE:?HF_ENV_FILE is required}"
JOB_SCRIPT="${RUN_DIR}/scripts/farmshare_static_lgbm_l40s.sbatch"

[[ -x "${TRAIN_VENV}/bin/python" ]] || { echo "missing TRAIN_VENV=${TRAIN_VENV}" >&2; exit 2; }
[[ -r "${WANDB_ENV_FILE}" ]] || { echo "missing W&B session file=${WANDB_ENV_FILE}" >&2; exit 2; }
[[ -r "${HF_ENV_FILE}" ]] || { echo "missing HF session file=${HF_ENV_FILE}" >&2; exit 2; }
[[ -f "${RUN_DIR}/manifest/ready.json" ]] || { echo "missing read-only data manifest" >&2; exit 2; }
[[ -f "${RUN_DIR}/OLMo-core/.edullm/runpod/entrypoint.py" ]] || { echo "missing patched code" >&2; exit 2; }
[[ -f "${JOB_SCRIPT}" ]] || { echo "missing job script=${JOB_SCRIPT}" >&2; exit 2; }
(
  source "${WANDB_ENV_FILE}"
  [[ -n "${WANDB_API_KEY:-}" ]]
) || { echo "W&B session lacks WANDB_API_KEY" >&2; exit 2; }
(
  source "${HF_ENV_FILE}"
  [[ -n "${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}" ]]
) || { echo "HF session lacks HF_TOKEN/HUGGING_FACE_HUB_TOKEN" >&2; exit 2; }
bash -n "${JOB_SCRIPT}"

export_args=(
  "--export=ALL,RUN_DIR=${RUN_DIR},TRAIN_VENV=${TRAIN_VENV},WANDB_ENV_FILE=${WANDB_ENV_FILE},HF_ENV_FILE=${HF_ENV_FILE}"
)
sbatch --test-only --chdir="${RUN_DIR}" "${export_args[@]}" "${JOB_SCRIPT}"
job_id="$(sbatch --parsable --chdir="${RUN_DIR}" "${export_args[@]}" "${JOB_SCRIPT}")"
printf '%s\n' "${job_id}" | tee "${RUN_DIR}/submitted_job_id.txt"
