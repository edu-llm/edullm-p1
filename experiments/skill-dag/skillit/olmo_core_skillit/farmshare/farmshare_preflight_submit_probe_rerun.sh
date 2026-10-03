#!/usr/bin/env bash
# Preflight and submit the staged 370M probe-arm rerun (arm index 0) from FarmShare login.
#
# Run only after stage_probe_rerun.sh has built RUN_DIR. No dataset bytes or credentials
# are copied.
#
# Required: RUN_DIR, TRAIN_VENV, WANDB_ENV_FILE, HF_ENV_FILE.
# Optional: AFTER_JOB_ID (queue behind that job with --dependency=afterany:<id>);
#           TEST_ONLY=1 (run every check and `sbatch --test-only`, submit nothing).
set -Eeuo pipefail

RUN_DIR="${RUN_DIR:?RUN_DIR is required}"
TRAIN_VENV="${TRAIN_VENV:?TRAIN_VENV is required}"
WANDB_ENV_FILE="${WANDB_ENV_FILE:?WANDB_ENV_FILE is required}"
HF_ENV_FILE="${HF_ENV_FILE:?HF_ENV_FILE is required}"
AFTER_JOB_ID="${AFTER_JOB_ID:-}"
TEST_ONLY="${TEST_ONLY:-0}"
JOB_SCRIPT="${RUN_DIR}/scripts/farmshare_probe_rerun_l40s.sbatch"
EDULLM_DIR="${RUN_DIR}/OLMo-core/.edullm"
ARM_ROOT="${RUN_DIR}/runs/probe"

[[ -z "${AFTER_JOB_ID}" || "${AFTER_JOB_ID}" =~ ^[0-9]+$ ]] || { echo "invalid AFTER_JOB_ID" >&2; exit 2; }
[[ -x "${TRAIN_VENV}/bin/python" ]] || { echo "missing TRAIN_VENV=${TRAIN_VENV}" >&2; exit 2; }
[[ -r "${WANDB_ENV_FILE}" ]] || { echo "missing W&B session file=${WANDB_ENV_FILE}" >&2; exit 2; }
[[ -r "${HF_ENV_FILE}" ]] || { echo "missing HF session file=${HF_ENV_FILE}" >&2; exit 2; }
[[ -f "${RUN_DIR}/manifest/ready.json" ]] || { echo "missing read-only data manifest" >&2; exit 2; }
[[ -f "${RUN_DIR}/A_offline.json" ]] || { echo "missing staged A_offline.json" >&2; exit 2; }
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

# The patched recipe must define the probe arm at index 0 and carry the staged matrix.
PYTHONPATH="${EDULLM_DIR}" "${TRAIN_VENV}/bin/python" - "${RUN_DIR}/A_offline.json" <<'PY'
import json
import sys

import numpy as np

import skillit_math

arm = skillit_math.arm_by_index(0)
if (arm.arm_id, arm.a_mode, skillit_math.data_seed(0)) != ("probe", "probe", 42):
    raise SystemExit("arm 0 does not match the patched recipe")
staged = np.asarray(json.load(open(sys.argv[1], encoding="utf-8"))["A"], dtype=np.float64)
if not np.array_equal(skillit_math.offline_a(), staged):
    raise SystemExit("the recipe's offline A is not the staged A_offline.json")
print(f"recipe arm 0: {arm.arm_id} a_mode={arm.a_mode} wandb_project={arm.wandb_project}")
print(f"offline_a_source_sha256={skillit_math.OFFLINE_A_SOURCE_SHA256}")
PY

sbatch_args=(
  "--chdir=${RUN_DIR}"
  "--export=ALL,RUN_DIR=${RUN_DIR},TRAIN_VENV=${TRAIN_VENV},WANDB_ENV_FILE=${WANDB_ENV_FILE},HF_ENV_FILE=${HF_ENV_FILE}"
)
if [[ -n "${AFTER_JOB_ID}" ]]; then
  sbatch_args+=("--dependency=afterany:${AFTER_JOB_ID}")
fi
sbatch --test-only "${sbatch_args[@]}" "${JOB_SCRIPT}"
if [[ "${TEST_ONLY}" == "1" ]]; then
  echo "TEST_ONLY=1: not submitting the probe rerun"
  exit 0
fi
job_id="$(sbatch --parsable "${sbatch_args[@]}" "${JOB_SCRIPT}")"
printf '%s\n' "${job_id}" | tee "${RUN_DIR}/submitted_job_id_probe.txt"
