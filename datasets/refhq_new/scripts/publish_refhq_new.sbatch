#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --exclude=wheat-01
#SBATCH --job-name=refhq-new-layout
#SBATCH --output=logs/refhq_new_layout_%j.out
#SBATCH --error=logs/refhq_new_layout_%j.err

set -Eeuo pipefail

: "${RUN_DIR:?}" "${STAGE_DIR:?}"

VENV="${VENV:-${RUN_DIR}/venv}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

: "${REFHQ_NEW_SCRIPTS:=${RUN_DIR}/datasets/refhq_new/scripts}"
SCRATCH_ROOT="${SCRATCH_ROOT:-${RUN_DIR}}"
TOKENIZED_ROOT="${TOKENIZED_ROOT:-}"
MANIFEST="${MANIFEST:-${SCRATCH_ROOT}/manifests/tokenized_manifest.json}"

# shellcheck disable=SC1091
source "${REFHQ_NEW_SCRIPTS}/lib.sh"
refhq_new_export_pythonpath "${RUN_DIR}"

LAYOUT_ARGS=(
  --scratch-root "${SCRATCH_ROOT}"
  --stage-dir "${STAGE_DIR}"
  --dataset-id "${DATASET_ID:-pretrain/refhq-instruct}"
  --force
)
if [[ -n "${TOKENIZED_ROOT}" ]]; then
  LAYOUT_ARGS+=(--tokenized-root "${TOKENIZED_ROOT}")
fi
if [[ -f "${MANIFEST}" ]]; then
  LAYOUT_ARGS+=(--manifest "${MANIFEST}")
fi
# finalize_upload.py already staged tokens/ into STAGE_DIR.
if [[ -d "${STAGE_DIR}/tokens" ]]; then
  LAYOUT_ARGS+=(--skip-stage)
fi

python "${REFHQ_NEW_SCRIPTS}/publish_refhq_new.py" "${LAYOUT_ARGS[@]}"
