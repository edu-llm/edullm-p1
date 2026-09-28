#!/usr/bin/env bash
# Score the whole RegMix corpus once against the frozen Instruct reference, on
# the 1-GPU qos=normal lane. Separate from launch.sh (which runs the 4-GPU
# qos=gpu production training entrypoint) -- this runs score_reference.py
# instead.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}/.edullm"

# SCORE_REFERENCE_EXTRA_ARGS lets a smoke/validation invocation pass e.g.
# "--max-instances 20000 --output-root /some/validation/dir" without editing
# score_reference_job.sbatch. Intentionally word-split (unquoted) below.
exec "${PYTHON}" "${REPO_DIR}/.edullm/farmshare/score_reference.py" \
  --stage-root "${STAGE_ROOT}" \
  ${SCORE_REFERENCE_EXTRA_ARGS:-}
