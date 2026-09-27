#!/usr/bin/env bash
# Bind the local, pinned FarmShare corpus directories into ready.json.
#
# No download and no AWS: STAGE_MODE=corpora (the default) just verifies the
# pinned source files exist and records their sizes. STAGE_MODE=references
# additionally materializes and averages the reference checkpoints trained by
# the hq-reference/instruct-reference arms; run it only after those two arms
# have finished.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/config.env"

STAGE_MODE="${STAGE_MODE:-corpora}"

bash "${SCRIPT_DIR}/setup_venv.sh"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}/.edullm"

stage_args=(
  "${REPO_DIR}/.edullm/farmshare/stage_local.py"
  --stage-root "${STAGE_ROOT}"
  --mode "${STAGE_MODE}"
)
if [[ "${STAGE_MODE}" == "references" ]]; then
  stage_args+=(--run-root "${RUN_ROOT}")
fi

"${PYTHON}" "${stage_args[@]}"

[[ -f "${INPUT_MANIFEST}" ]] || {
  echo "staging did not write ${INPUT_MANIFEST}" >&2
  exit 2
}
echo "stage_ok mode=${STAGE_MODE} manifest=${INPUT_MANIFEST}"
