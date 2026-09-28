#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --constraint=GPU_SKU:L40S
#SBATCH --exclude=wheat-01
#SBATCH --job-name=token-selection-score-reference
#SBATCH --output=logs/score-reference-%j.out
#SBATCH --error=logs/score-reference-%j.err

set -Eeuo pipefail
: "${RUN_DIR:?}"
cd "${RUN_DIR}"
SCRIPTS_DIR="${SCRIPTS_DIR:-${RUN_DIR}/scripts}"
# shellcheck disable=SC1091
source "${SCRIPTS_DIR}/common.sh"
bash "${SCRIPTS_DIR}/setup_venv.sh"
bash "${SCRIPTS_DIR}/score_reference.sh"
