#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --constraint=GPU_SKU:L40S
#SBATCH --exclude=wheat-01
# oat-01..06 are NOT excluded: they are the only nodes carrying L40S GPUs
# (barley-01..04's broken /scratch mount, from the manifest-rebuild CPU
# jobs, is unrelated to these oat GPU nodes).
#SBATCH --job-name=token-selection-train
#SBATCH --output=logs/train-%j.out
#SBATCH --error=logs/train-%j.err

set -Eeuo pipefail
: "${RUN_DIR:?}"
cd "${RUN_DIR}"
SCRIPTS_DIR="${SCRIPTS_DIR:-${RUN_DIR}/scripts}"
# shellcheck disable=SC1091
source "${SCRIPTS_DIR}/common.sh"
bash "${SCRIPTS_DIR}/setup_venv.sh"
bash "${SCRIPTS_DIR}/launch.sh"
