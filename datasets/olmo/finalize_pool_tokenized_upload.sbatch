#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --exclude=wheat-01
#SBATCH --job-name=pool-tok-check
#SBATCH --output=logs/pool-tok-check-%j.out
#SBATCH --error=logs/pool-tok-check-%j.err

set -Eeuo pipefail

: "${RUN_DIR:?}" "${VENV:?}"

# shellcheck disable=SC1091
source "${VENV}/bin/activate"

python "${RUN_DIR}/scripts/finalize_pool_tokenized_upload.py" --run-dir "${RUN_DIR}"
