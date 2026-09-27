#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --exclude=wheat-01
#SBATCH --job-name=token-selection-stage
#SBATCH --output=logs/stage-%j.out
#SBATCH --error=logs/stage-%j.err

set -Eeuo pipefail
: "${RUN_DIR:?}"
cd "${RUN_DIR}"
bash "${SCRIPTS_DIR:-${RUN_DIR}/scripts}/stage.sh"
