#!/bin/bash
#SBATCH --job-name=ng13-index
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:25:00
#SBATCH --output=%x-%j.out
set -euo pipefail
RUN="$1"
cd "$RUN"
export HF_HOME=/scratch/users/nzhao2/hf-cache
export HF_DATASETS_TRUST_REMOTE_CODE=1
/scratch/users/nzhao2/refhq-regmix-5p5b-v1/venv/bin/python build_eval_index.py "$RUN"
