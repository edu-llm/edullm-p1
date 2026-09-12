#!/bin/bash
#SBATCH --job-name=ng13-scan
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:30:00
#SBATCH --output=logs/scan-%A_%a.out
set -euo pipefail
RUN="$1"
cd "$RUN"
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" shards.tsv)
CORPUS=$(printf '%s' "$LINE" | cut -f1)
SOURCE=$(printf '%s' "$LINE" | cut -f2)
SHARD=$(printf '%s' "$LINE" | cut -f3)
OUT="results/$(printf '%05d' "$SLURM_ARRAY_TASK_ID").json"
/scratch/users/nzhao2/refhq-regmix-5p5b-v1/venv/bin/python scan_shard.py \
  "$RUN/eval_index.pkl" "$SHARD" "$OUT" "$CORPUS" "$SOURCE"
