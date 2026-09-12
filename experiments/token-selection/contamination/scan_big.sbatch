#!/bin/bash
#SBATCH --job-name=ng13-big
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=logs/big-%A_%a.out
set -euo pipefail
RUN="$1"
cd "$RUN"
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" big_shards.tsv)
CORPUS=$(printf '%s' "$LINE" | cut -f1)
SOURCE=$(printf '%s' "$LINE" | cut -f2)
SHARD=$(printf '%s' "$LINE" | cut -f3)
OUT="results/big-$(printf '%03d' "$SLURM_ARRAY_TASK_ID").json"
/scratch/users/nzhao2/refhq-regmix-5p5b-v1/venv/bin/python scan_big.py \
  "$RUN/eval_index.pkl" "$SHARD" "$OUT" "$CORPUS" "$SOURCE" 13
