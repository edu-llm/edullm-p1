#!/bin/bash
#SBATCH --job-name=ng13-strict
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:40:00
#SBATCH --output=logs/strict-%A_%a.out
set -euo pipefail
export PYTHONHASHSEED=0          # the prior pipeline hashes with builtin hash()
RUN="$1"
cd "$RUN"
C=/scratch/users/nzhao2/agent-runs/contam-20260910-060952
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" strict_shards.tsv)
CORPUS=$(printf '%s' "$LINE" | cut -f1)
SOURCE=$(printf '%s' "$LINE" | cut -f2)
PATH_IN=$(printf '%s' "$LINE" | cut -f3)
NSH=$(printf '%s' "$LINE" | cut -f4)
SID=$(printf '%s' "$LINE" | cut -f5)
OUT="strict_hits/${CORPUS}__${SOURCE}__$(printf '%05d' "$SLURM_ARRAY_TASK_ID").json"
mkdir -p strict_hits
/scratch/users/nzhao2/refhq-regmix-5p5b-v1/venv/bin/python "$C/contam_scan.py" \
  "$C/eval_ngrams_strict.pkl" "$PATH_IN" "$NSH" "$SID" "$OUT"
