#!/bin/bash
#SBATCH --job-name=ng13-res
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:50:00
#SBATCH --output=logs/res-%A_%a.out
set -euo pipefail
export PYTHONHASHSEED=0
export NG13_TOKENIZER="$1/tokenizer"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
RUN="$1"
OFFSET=${2:-0}
cd "$RUN"
C=/scratch/users/nzhao2/agent-runs/contam-20260910-060952
IDX=$((OFFSET + SLURM_ARRAY_TASK_ID))
LINE=$(sed -n "$((IDX + 1))p" res_shards.tsv)
CORPUS=$(printf '%s' "$LINE" | cut -f1)
DOM=$(printf '%s' "$LINE" | cut -f2)
SHARD=$(printf '%s' "$LINE" | cut -f3)
S=$(printf '%s' "$LINE" | cut -f4)
E=$(printf '%s' "$LINE" | cut -f5)
mkdir -p res_hits
OUT="res_hits/${CORPUS}__${DOM}__$(printf '%05d' "$IDX").json"
/scratch/users/nzhao2/refhq-regmix-5p5b-v1/venv/bin/python scan_tok.py \
  "$C/eval_ngrams_strict.pkl" "$SHARD" "$OUT" "$CORPUS" "$DOM" "$S" "$E"
