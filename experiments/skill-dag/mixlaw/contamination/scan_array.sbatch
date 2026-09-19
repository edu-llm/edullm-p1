#!/usr/bin/env bash
# Scan a corpus's text, one array task per (domain, shard, path) line of a
# task-list TSV. One line per array index:
#
#   <domain>\t<shard-id>\t<absolute path to a shard file OR a directory of them>
#
# Sharded by TEXT FILE rather than by domain wherever a corpus's per-domain
# file counts are wildly uneven (e.g. one domain shipping 100+ files against
# another's single file) -- a per-domain array would leave one task doing most
# of the work while the rest idle. Build the task-list TSV once per corpus
# (see README.md's "Reproducing" section for exactly how it was built for
# each of the three corpora this paper scans) and pass its line count minus
# one as --array.
#
# Everything this invokes is stdlib-only (`scan_corpus.py`, `item_identity.py`
# next to it), so plain python3 works and no venv is needed.
#
# Requires: CN_TASKLIST, CN_INDEX, CN_OUT
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=03:00:00
#SBATCH --job-name=contam-scan
#SBATCH --output=logs/scan-%A_%a.out
#SBATCH --error=logs/scan-%A_%a.err
set -Eeuo pipefail

: "${CN_TASKLIST:?}"
: "${CN_INDEX:?}"
: "${CN_OUT:?}"

# Resolve scan_corpus.py relative to this script rather than to a checkout
# path baked in at submit time, so this sbatch file runs unmodified wherever
# this directory is checked out.
CONTAM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${CN_TASKLIST}")
[ -n "${LINE}" ] || { echo "no task line for id ${SLURM_ARRAY_TASK_ID}" >&2; exit 2; }
DOMAIN=$(printf '%s' "${LINE}" | cut -f1)
SHARD=$(printf '%s' "${LINE}" | cut -f2)
TEXT=$(printf '%s' "${LINE}" | cut -f3)

echo "task ${SLURM_ARRAY_TASK_ID}: domain=${DOMAIN} shard=${SHARD} on $(hostname)"
echo "text=${TEXT}"

exec python3 "${CONTAM_DIR}/scan_corpus.py" \
  --index "${CN_INDEX}" \
  --trim "${TEXT}" \
  --domain "${DOMAIN}" \
  --shard "${SHARD}" \
  --out-dir "${CN_OUT}"
