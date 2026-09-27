#!/usr/bin/env bash
# Dolma2-tokenize an OLMo-mix pool that a submit_olmo_mix_sample.sh run already
# downloaded to local scratch; write tokenized/ and plan/tokenized_manifest.json
# next to it. Does not touch AWS -- SRC_RUN_DIR is that sampling run's RUN_DIR.
set -Eeuo pipefail

SUNET="${SUNET:-nzhao2}"
RUN_NAME="${RUN_NAME:-}"
SRC_RUN_DIR="${SRC_RUN_DIR:?set SRC_RUN_DIR to the submit_olmo_mix_sample.sh run to tokenize}"
RUN_DIR="${RUN_DIR:-/scratch/users/${SUNET}/agent-runs/${RUN_NAME:-olmo-mix-dolma2-tok-$(date +%Y%m%d-%H%M%S)}}"
EDULLM_ROOT="${EDULLM_ROOT:-/scratch/users/${SUNET}/agent-runs/edullm-farmshare-staging}"
BASE_RUN_DIR="${BASE_RUN_DIR:-}"
SKIP_LINK="${SKIP_LINK:-0}"
TOK_CONCURRENCY="${TOK_CONCURRENCY:-40}"

mkdir -p "${RUN_DIR}/scripts" "${RUN_DIR}/logs" "${RUN_DIR}/data" "${RUN_DIR}/plan" "${RUN_DIR}/tokenized/shards"
cd "${RUN_DIR}"

OLMO_ROOT="${EDULLM_ROOT}/datasets/olmo"
for f in build_pool_tokenize_map.py \
  tokenize_olmo_shard.py tokenize_olmo_shard.sbatch \
  finalize_pool_tokenized_upload.py finalize_pool_tokenized_upload.sbatch; do
  cp -a "${OLMO_ROOT}/${f}" "${RUN_DIR}/scripts/"
done
sed -i 's/\r$//' "${RUN_DIR}/scripts/"*.{sbatch,py} 2>/dev/null || true

MANIFEST="${RUN_DIR}/plan/manifest.jsonl"
SUMMARY="${RUN_DIR}/plan/summary.json"
cp -a "${SRC_RUN_DIR}/plan/manifest.jsonl" "${MANIFEST}"
cp -a "${SRC_RUN_DIR}/plan/summary.json" "${SUMMARY}" 2>/dev/null || true
N=$(wc -l < "${MANIFEST}")

if [[ "${SKIP_LINK}" == "1" ]]; then
  echo "skip_link=1"
else
  echo "hardlinking ${SRC_RUN_DIR}/data -> ${RUN_DIR}/data"
  # SRC_RUN_DIR/data was populated directly by download_olmo_shard.py, keyed by
  # the same manifest "path" values build_pool_tokenize_map.py resolves below;
  # hardlink it in place of a download (fall back to copy across filesystems).
  cp -al "${SRC_RUN_DIR}/data/." "${RUN_DIR}/data/" 2>/dev/null || cp -a "${SRC_RUN_DIR}/data/." "${RUN_DIR}/data/"
fi

if [[ -n "${BASE_RUN_DIR}" && -f "${BASE_RUN_DIR}/.hf_token" ]]; then
  cp -a "${BASE_RUN_DIR}/.hf_token" "${RUN_DIR}/.hf_token"
fi
HF_TOKEN_FILE="${HF_TOKEN_FILE:-${RUN_DIR}/.hf_token}"

if [[ ! -x "${RUN_DIR}/venv/bin/python" ]]; then
  python3 -m venv "${RUN_DIR}/venv"
fi
# shellcheck disable=SC1091
source "${RUN_DIR}/venv/bin/activate"
pip install -U pip wheel
pip install tqdm transformers "numpy<2.1" zstandard sentencepiece protobuf

export EDULLM_ROOT RUN_DIR

if [[ -n "${BASE_RUN_DIR}" && -d "${BASE_RUN_DIR}" ]]; then
  echo "staging local base run ${BASE_RUN_DIR}"
  for domain in algebraic-stack arxiv open-web-math pes2o starcoder wiki; do
    for suffix in trimmed upsampled; do
      src="${BASE_RUN_DIR}/trim/${domain}/${domain}-${suffix}.json.gz"
      if [[ ! -f "${src}" ]]; then
        src="${BASE_RUN_DIR}/data/data/${domain}/${domain}-${suffix}.json.gz"
      fi
      if [[ -f "${src}" ]]; then
        dst="${RUN_DIR}/data/data/${domain}/${domain}-${suffix}.json.gz"
        mkdir -p "$(dirname "${dst}")"
        if [[ ! -e "${dst}" ]]; then
          ln -s "${src}" "${dst}" 2>/dev/null || cp -a "${src}" "${dst}"
        fi
      fi
    done
  done
  if [[ -d "${BASE_RUN_DIR}/data/data/dclm" ]]; then
    find "${BASE_RUN_DIR}/data/data/dclm" -type f \( -name '*.zstd' -o -name '*.json.gz' -o -name '*.jsonl.zstd' \) | while read -r src; do
      rel="${src#${BASE_RUN_DIR}/data/}"
      dst="${RUN_DIR}/data/${rel}"
      mkdir -p "$(dirname "${dst}")"
      if [[ ! -e "${dst}" ]]; then
        ln -s "${src}" "${dst}" 2>/dev/null || cp -a "${src}" "${dst}"
      fi
    done
  fi
fi

cat > "${RUN_DIR}/env.sh" <<EOF
RUN_DIR=${RUN_DIR}
VENV=${RUN_DIR}/venv
MANIFEST=${MANIFEST}
SUMMARY=${SUMMARY}
LOCAL_ROOT=${RUN_DIR}/data
SRC_RUN_DIR=${SRC_RUN_DIR}
HF_TOKEN_FILE=${HF_TOKEN_FILE}
EOF

SBATCH_EXPORT_COMMON="RUN_DIR=${RUN_DIR},VENV=${RUN_DIR}/venv,MANIFEST=${MANIFEST},HF_TOKEN_FILE=${HF_TOKEN_FILE}"

TRIM_ARGS=()
if [[ -n "${BASE_RUN_DIR}" && -d "${BASE_RUN_DIR}/trim" ]]; then
  TRIM_ARGS=(--trim-root "${BASE_RUN_DIR}/trim")
fi

MAP_JOB=$(sbatch --parsable --exclude=wheat-01 \
  --cpus-per-task=4 \
  --mem=8G \
  --time=01:00:00 \
  --job-name=pool-map \
  --chdir="${RUN_DIR}" \
  --output="${RUN_DIR}/logs/map-%j.out" \
  --error="${RUN_DIR}/logs/map-%j.err" \
  --wrap="bash -lc 'set -Eeuo pipefail; source ${RUN_DIR}/venv/bin/activate; python ${RUN_DIR}/scripts/build_pool_tokenize_map.py --manifest ${MANIFEST} --data-root ${RUN_DIR}/data ${TRIM_ARGS[*]} --out-dir ${RUN_DIR}/tokenized --map-file ${RUN_DIR}/tokenize_map.txt'")
echo "map_job=${MAP_JOB}"

mkdir -p "${RUN_DIR}/hf-cache"

TOK_JOB=$(sbatch --parsable --exclude=wheat-01 \
  --dependency=afterok:${MAP_JOB} \
  --array=0-$((N - 1))%${TOK_CONCURRENCY} \
  --chdir="${RUN_DIR}" \
  --export=ALL,${SBATCH_EXPORT_COMMON} \
  --output="${RUN_DIR}/logs/tokenize_%A_%a.out" \
  --error="${RUN_DIR}/logs/tokenize_%A_%a.err" \
  "${RUN_DIR}/scripts/tokenize_olmo_shard.sbatch")
echo "tokenize_job=${TOK_JOB}"

CHECK_JOB=$(sbatch --parsable --exclude=wheat-01 \
  --dependency=afterok:${TOK_JOB} \
  --chdir="${RUN_DIR}" \
  --export=ALL,${SBATCH_EXPORT_COMMON} \
  "${RUN_DIR}/scripts/finalize_pool_tokenized_upload.sbatch")
echo "check_job=${CHECK_JOB}"

echo "RUN_DIR=${RUN_DIR}"
echo "manifest_shards=${N}"
echo "${RUN_DIR}/tokenized/"
