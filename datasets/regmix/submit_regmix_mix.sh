#!/usr/bin/env bash
# Build the RegMix-aligned 10B mix on FarmShare from a local olmohq run's pool.
set -Eeuo pipefail

SUNET="${SUNET:-nzhao2}"
RUN_NAME="${RUN_NAME:-regmix-10b-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${RUN_DIR:-/scratch/users/${SUNET}/agent-runs/${RUN_NAME}}"
EDULLM_ROOT="${EDULLM_ROOT:-/scratch/users/${SUNET}/agent-runs/edullm-farmshare-staging}"
SRC_RUN_DIR="${SRC_RUN_DIR:?set SRC_RUN_DIR to the local olmohq run to sample from}"
SEED="${SEED:-42}"
DOMAIN_LIST="${DOMAIN_LIST:-dclm arxiv starcoder pes2o open-web-math algebraic-stack wiki}"

mkdir -p "${RUN_DIR}/scripts" "${RUN_DIR}/logs" "${RUN_DIR}/data" "${RUN_DIR}/plan" "${RUN_DIR}/trim"
cd "${RUN_DIR}"

# Sync pipeline scripts into the isolated run dir.
REGMIX_ROOT="${EDULLM_ROOT}/datasets/regmix"
DATASETS_SHARED="${EDULLM_ROOT}/datasets"
cp -a "${REGMIX_ROOT}/plan_regmix_mix.py" "${RUN_DIR}/scripts/"
cp -a "${REGMIX_ROOT}/finalize_regmix_upload.py" "${RUN_DIR}/scripts/"
cp -a "${REGMIX_ROOT}/trim_regmix_domain.sbatch" "${RUN_DIR}/scripts/"
cp -a "${DATASETS_SHARED}/trim_and_tokenize_regmix.py" "${RUN_DIR}/scripts/"
cp -a "${DATASETS_SHARED}/olmo_shard_utils.py" "${RUN_DIR}/scripts/"

# Venv
if [[ ! -x "${RUN_DIR}/venv/bin/python" ]]; then
  python3 -m venv "${RUN_DIR}/venv"
fi
# shellcheck disable=SC1091
source "${RUN_DIR}/venv/bin/activate"
pip install -U pip wheel
pip install tqdm transformers zstandard

export EDULLM_ROOT RUN_DIR

echo "hardlinking ${SRC_RUN_DIR}/data -> ${RUN_DIR}/data"
cp -al "${SRC_RUN_DIR}/data/." "${RUN_DIR}/data/" 2>/dev/null || cp -a "${SRC_RUN_DIR}/data/." "${RUN_DIR}/data/"

POOL_SUMMARY="${RUN_DIR}/plan/pool_summary.json"
cp -a "${SRC_RUN_DIR}/plan/summary.json" "${POOL_SUMMARY}" 2>/dev/null || true

python "${RUN_DIR}/scripts/plan_regmix_mix.py" \
  --local-data-root "${SRC_RUN_DIR}" \
  --seed "${SEED}" \
  --out-dir "${RUN_DIR}/plan" \
  --pool-summary "${POOL_SUMMARY}"

MANIFEST="${RUN_DIR}/plan/manifest.jsonl"
SUMMARY="${RUN_DIR}/plan/summary.json"
NDOMS=$(echo "${DOMAIN_LIST}" | wc -w)

cat > "${RUN_DIR}/env.sh" <<EOF
RUN_DIR=${RUN_DIR}
VENV=${RUN_DIR}/venv
MANIFEST=${MANIFEST}
SUMMARY=${SUMMARY}
LOCAL_ROOT=${RUN_DIR}/data
SRC_RUN_DIR=${SRC_RUN_DIR}
DOMAIN_LIST="${DOMAIN_LIST}"
EDULLM_ROOT=${EDULLM_ROOT}
EOF

echo "manifest_files=$(wc -l < "${MANIFEST}")"
echo "domains=${NDOMS}"
echo "RUN_DIR=${RUN_DIR}" | tee "${RUN_DIR}/RUN_DIR.txt"
cat "${SUMMARY}"

# Per-domain document trim (shards are already local from the hardlink above).
TRIM_JOB=$(sbatch --parsable --exclude=wheat-01 \
  --array=0-$((NDOMS - 1)) \
  --chdir="${RUN_DIR}" \
  --export=ALL,RUN_DIR,VENV,MANIFEST,SUMMARY,DOMAIN_LIST \
  "${RUN_DIR}/scripts/trim_regmix_domain.sbatch")
echo "trim_job_id=${TRIM_JOB}"
echo "${TRIM_JOB}" > "${RUN_DIR}/trim_job_id.txt"

# Build the final manifest + local staging layout after all trims succeed.
FINAL_JOB=$(sbatch --parsable --exclude=wheat-01 \
  --partition=normal \
  --cpus-per-task=8 \
  --mem=32G \
  --time=06:00:00 \
  --dependency="afterok:${TRIM_JOB}" \
  --job-name=regmix-finalize \
  --chdir="${RUN_DIR}" \
  --output="${RUN_DIR}/logs/finalize-%j.out" \
  --error="${RUN_DIR}/logs/finalize-%j.err" \
  --wrap="set -Eeuo pipefail; source ${RUN_DIR}/env.sh; export EDULLM_ROOT RUN_DIR; source ${VENV}/bin/activate; python ${RUN_DIR}/scripts/finalize_regmix_upload.py --run-dir ${RUN_DIR}")
echo "finalize_job_id=${FINAL_JOB}"
echo "${FINAL_JOB}" > "${RUN_DIR}/finalize_job_id.txt"

echo "submitted trim=${TRIM_JOB} finalize=${FINAL_JOB}"
