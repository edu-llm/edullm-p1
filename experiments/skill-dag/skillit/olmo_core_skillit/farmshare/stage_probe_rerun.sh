#!/usr/bin/env bash
# Stage the 370M probe-arm rerun on FarmShare (login node, no GPU, no credentials).
#
# Copies an already staged and patched bundle (arms 0-6, see
# patch_static_validation_arms.py) to a new RUN_DIR, swaps in the rebuilt offline matrix
# with patch_probe_matrix.py, and checks that exactly the recipe and skillit_math.py
# differ from the source bundle. The source bundle is only read.
#
# Required: SRC_BUNDLE (e.g. .../static-validation-20260926-210344),
#           RUN_DIR (new, must not exist),
#           SCRIPTS_SRC (directory holding patch_probe_matrix.py,
#                        farmshare_probe_rerun_l40s.sbatch,
#                        farmshare_preflight_submit_probe_rerun.sh),
#           A_JSON (the rebuilt A_offline.json).
set -Eeuo pipefail

SRC_BUNDLE="${SRC_BUNDLE:?SRC_BUNDLE is required}"
RUN_DIR="${RUN_DIR:?RUN_DIR is required}"
SCRIPTS_SRC="${SCRIPTS_SRC:?SCRIPTS_SRC is required}"
A_JSON="${A_JSON:?A_JSON is required}"

[[ -d "${SRC_BUNDLE}/OLMo-core/.edullm" ]] || { echo "not a staged bundle: ${SRC_BUNDLE}" >&2; exit 2; }
[[ ! -e "${RUN_DIR}" ]] || { echo "refusing existing RUN_DIR=${RUN_DIR}" >&2; exit 2; }
[[ -f "${A_JSON}" ]] || { echo "missing ${A_JSON}" >&2; exit 2; }
for f in patch_probe_matrix.py farmshare_probe_rerun_l40s.sbatch farmshare_preflight_submit_probe_rerun.sh; do
  [[ -f "${SCRIPTS_SRC}/${f}" ]] || { echo "missing ${SCRIPTS_SRC}/${f}" >&2; exit 2; }
done

mkdir -p "${RUN_DIR}"/{scripts,logs,runs}
rsync -a --exclude __pycache__ "${SRC_BUNDLE}/OLMo-core/" "${RUN_DIR}/OLMo-core/"
ln -s "$(readlink -f "${SRC_BUNDLE}/manifest")" "${RUN_DIR}/manifest"
cp -a "${SRC_BUNDLE}/hf-cache" "${RUN_DIR}/hf-cache"
cp "${SCRIPTS_SRC}/patch_probe_matrix.py" "${SCRIPTS_SRC}/farmshare_probe_rerun_l40s.sbatch" \
  "${SCRIPTS_SRC}/farmshare_preflight_submit_probe_rerun.sh" "${RUN_DIR}/scripts/"
chmod +x "${RUN_DIR}/scripts/"*.sh "${RUN_DIR}/scripts/"*.sbatch
cp "${A_JSON}" "${RUN_DIR}/A_offline.json"

python3 "${RUN_DIR}/scripts/patch_probe_matrix.py" \
  --edullm-dir "${RUN_DIR}/OLMo-core/.edullm" \
  --a-offline-json "${RUN_DIR}/A_offline.json" | tee "${RUN_DIR}/logs/patch_probe_matrix.out"

# Exactly the recipe and skillit_math.py may differ from the source bundle.
changed="$({ diff -rq --exclude __pycache__ "${SRC_BUNDLE}/OLMo-core" "${RUN_DIR}/OLMo-core" || true; } | sort)"
expected="$(printf '%s\n' \
  "Files ${SRC_BUNDLE}/OLMo-core/.edullm/skillit_math.py and ${RUN_DIR}/OLMo-core/.edullm/skillit_math.py differ" \
  "Files ${SRC_BUNDLE}/OLMo-core/.edullm/skillit_recipe.json and ${RUN_DIR}/OLMo-core/.edullm/skillit_recipe.json differ" | sort)"
if [[ "${changed}" != "${expected}" ]]; then
  echo "unexpected differences from the source bundle:" >&2
  echo "${changed}" >&2
  exit 3
fi

{
  echo "staged_on=$(date -Is)"
  echo "source_bundle=${SRC_BUNDLE}"
  echo "input_manifest=$(readlink -f "${RUN_DIR}/manifest")/ready.json"
  echo "changes=skillit_recipe.json (skillit.offline_a, offline_a_source_sha256) and skillit_math.py (two pins) only"
  sha256sum "${A_JSON}" "${RUN_DIR}/A_offline.json" "${RUN_DIR}/OLMo-core/.edullm/skillit_recipe.json" \
    "${RUN_DIR}/OLMo-core/.edullm/skillit_math.py"
} > "${RUN_DIR}/staging_provenance.txt"
cat "${RUN_DIR}/staging_provenance.txt"
echo "staged ${RUN_DIR}"
