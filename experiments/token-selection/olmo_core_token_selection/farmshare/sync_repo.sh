#!/usr/bin/env bash
# Sync a local OLMo-core worktree to FarmShare scratch (no secrets).
#
# Refuses a dirty tree, and stamps the synced .edullm/ with the exact commit
# that was synced (read by token_selection_entrypoint.py's git_commit() and
# logged into every run's W&B config and identity, closing the old
# git_commit=None gap).
set -Eeuo pipefail

: "${RUN_DIR:?}"
: "${LOCAL_REPO:?}"
: "${SOCK:?}"
: "${HOST:?}"

# COMMIT/DIRTY may be precomputed by the caller (e.g. on a host whose git
# can't resolve this worktree's .git pointer file, such as WSL against a
# Windows worktree) and passed in via env instead of recomputed here.
if [[ -n "${DIRTY:-}" ]]; then
  if [[ "${DIRTY}" != "0" ]]; then
    echo "refusing to sync a dirty tree at ${LOCAL_REPO}; commit first" >&2
    exit 2
  fi
elif [[ -n "$(git -C "${LOCAL_REPO}" status --porcelain)" ]]; then
  echo "refusing to sync a dirty tree at ${LOCAL_REPO}; commit first" >&2
  exit 2
fi
COMMIT="${COMMIT:-$(git -C "${LOCAL_REPO}" rev-parse HEAD)}"

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" \
  "mkdir -p '${RUN_DIR}/OLMo-core' '${RUN_DIR}/scripts' '${RUN_DIR}/logs' && chmod 700 '${RUN_DIR}'"

tar -C "${LOCAL_REPO}" -czf - pyproject.toml src .edullm | \
  ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" \
    "tar -xzf - -C '${RUN_DIR}/OLMo-core'"

tar -C "${LOCAL_REPO}/.edullm/farmshare" -czf - . | \
  ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" \
    "tar -xzf - -C '${RUN_DIR}/scripts'"

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" \
  "printf '%s' '${COMMIT}' > '${RUN_DIR}/OLMo-core/.edullm/GIT_COMMIT' && \
   find '${RUN_DIR}/scripts' -type f \( -name '*.sh' -o -name '*.sbatch' -o -name 'config.env' \) -exec sed -i 's/\r$//' {} + && \
   chmod +x '${RUN_DIR}/scripts'/*.sh 2>/dev/null || true"

echo "sync_ok run_dir=${RUN_DIR} commit=${COMMIT}"
