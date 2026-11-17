#!/bin/bash
# Called automatically by train.sh on training exit (success or failure).
# Syncs checkpoints, rollouts, and logs to GCS, then shuts down the VM.
# Not meant to be run directly — it is invoked by train.sh inside tmux.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../.env"
REMOTE_DIR="${GCP_REMOTE_DIR:-/home/${USER}/fme-rhacking}"

mkdir -p "${REMOTE_DIR}/logs"
LOG="${REMOTE_DIR}/logs/post_training.log"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "${LOG}"; }

log "==> Syncing training outputs to gs://${GCP_BUCKET}"

SYNC_FAILED=false

for dir in checkpoints rollouts logs; do
    src="${REMOTE_DIR}/${dir}"

    # Skip directories that don't exist or are empty — normal for eval-only runs.
    if [[ ! -d "${src}" ]] || [[ -z "$(ls -A "${src}" 2>/dev/null)" ]]; then
        log "    ${dir}/ — empty or absent, skipping"
        continue
    fi

    log "    ${dir}/"
    if ! gcloud storage rsync -r "${src}" "gs://${GCP_BUCKET}/${dir}" \
            2>&1 | tee -a "${LOG}"; then
        log "    ERROR: gcloud storage rsync failed for ${dir}/"
        SYNC_FAILED=true
    fi
done

if ${SYNC_FAILED}; then
    log "==> GCS sync FAILED for one or more directories."
    log "    VM is being left RUNNING so the data can be rescued manually."
    log "    Rescue: bash infra/fetch.sh  (direct rsync over IAP)"
    log "    Then stop: bash infra/stop.sh"
    exit 1
fi

log "==> GCS sync complete."
log "    Shutting down VM in 60 s.  To cancel: sudo shutdown -c"
sudo shutdown -h +1
