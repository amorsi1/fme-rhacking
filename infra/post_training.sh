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

for dir in checkpoints rollouts logs; do
    log "    ${dir}/"
    gsutil -m rsync -r "${REMOTE_DIR}/${dir}" "gs://${GCP_BUCKET}/${dir}" \
        2>&1 | tee -a "${LOG}" \
        || log "    WARNING: gsutil rsync failed for ${dir}/ — continuing"
done

log "==> GCS sync complete."
if [[ "${NO_SHUTDOWN:-0}" == "1" ]]; then
    log "    NO_SHUTDOWN=1 — skipping shutdown."
else
    log "    Shutting down VM in 60 s.  To cancel: sudo shutdown -c"
    sudo shutdown -h +1
fi
