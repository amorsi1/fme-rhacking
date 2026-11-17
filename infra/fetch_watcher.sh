#!/bin/bash
# Runs in the background on your local machine (launched by train.sh).
# Polls GCP until the training VM reaches TERMINATED, then fetches
# all training outputs from GCS automatically.
#
# Log: logs/fetch_watcher.log in the repo root.
# Kill: kill $(cat logs/fetch_watcher.pid)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../.env"

REPO_ROOT="${SCRIPT_DIR}/.."
PID_FILE="${REPO_ROOT}/logs/fetch_watcher.pid"
POLL_INTERVAL=120  # seconds between GCP status checks

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

cleanup() {
    rm -f "${PID_FILE}"
}
trap cleanup EXIT

log "Watching ${GCP_INSTANCE} — will fetch once it reaches TERMINATED."
log "Poll interval: ${POLL_INTERVAL}s.  Kill: kill \$(cat logs/fetch_watcher.pid)"

while true; do
    sleep "${POLL_INTERVAL}"

    STATUS=$(gcloud compute instances describe "${GCP_INSTANCE}" \
        --project="${GCP_PROJECT}" \
        --zone="${GCP_ZONE}" \
        --format="value(status)" 2>/dev/null || echo "UNKNOWN")

    log "VM status: ${STATUS}"

    if [[ "${STATUS}" == "TERMINATED" ]]; then
        log "VM terminated. Running fetch.sh --from-gcs ..."
        if bash "${SCRIPT_DIR}/fetch.sh" --from-gcs; then
            log "Fetch complete."
        else
            log "ERROR: fetch.sh --from-gcs failed (see above). Outputs may still be on the VM disk."
            log "       Rescue: bash infra/fetch.sh  (direct rsync over IAP after restarting the VM)"
            exit 1
        fi
        break
    fi
done
