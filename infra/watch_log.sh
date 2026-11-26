#!/bin/bash
# Follow the live training log uploaded to GCS by the VM — no SSH or IAP needed.
#
# train.sh starts a sidecar on the VM that copies the training log to
# gs://$GCP_BUCKET/logs/live.log every 30 s.  This script polls that object
# with byte-range reads so only new bytes are downloaded each cycle.
#
# Usage:  bash infra/watch_log.sh
# Stop:   Ctrl-C
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../.env"

GCS_LOG="gs://${GCP_BUCKET}/logs/live.log"
POLL=30  # seconds between polls

echo "==> Live log: ${GCS_LOG}"
echo "    Polls every ${POLL}s — expect up to ${POLL}s of lag. Ctrl-C to stop."
echo ""

OFFSET=0
WAITED=0

while true; do
    sleep "${POLL}"

    SIZE=$(gsutil stat "${GCS_LOG}" 2>/dev/null \
        | awk '/Content-Length:/{print $2}') || SIZE=""

    if [[ -z "${SIZE}" ]]; then
        WAITED=$((WAITED + POLL))
        # Print a waiting notice once per minute so it's clear the script is alive.
        if (( WAITED % 60 == 0 )); then
            echo "[waiting for log to appear in GCS — ${WAITED}s elapsed]" >&2
        fi
        continue
    fi
    WAITED=0

    if (( SIZE > OFFSET )); then
        # Fetch only bytes we haven't seen yet.
        gsutil cat -r "${OFFSET}-$((SIZE - 1))" "${GCS_LOG}" 2>/dev/null
        OFFSET=${SIZE}
    fi
done
