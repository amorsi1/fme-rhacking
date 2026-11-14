#!/bin/bash
# Pull training outputs (checkpoints, rollouts, logs) from the VM to local.
# Optionally archives checkpoints to GCS for long-term storage.
#
# Usage:
#   bash infra/fetch.sh           # rsync from VM to local only
#   bash infra/fetch.sh --archive # also push checkpoints to GCS bucket
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/.."
source "${REPO_ROOT}/.env"

REMOTE_DIR="${GCP_REMOTE_DIR:-/home/${USER}/fme-rhacking}"
ARCHIVE=false
[[ "${1:-}" == "--archive" ]] && ARCHIVE=true

SSH_HOST=$(gcloud compute instances describe "${GCP_INSTANCE}" \
    --project="${GCP_PROJECT}" \
    --zone="${GCP_ZONE}" \
    --format="value(networkInterfaces[0].accessConfigs[0].natIP)")

SSH_KEY="${HOME}/.ssh/google_compute_engine"
SSH="ssh -i ${SSH_KEY} -o StrictHostKeyChecking=no"

echo "==> Fetching training outputs from ${GCP_INSTANCE}"

for dir in checkpoints rollouts logs; do
    echo "  ${dir}/"
    rsync -az --progress \
        -e "${SSH}" \
        "${USER}@${SSH_HOST}:${REMOTE_DIR}/${dir}/" \
        "${REPO_ROOT}/${dir}/"
done

if ${ARCHIVE}; then
    echo "==> Archiving checkpoints to gs://${GCP_BUCKET}/checkpoints"
    gsutil -m rsync -r \
        "${REPO_ROOT}/checkpoints" \
        "gs://${GCP_BUCKET}/checkpoints"
    echo "Archive complete: gs://${GCP_BUCKET}/checkpoints"
fi

echo "==> Fetch complete"
