#!/bin/bash
# Pull training outputs (checkpoints, rollouts, logs) from the VM or GCS to local.
#
# Usage:
#   bash infra/fetch.sh             # rsync directly from the running VM
#   bash infra/fetch.sh --from-gcs  # pull from GCS (use when VM has shut down)
#   bash infra/fetch.sh --archive   # rsync from VM AND push checkpoints to GCS
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/.."
source "${REPO_ROOT}/.env"

REMOTE_DIR="${GCP_REMOTE_DIR:-/home/${USER}/fme-rhacking}"
ARCHIVE=false
FROM_GCS=false
for arg in "$@"; do
    case "${arg}" in
        --archive)  ARCHIVE=true ;;
        --from-gcs) FROM_GCS=true ;;
    esac
done

if ${FROM_GCS}; then
    echo "==> Pulling training outputs from gs://${GCP_BUCKET}"
    for dir in checkpoints rollouts logs; do
        echo "  ${dir}/"
        gsutil -m rsync -r \
            "gs://${GCP_BUCKET}/${dir}" \
            "${REPO_ROOT}/${dir}/"
    done
    echo "==> Fetch complete"
    exit 0
fi

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
