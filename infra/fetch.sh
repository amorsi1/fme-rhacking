#!/bin/bash
# Pull training outputs (checkpoints, rollouts, logs) from the VM or GCS to local.
#
# Usage:
#   bash infra/fetch.sh             # rsync from the running VM (over IAP; slow for large checkpoints — prefer --from-gcs)
#   bash infra/fetch.sh --from-gcs  # pull from GCS (use when VM has shut down)
#   bash infra/fetch.sh --archive   # rsync from VM AND push checkpoints to GCS
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/.."
source "${SCRIPT_DIR}/common.sh"

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

echo "==> Fetching training outputs from ${GCP_INSTANCE}"

for dir in checkpoints rollouts logs; do
    echo "  ${dir}/"
    rsync -az --progress \
        -e "${RSYNC_SSH}" \
        "${SSH_USER}@${GCP_INSTANCE}:${REMOTE_DIR}/${dir}/" \
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
