#!/bin/bash
# Push local code (and .env) to the VM, then install/update dependencies.
# Safe to run repeatedly — rsync is incremental.
#
# Excludes: .venv/, vendor/, data/, checkpoints/, rollouts/, wandb/ (large/generated).
# Includes: .env  (so the VM picks up WANDB_API_KEY and other secrets).
#
# Usage: bash infra/sync.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/.."
source "${REPO_ROOT}/.env"

REMOTE_DIR="${GCP_REMOTE_DIR:-/home/${USER}/fme-rhacking}"

# Resolve the SSH destination that gcloud would use.
SSH_HOST=$(gcloud compute instances describe "${GCP_INSTANCE}" \
    --project="${GCP_PROJECT}" \
    --zone="${GCP_ZONE}" \
    --format="value(networkInterfaces[0].accessConfigs[0].natIP)")

SSH_KEY="${HOME}/.ssh/google_compute_engine"

echo "==> Syncing code to ${GCP_INSTANCE} (${SSH_HOST})"
rsync -az --progress \
    --exclude='.venv/' \
    --exclude='vendor/' \
    --exclude='data/' \
    --exclude='checkpoints/' \
    --exclude='rollouts/' \
    --exclude='wandb/' \
    --exclude='logs/' \
    --exclude='__pycache__/' \
    --exclude='*.egg-info/' \
    --exclude='.git/' \
    --exclude='.context/' \
    -e "ssh -i ${SSH_KEY} -o StrictHostKeyChecking=no" \
    "${REPO_ROOT}/" \
    "${USER}@${SSH_HOST}:${REMOTE_DIR}/"

echo "==> Installing dependencies on VM"
gcloud compute ssh "${GCP_INSTANCE}" \
    --project="${GCP_PROJECT}" \
    --zone="${GCP_ZONE}" <<REMOTE
set -euo pipefail
source "\${HOME}/.local/bin/env"  # make uv available
cd "${REMOTE_DIR}"
uv sync --extra gpu
REMOTE

echo "==> Sync complete"
