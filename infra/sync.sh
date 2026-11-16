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
source "${SCRIPT_DIR}/common.sh"

echo "==> Syncing code to ${GCP_INSTANCE} (via IAP)"
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
    -e "${RSYNC_SSH}" \
    "${REPO_ROOT}/" \
    "${SSH_USER}@${GCP_INSTANCE}:${REMOTE_DIR}/"

echo "==> Installing dependencies on VM"
"${GCLOUD_SSH[@]}" <<REMOTE
set -euo pipefail
source "\${HOME}/.local/bin/env"  # make uv available
cd "${REMOTE_DIR}"
uv sync --extra gpu
REMOTE

echo "==> Sync complete"
