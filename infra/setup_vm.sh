#!/bin/bash
# One-time VM initialization. SSHs into the VM and installs all dependencies.
#
# Run after infra/provision.sh, once the VM is up and NVIDIA driver is ready.
# Usage: bash infra/setup_vm.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"


echo "==> Waiting for SSH availability..."
until "${GCLOUD_SSH[@]}" \
    --command="echo ok" &>/dev/null; do
  sleep 10
done

echo "==> Running setup on ${GCP_INSTANCE}"
"${GCLOUD_SSH[@]}" <<REMOTE
set -euo pipefail

# Wait for the DLVM NVIDIA driver install to finish.
while ! nvidia-smi &>/dev/null; do
  echo "  Waiting for NVIDIA driver..."; sleep 15
done
echo "GPU ready:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Install uv if not present.
if ! command -v uv &>/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  source "\${HOME}/.local/bin/env"
fi

# Create repo directory and prepare for sync.
mkdir -p "${REMOTE_DIR}"

echo "Setup complete. Run: bash infra/sync.sh"
REMOTE
