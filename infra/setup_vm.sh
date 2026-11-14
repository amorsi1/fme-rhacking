#!/bin/bash
# One-time VM initialization. SSHs into the VM and installs all dependencies.
#
# Run after infra/provision.sh, once the VM is up and NVIDIA driver is ready.
# Usage: bash infra/setup_vm.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../.env"

REMOTE_DIR="${GCP_REMOTE_DIR:-/home/${USER}/fme-rhacking}"

echo "==> Waiting for SSH availability..."
until gcloud compute ssh "${GCP_INSTANCE}" \
    --project="${GCP_PROJECT}" \
    --zone="${GCP_ZONE}" \
    --command="echo ok" &>/dev/null; do
  sleep 10
done

echo "==> Running setup on ${GCP_INSTANCE}"
gcloud compute ssh "${GCP_INSTANCE}" \
    --project="${GCP_PROJECT}" \
    --zone="${GCP_ZONE}" <<REMOTE
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
mkdir -p "${REMOTE_DIR}/vendor"

# Clone verl (the only vendor dependency not in pyproject.toml).
if [ ! -d "${REMOTE_DIR}/vendor/Countdown-Code" ]; then
  git clone --filter=blob:none \
      https://github.com/zohaib-khan5040/Countdown-Code \
      "${REMOTE_DIR}/vendor/Countdown-Code"
  git -C "${REMOTE_DIR}/vendor/Countdown-Code" \
      checkout 170ee8139cd836d6ab7b7b3a70c2063906363db0
fi

echo "Setup complete. Run: bash infra/sync.sh"
REMOTE
