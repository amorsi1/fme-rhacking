#!/bin/bash
# Create the GCP VM, persistent SSD, and GCS bucket.
# Run once; safe to re-run (existing resources are skipped).
#
# Prerequisites:
#   gcloud auth login && gcloud auth application-default login
#   cp .env.example .env  # then fill in .env
#
# Usage: bash infra/provision.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../.env"

echo "==> Creating GCS bucket gs://${GCP_BUCKET}"
gsutil mb -p "${GCP_PROJECT}" \
    -l "${GCP_ZONE%%-*}" \
    "gs://${GCP_BUCKET}" 2>/dev/null \
    || echo "    Bucket already exists — skipping"

echo "==> Creating VM ${GCP_INSTANCE}"
gcloud compute instances create "${GCP_INSTANCE}" \
    --project="${GCP_PROJECT}" \
    --zone="${GCP_ZONE}" \
    --machine-type="${GCP_MACHINE_TYPE:-g2-standard-24}" \
    --image-family=common-cu129-ubuntu-2204-nvidia-580 \
    --image-project=deeplearning-platform-release \
    --boot-disk-size="${GCP_DISK_SIZE:-300GB}" \
    --boot-disk-type=pd-ssd \
    --maintenance-policy=TERMINATE \
    --no-restart-on-failure \
    --scopes=cloud-platform \
    --metadata="install-nvidia-driver=True"

echo
echo "VM created. Next steps:"
echo "  1. Wait ~3 min for NVIDIA driver install, then:"
echo "     bash infra/setup_vm.sh"
echo "  2. Push code and data:"
echo "     bash infra/sync.sh"
echo "  3. Launch training:"
echo "     bash infra/train.sh configs/rlvr/run_qwen2.5-coder-3b-fft.sh"
