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

# Fail fast if required variables are unset — the original bug that created a
# bucket in the wrong project was caused by empty GCP_PROJECT silently falling
# back to gcloud's default.
: "${GCP_PROJECT:?Set GCP_PROJECT in .env (see .env.example)}"
: "${GCP_ZONE:?Set GCP_ZONE in .env (see .env.example)}"
: "${GCP_BUCKET:?Set GCP_BUCKET in .env (see .env.example)}"
: "${GCP_INSTANCE:?Set GCP_INSTANCE in .env (see .env.example)}"

REGION="${GCP_ZONE%-*}"

echo "==> Creating GCS bucket gs://${GCP_BUCKET}"
if gcloud storage buckets list --project="${GCP_PROJECT}" \
        --format='value(name)' | grep -qx "${GCP_BUCKET}"; then
    echo "    Bucket already exists in ${GCP_PROJECT} — skipping"
elif gcloud storage buckets describe "gs://${GCP_BUCKET}" \
        --format='value(name)' &>/dev/null; then
    # The bucket exists but is owned by a different project (GCS names are a
    # single global namespace, so this is possible even across your own projects).
    echo "ERROR: gs://${GCP_BUCKET} exists but is owned by another project." >&2
    echo "       Bucket names are a global namespace. Delete it or pick a" >&2
    echo "       different GCP_BUCKET in .env, then re-run." >&2
    exit 1
else
    gcloud storage buckets create "gs://${GCP_BUCKET}" \
        --project="${GCP_PROJECT}" --location="${REGION}" \
        --uniform-bucket-level-access
    # Sanity-check: confirm the VM's service account can write.
    # Surfaces 403s at provision time rather than silently at shutdown.
    SA="${GCP_PROJECT}@appspot.gserviceaccount.com"
    echo "    Verifying write access for compute SA ..."
    echo probe | gcloud storage cp - "gs://${GCP_BUCKET}/_probe.txt" \
        --billing-project="${GCP_PROJECT}" 2>/dev/null \
        && gcloud storage rm "gs://${GCP_BUCKET}/_probe.txt" \
              --billing-project="${GCP_PROJECT}" 2>/dev/null \
        && echo "    Write access OK" \
        || echo "    WARNING: write probe failed — check IAM for the compute SA"
fi

echo "==> Setting up IAP + Cloud NAT (VM gets no external IP)"
bash "${SCRIPT_DIR}/network.sh"

echo "==> Creating VM ${GCP_INSTANCE}"
gcloud compute instances create "${GCP_INSTANCE}" \
    --project="${GCP_PROJECT}" \
    --zone="${GCP_ZONE}" \
    --machine-type="${GCP_MACHINE_TYPE:?Set GCP_MACHINE_TYPE in .env (see .env.example)}" \
    --image-family=common-cu129-ubuntu-2204-nvidia-580 \
    --image-project=deeplearning-platform-release \
    --boot-disk-size="${GCP_DISK_SIZE:-300GB}" \
    --boot-disk-type=pd-ssd \
    --maintenance-policy=TERMINATE \
    --no-restart-on-failure \
    --no-address \
    --scopes=cloud-platform \
    --metadata="install-nvidia-driver=True"

echo
echo "VM created. Next steps:"
echo "  1. Wait ~3 min for NVIDIA driver install (VM reaches the internet via Cloud NAT), then:"
echo "     bash infra/setup_vm.sh"
echo "  2. Push code:"
echo "     bash infra/sync.sh"
echo "  3. Push training data (required before launching a run):"
echo "     bash infra/sync.sh --with-data"
echo "  4. Launch training:"
echo "     bash infra/train.sh configs/rlvr/run_qwen2.5-coder-3b-fft.sh"
