#!/bin/bash
# One-time network setup for a VM with NO external IP (required when the org
# enforces constraints/compute.vmExternalIpAccess). Idempotent; called by
# provision.sh, but safe to run on its own.
#
#   1. IAP API + firewall rule  -> lets `gcloud compute ssh --tunnel-through-iap` reach port 22
#   2. Cloud Router + NAT       -> gives the VM outbound internet (pip, HF, W&B, GitHub)
#
# Usage: bash infra/network.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

NETWORK="${GCP_NETWORK:-default}"
REGION="${GCP_ZONE%-*}"   # us-central1-a -> us-central1
IAP_RANGE="35.235.240.0/20"

# ensure <description> <describe-cmd...> -- <create-cmd...>: create only if describe fails.
ensure() {
    local desc="$1"; shift
    local -a check=() create=()
    while [[ "$1" != "--" ]]; do check+=("$1"); shift; done; shift
    create=("$@")
    if "${check[@]}" &>/dev/null; then
        echo "    ${desc} exists — skipping"
    else
        echo "    Creating ${desc}"
        "${create[@]}"
    fi
}

echo "==> Enabling IAP API"
gcloud services enable iap.googleapis.com --project="${GCP_PROJECT}"

echo "==> Firewall: allow IAP -> tcp:22"
ensure "firewall rule allow-iap-ssh" \
    gcloud compute firewall-rules describe allow-iap-ssh --project="${GCP_PROJECT}" -- \
    gcloud compute firewall-rules create allow-iap-ssh --project="${GCP_PROJECT}" \
        --network="${NETWORK}" --allow=tcp:22 --source-ranges="${IAP_RANGE}"

echo "==> Cloud NAT in ${REGION}"
ensure "router fme-router" \
    gcloud compute routers describe fme-router --project="${GCP_PROJECT}" --region="${REGION}" -- \
    gcloud compute routers create fme-router --project="${GCP_PROJECT}" \
        --region="${REGION}" --network="${NETWORK}"
ensure "NAT fme-nat" \
    gcloud compute routers nats describe fme-nat --router=fme-router --project="${GCP_PROJECT}" --region="${REGION}" -- \
    gcloud compute routers nats create fme-nat --project="${GCP_PROJECT}" --region="${REGION}" \
        --router=fme-router --auto-allocate-nat-external-ips --nat-all-subnet-ip-ranges

echo "==> Network ready"
