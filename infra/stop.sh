#!/bin/bash
# Stop the training VM to avoid idle GPU charges.
# Data on the boot disk is preserved; restart with:
#   gcloud compute instances start <instance> --zone=<zone>
#
# Usage: bash infra/stop.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../.env"

echo "==> Stopping ${GCP_INSTANCE} (data on disk is preserved)"
gcloud compute instances stop "${GCP_INSTANCE}" \
    --project="${GCP_PROJECT}" \
    --zone="${GCP_ZONE}"

echo "Restart:  gcloud compute instances start ${GCP_INSTANCE} --zone=${GCP_ZONE}"
echo "Then re-attach to any tmux session with: bash infra/train.sh ..."
