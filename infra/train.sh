#!/bin/bash
# Launch a training script on the VM inside a named tmux session.
# Disconnect-safe: training keeps running after you close the terminal.
#
# When training finishes (success or failure), the VM automatically:
#   1. Syncs checkpoints/, rollouts/, and logs/ to GCS (gs://$GCP_BUCKET)
#   2. Shuts itself down (60-second grace period)
#
# Retrieve results afterwards with:
#   bash infra/fetch.sh --from-gcs
#
# Usage:
#   bash infra/train.sh <config-script> [extra verl args]
#
# Examples:
#   bash infra/train.sh configs/rlvr/run_qwen2.5-coder-3b-fft.sh
#   bash infra/train.sh configs/sft/run_qwen2.5-coder-3b-fft.sh 2 checkpoints/sft-3b
#   bash infra/train.sh configs/rlvr/run_qwen2.5-coder-7b-lora.sh
#
# To attach to the running session later:
#   gcloud compute ssh <instance> --zone=<zone> -- tmux attach -t training
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../.env"

CONFIG_SCRIPT="${1:?Usage: train.sh <config-script> [extra args]}"
shift || true
EXTRA_ARGS="${*:-}"

REMOTE_DIR="${GCP_REMOTE_DIR:-/home/${USER}/fme-rhacking}"
SESSION="training"

echo "==> Launching '${CONFIG_SCRIPT}' on ${GCP_INSTANCE}"

gcloud compute ssh "${GCP_INSTANCE}" \
    --project="${GCP_PROJECT}" \
    --zone="${GCP_ZONE}" <<REMOTE
set -euo pipefail
source "\${HOME}/.local/bin/env"
cd "${REMOTE_DIR}"
source .env

# Kill any previous session with the same name so the script starts fresh.
tmux kill-session -t ${SESSION} 2>/dev/null || true
mkdir -p logs

tmux new-session -d -s ${SESSION} \
    "cd ${REMOTE_DIR} && source .env && source \${HOME}/.local/bin/env && \
     bash ${CONFIG_SCRIPT} ${EXTRA_ARGS} 2>&1 | tee logs/training_\$(date +%Y%m%d_%H%M%S).log ; \
     bash ${REMOTE_DIR}/infra/post_training.sh"

echo ""
echo "Training started in tmux session '${SESSION}'."
echo "Attach:  gcloud compute ssh ${GCP_INSTANCE} --zone=${GCP_ZONE} -- tmux attach -t ${SESSION}"
echo "Logs:    ${REMOTE_DIR}/logs/"
echo ""
echo "On completion the VM will sync outputs to gs://${GCP_BUCKET} and shut down."
echo "Retrieve results: bash infra/fetch.sh --from-gcs"
REMOTE
