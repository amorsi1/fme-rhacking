#!/bin/bash
# Launch a training script on the VM inside a named tmux session.
# Disconnect-safe: training keeps running after you close the terminal.
#
# When training finishes (success or failure):
#   On the VM:    outputs are synced to GCS, then the VM shuts down.
#   On your Mac:  a background watcher (fetch_watcher.sh) detects the shutdown
#                 and runs `fetch.sh --from-gcs` to pull results locally.
#
# The watcher log is at logs/fetch_watcher.log.
# To kill it early: kill $(cat logs/fetch_watcher.pid)
#
# Usage:
#   bash infra/train.sh <config-script> [extra verl args]
#
# Examples:
#   bash infra/train.sh configs/rlvr/run_qwen2.5-coder-3b-fft.sh
#   bash infra/train.sh configs/sft/run_qwen2.5-coder-3b-fft.sh 2 checkpoints/sft-3b
#   bash infra/train.sh configs/rlvr/run_qwen2.5-coder-7b-lora.sh
#
# To follow the live log without SSH:
#   bash infra/watch_log.sh   (polls GCS every 30 s)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

CONFIG_SCRIPT="${1:?Usage: train.sh <config-script> [extra args]}"
shift || true
EXTRA_ARGS="${*:-}"

SESSION="training"

echo "==> Launching '${CONFIG_SCRIPT}' on ${GCP_INSTANCE}"

"${GCLOUD_SSH[@]}" <<REMOTE
set -euo pipefail
source "\${HOME}/.local/bin/env"
cd "${REMOTE_DIR}"
source .env

# Kill any previous session with the same name so the script starts fresh.
tmux kill-session -t ${SESSION} 2>/dev/null || true
mkdir -p logs

tmux new-session -d -s ${SESSION} \
    "cd ${REMOTE_DIR} && source .env && source \${HOME}/.local/bin/env && \
     source ${REMOTE_DIR}/.venv/bin/activate && \
     bash ${CONFIG_SCRIPT} ${EXTRA_ARGS} 2>&1 | tee logs/training_\$(date +%Y%m%d_%H%M%S).log ; \
     bash ${REMOTE_DIR}/infra/post_training.sh"

# Start a GCS log-sync sidecar: copies the live log to GCS every 30 s so
# watch_log.sh can follow it without SSH or an IAP tunnel.
nohup bash -c 'while sleep 30; do
    LOG=\$(ls -t ${REMOTE_DIR}/logs/training_*.log 2>/dev/null | head -1)
    [[ -n "\$LOG" ]] && gsutil -q cp "\$LOG" "gs://${GCP_BUCKET}/logs/live.log" 2>/dev/null || true
done' >> "${REMOTE_DIR}/logs/logsync.log" 2>&1 &
echo "\$!" > "${REMOTE_DIR}/logs/logsync.pid"

echo ""
echo "Training started in tmux session '${SESSION}'."
echo "Live log: bash infra/watch_log.sh   (polls GCS every 30 s — no SSH needed)"
echo "Logs:     ${REMOTE_DIR}/logs/"
echo ""
echo "On completion the VM will sync outputs to gs://${GCP_BUCKET} and shut down."
REMOTE

# Kill any stale watcher from a previous run.
REPO_ROOT="${SCRIPT_DIR}/.."
PID_FILE="${REPO_ROOT}/logs/fetch_watcher.pid"
if [[ -f "${PID_FILE}" ]]; then
    OLD_PID=$(cat "${PID_FILE}")
    if kill -0 "${OLD_PID}" 2>/dev/null; then
        echo "==> Stopping previous fetch watcher (PID ${OLD_PID})"
        kill "${OLD_PID}" 2>/dev/null || true
    fi
    rm -f "${PID_FILE}"
fi

echo "==> Follow the live log (no SSH):  bash infra/watch_log.sh"
echo ""
mkdir -p "${REPO_ROOT}/logs"
nohup bash "${SCRIPT_DIR}/fetch_watcher.sh" \
    >> "${REPO_ROOT}/logs/fetch_watcher.log" 2>&1 &
WATCHER_PID=$!
disown "${WATCHER_PID}"
echo "${WATCHER_PID}" > "${PID_FILE}"
echo "==> Local watcher started (PID ${WATCHER_PID})."
echo "    Will fetch outputs automatically once VM shuts down."
echo "    Log: logs/fetch_watcher.log"
echo "    Kill: kill \$(cat logs/fetch_watcher.pid)"
