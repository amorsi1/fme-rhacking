# Shared setup for infra/*.sh — source it, don't execute it.
# Loads .env and defines the IAP-based SSH helpers. The VM has no external IP
# (org policy constraints/compute.vmExternalIpAccess), so every connection is
# tunnelled through Identity-Aware Proxy.

_INFRA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_INFRA_DIR}/../.env"

REMOTE_DIR="${GCP_REMOTE_DIR:-/home/${USER}/fme-rhacking}"
SSH_USER="${GCP_SSH_USER:-${USER}}"   # override if you use OS Login (different remote username)

# Interactive / scripted ssh: "${GCLOUD_SSH[@]}" [-- cmd]  or  "${GCLOUD_SSH[@]}" <<EOF ... EOF
GCLOUD_SSH=(gcloud compute ssh "${GCP_INSTANCE}"
    --project="${GCP_PROJECT}" --zone="${GCP_ZONE}" --tunnel-through-iap)

# rsync transport: plain ssh whose connection goes through an IAP tunnel.
# Needs ~/.ssh/google_compute_engine to be authorised on the VM, which the first
# `gcloud compute ssh` (setup_vm.sh) takes care of.
RSYNC_SSH="ssh -i ${HOME}/.ssh/google_compute_engine \
-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
-o 'ProxyCommand=gcloud compute start-iap-tunnel ${GCP_INSTANCE} 22 --listen-on-stdin --project=${GCP_PROJECT} --zone=${GCP_ZONE}'"
