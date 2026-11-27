# shellcheck shell=bash
# Sourced by train.sh inside the tmux session, before the training script.
#
# A login shell on the GCP VM image loads /etc/profile.d/nccl_env.sh, which
# exports NCCL settings for A3 Ultra (H200, gIB/RoCE): NCCL_NET=gIB, a tuner
# config, LD_LIBRARY_PATH=/usr/local/gib/lib64, and more. train.sh feeds its
# script to ssh on stdin, which gives a login shell, and tmux inherits it. On
# the single L4 NCCL then tries to bring up the gIB plugin and the Ray
# WorkerDict dies silently during FSDP init (no traceback, no OOM). A
# non-login `ssh -- 'bash -s'` never saw these, so it trained fine.
#
# Nothing here targets multi-GPU/multi-node tuning we rely on, so drop it all.
while IFS= read -r _v; do unset "${_v}"; done < <(compgen -e | grep '^NCCL_' || true)
unset _v
LD_LIBRARY_PATH="$(printf '%s' "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v '^/usr/local/gib' | paste -sd: - || true)"
export LD_LIBRARY_PATH
