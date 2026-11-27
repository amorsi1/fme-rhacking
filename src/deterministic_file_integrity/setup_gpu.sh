#!/usr/bin/env bash
set -euo pipefail

if [[ $(uname -s) != Linux || $(uname -m) != x86_64 ]]; then
    echo "Run this on the Linux x86_64 GPU VM." >&2
    exit 1
fi

cd "$(dirname "$0")/../.."
experiment=src/deterministic_file_integrity
venv="$PWD/$experiment/.venv"
uv --no-config venv --allow-existing --python 3.12 "$venv"
# Resolve transitive packages as of the training fork's commit date.
uv --no-config pip install --python "$venv/bin/python" \
    --exclude-newer 2026-03-11T00:00:00Z -r "$experiment/requirements-gpu.txt"
uv --no-config pip check --python "$venv/bin/python"

# Make this checkout importable from Ray workers as well as the driver.
"$venv/bin/python" - <<'PY'
from pathlib import Path
import site

Path(site.getsitepackages()[0], "countdown_checkout.pth").write_text(f"{Path.cwd() / 'src'}\n")
PY

VLLM_USE_V1=0 "$venv/bin/python" - <<'PY'
from importlib.metadata import version

import torch
from flash_attn.bert_padding import unpad_input
from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout
from deterministic_file_integrity.rollout_config import SeededRolloutConfig

if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see a CUDA GPU. Check nvidia-smi on the VM.")
print(f"GPU preflight passed: vLLM {version('vllm')}, torch {torch.__version__}, {torch.cuda.get_device_name()}")
PY
