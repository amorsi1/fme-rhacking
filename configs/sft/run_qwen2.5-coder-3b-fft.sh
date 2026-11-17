#!/bin/bash
# Adapted from Countdown-Code by Zohaib Khan et al.
# https://github.com/zohaib-khan5040/Countdown-Code @ 170ee81
# Upstream ships no LICENSE file; used here for internal research under the
# original authors' copyright. See CLAUDE.md for local modifications.
set -x

# Run from the repository root: paths below resolve against $(pwd), and the
# datagen scripts write their parquet output under <repo>/data/.
PROJECT_DIR="$(pwd)"

# Suppress set -x tracing while loading secrets so they never appear in logs.
{ set +x; } 2>/dev/null
if [ -f "${PROJECT_DIR}/.env" ]; then
    set -a; source "${PROJECT_DIR}/.env"; set +a
fi
set -x

PROJECT_NAME="${WANDB_PROJECT:-verl_reasoning_safety}"
TRAIN_PATH="${PROJECT_DIR}/data/sft/train.parquet"
TEST_PATH="${PROJECT_DIR}/data/sft/test.parquet"

MODEL_PATH="Qwen/Qwen2.5-Coder-3B-Instruct"
EXPERIMENT_NAME="sft_qwen25_coder_7b_orig"

if [ "$#" -lt 2 ]; then
    echo "Usage: run_qwen2.5-coder-3b-lora.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2

# Shift the arguments so $@ refers to the rest
shift 2

# data/ is gitignored and excluded by infra/sync.sh, so a fresh VM has no parquet
# files at all. Generate whatever is missing before verl starts, rather than
# discovering it after the model has been pulled. SKIP_DATA_CHECK=1 bypasses it.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/ensure_data.sh"
ensure_sft_data

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$TRAIN_PATH \
    data.val_files=$TEST_PATH \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    optim.lr=1e-4 \
    data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.micro_batch_size_per_gpu=64 \
    model.partial_pretrain=$MODEL_PATH \
    trainer.default_local_dir=$save_path \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.logger=console \
    trainer.total_epochs=5 \
    model.strategy=fsdp \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=true \
    $@