#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || ! $1 =~ ^(baseline|integrity)$ || ! $2 =~ ^[0-9]+$ || ! $3 =~ ^[a-zA-Z0-9_.-]+$ ]]; then
    echo "Usage: $0 {baseline|integrity} SEED INIT_TAG [verl overrides...]" >&2
    exit 2
fi
condition=$1
seed=$2
run="file-integrity-$3-$condition-s$seed"
shift 3
enforce=false
if [[ "$condition" == integrity ]]; then
    enforce=true
fi

cd "$(dirname "$0")/../.."
data_dir="$PWD/data/deterministic_file_integrity/rlvr"
run_dir="deterministic_file_integrity/$run"
export PATH="${VIRTUAL_ENV:-$PWD/.venv}/bin:$PATH"
export WANDB_DIR="$PWD/outputs/$run_dir"
# The shared launcher's data guard targets data/rlvr. This experiment prepares its own parquet.
SKIP_DATA_CHECK=1 exec bash configs/rlvr/run_qwen2.5-coder-3b-fft.sh \
    "data.train_files=$data_dir/train.parquet" "data.val_files=$data_dir/test.parquet" \
    "actor_rollout_ref.model.path=${INIT_MODEL:?set the baseline snapshot}" \
    actor_rollout_ref.rollout._target_=deterministic_file_integrity.rollout_config.SeededRolloutConfig \
    "+actor_rollout_ref.rollout.seed=$seed" "+reward_model.reward_kwargs.enforce_file_integrity=$enforce" \
    "trainer.experiment_name=$run" "trainer.default_local_dir=checkpoints/$run_dir" \
    "trainer.rollout_data_dir=rollouts/$run_dir/train" "trainer.validation_data_dir=rollouts/$run_dir/val" \
    "hydra.run.dir=outputs/$run_dir" "$@"
