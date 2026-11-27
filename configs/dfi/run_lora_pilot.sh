#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
source src/deterministic_file_integrity/.venv/bin/activate
export VLLM_USE_V1=0 NCCL_NET_PLUGIN=none NCCL_NET=Socket
export INIT_MODEL="${INIT_MODEL:-Qwen/Qwen2.5-Coder-3B-Instruct}"

exec bash configs/dfi/train.sh integrity 42 lora-pilot \
    trainer.n_gpus_per_node=1 trainer.total_training_steps=10 \
    trainer.val_before_train=false trainer.test_freq=-1 trainer.save_freq=-1 \
    data.train_batch_size=2 \
    actor_rollout_ref.model.lora_rank=256 actor_rollout_ref.model.lora_alpha=256 \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.n=2 actor_rollout_ref.rollout.gpu_memory_utilization=0.6
