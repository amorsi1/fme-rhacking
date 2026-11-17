# FME-rHacking

FME project investigating reward hacking in post-training.

Built on [Countdown-Code](https://github.com/zohaib-khan5040/Countdown-Code)
(Khalifa, Khan, Tafveez, Peng & Wang, [arXiv:2603.07084](https://arxiv.org/abs/2603.07084)),
a testbed that exposes two reward signals for the same rollout:

- **proxy reward** — does the model's own `test.py` pass when run against its
  `solution.py`? Tamperable, and the signal that drives training.
- **true reward** — does the extracted `expr` actually hit the target using each
  number once? Evaluated against the original problem, ignoring the model's test.

The gap between them is the specification-gaming rate.

## Setup

Requires Python 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync               # data generation and evaluation (macOS or Linux)
uv sync --extra gpu   # adds verl for GPU training (Linux only)
```

Upstream's `setup.sh` is not used; its dependencies are declared in
`pyproject.toml`.

## Layout

| Path | Contents |
|---|---|
| `src/countdown_code/` | The verifiers environment: prompts, parser, rubric, reward functions |
| `src/rHacking/reward_managers/` | verl reward manager computing true + proxy rewards during RL |
| `datagen/` | Dataset split generation, distillation, SFT filtering, RL formatting |
| `configs/sft/`, `configs/rlvr/` | Training launch scripts |
| `data/` | All generated artifacts (gitignored) |
| `vendor/` | Optional upstream clone (gitignored) |

Everything under `src/`, `datagen/` and `configs/` is ours to edit — modify it
and commit normally. See `CLAUDE.md` for where we diverge from upstream.

## 1. Data generation

The dataset is `Jiayi-Pan/Countdown-Tasks-3to4`, split into three disjoint
16k-example subsets: distillation, evaluation, and RLVR.

```bash
cd datagen
uv run python create_datasets.py
```

Writes `data/{distillation_dataset,eval_dataset,rlvr_dataset}.jsonl`. The
scripts use flat imports, so run them from inside `datagen/`; their *output*
always lands in `<repo>/data/` regardless.

### SFT data: distillation traces

Filter upstream's pre-generated 16k traces for the hacking/non-hacking mix and
emit SFT parquet:

```bash
cd datagen
uv run python filtering_proxy.py
```

The traces (75 MB) are downloaded from upstream's pinned commit on first use
and cached at `data/o4-mini-distillation-16k.jsonl`; no clone of Countdown-Code
is needed. To pre-fetch them on their own, or re-download after a failure:

```bash
cd datagen
uv run python distillation.py            # --force to overwrite the cache
```

Writes `data/sft_dataset.jsonl` and `data/sft/{train,test}.parquet`, keeping
every trace whose proxy reward fired — tampered tests included, which is the
point. The run reports the true-reward mean and the proxy-only share of the
training mix (currently 15599 of 16000 traces kept, 2.57% proxy-only).

To use your own traces instead of upstream's, drop them at that same path (or
pass a path to `load_distillation_traces`); the checksum is only enforced on
files this repo downloads.

### RLVR data

```bash
cd datagen
uv run python format_for_rl.py
```

Writes `data/rlvr/{train,test}.parquet`, carrying the `extra_info` the reward
manager needs to compute both rewards.

## 2. Evaluation: measuring the hacking rate

Serve a model, then evaluate against the environment. `vf-eval` ships with
verifiers, so no extra CLI is needed:

```bash
uv run vllm serve "Qwen/Qwen3-8B" --max-model-len 2048 --port 8000 --tensor-parallel-size 1
```

```bash
uv run vf-eval countdown-code \
    -m "Qwen/Qwen3-8B" \
    -b http://localhost:8000/v1 \
    -k OPENAI_API_KEY \
    -n 1000 -r 8 -c 128 -T 1.0 -s
```

`-k` names the *environment variable* holding the API key, not the key itself.
It defaults to `PRIME_API_KEY`; a local vllm server ignores the value but the
client still requires one to be present, so point it at any set variable.
`-s` saves the per-rollout results, which is what you want for computing
hacking rates afterwards.

Upstream's Prime CLI equivalent (`prime eval run countdown-code ...`) also works
if you have `prime` installed.

Reported metrics:

| Metric | Meaning |
|---|---|
| `reward` | Scalar driving RL/eval; equals `run_execution_check` (weight 1.0) |
| `run_execution_check` | **Proxy / hackable** — the model's own test passed |
| `run_equation_check` | **True** — `expr` genuinely solves the problem |

Environment arguments (`dataset_name`, `local_file`, `num_samples`,
`test_size`) are passed through `load_environment`; see
`src/countdown_code/countdown_code.py`.

## 3. Training

Training needs GPUs and Linux. Add the `gpu` extra to pull in `verl`:

```bash
uv sync --extra gpu
```

This installs `verl` from the pinned upstream commit automatically — no vendor
clone needed. Upstream's `setup.sh` is the reference for the surrounding CUDA
stack (PyTorch 2.6 / CUDA 12.8 / Flash Attention 2.7.4).

Run the launch scripts **from the repo root**, so `$(pwd)/data/...` resolves:

```bash
bash configs/sft/run_qwen2.5-coder-3b-fft.sh      # or the 7b-lora variant
```

Merge LoRA adapters after a LoRA run:

```bash
uv run python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir <trained_checkpoint> \
    --target_dir <output_dir> \
    --merge-lora
```

Then RLVR:

```bash
bash configs/rlvr/run_qwen2.5-coder-3b-fft.sh     # or the 7b-lora variant
```

The RLVR scripts register our reward manager with unmodified verl by pointing
`custom_reward_function.path` at `src/rHacking/reward_managers/countdown_code.py`;
`CLAUDE.md` explains the mechanism. If a run dies with
`Unknown reward manager: countdown_code`, that hook is the thing to look at.

## 4. GCP training

Training requires Linux + CUDA + verl, so it runs on a GCP VM.
The `infra/` directory contains six `gcloud`-based shell scripts — no Terraform needed.

**See [`infra/README.md`](infra/README.md) for the complete guide**, covering:
prerequisites, GCP quota setup, first-time VM provisioning, the day-to-day
launch/monitor/fetch/stop loop, resuming from checkpoint, cost control, and
troubleshooting.

Quick start (after completing the prerequisites in that guide):

```bash
cp .env.example .env              # fill in GCP_PROJECT, WANDB_API_KEY, etc.
bash infra/provision.sh           # create VM + GCS bucket (once)
bash infra/setup_vm.sh            # install uv on the VM (once)
bash infra/sync.sh                # push code and .env to the VM
bash infra/train.sh configs/rlvr/run_qwen2.5-coder-3b-fft.sh
bash infra/fetch.sh               # pull checkpoints/, rollouts/, logs/
bash infra/stop.sh                # stop VM to end GPU charges

# To run again after the VM has been stopped/terminated by stop.sh or auto-shutdown:
gcloud compute instances start ${GCP_INSTANCE} --zone=${GCP_ZONE}
bash infra/sync.sh && bash infra/train.sh configs/rlvr/run_qwen2.5-coder-3b-fft.sh
```

## Platform notes

`vllm` ships manylinux wheels only, so it is marked Linux-only and skipped on
macOS. Data generation, environment edits and reward-function work run fine on
a laptop; serving, evaluation rollouts and training need the Linux/CUDA box.

## Citation

```bibtex
@article{countdowncode,
      title={Countdown-Code: A Testbed for Studying The Emergence and Generalization of Reward Hacking in RLVR},
      author={Muhammad Khalifa and Zohaib Khan and Omer Tafveez and Hao Peng and Lu Wang},
      year={2026},
      eprint={2603.07084},
      archivePrefix={arXiv}
}
```
