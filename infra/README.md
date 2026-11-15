# GCP Training Guide

Step-by-step instructions for running verl training on GCP.
Training requires Linux + CUDA, so it cannot run on a Mac — this guide
covers how to provision a cloud GPU VM, run a job, and get results back.

## Table of contents

1. [Local prerequisites](#1-local-prerequisites)
2. [GCP project prerequisites](#2-gcp-project-prerequisites)
3. [Configure secrets](#3-configure-secrets)
4. [First-time VM setup](#4-first-time-vm-setup)
5. [Day-to-day workflow](#5-day-to-day-workflow)
6. [Monitoring a run](#6-monitoring-a-run)
7. [Retrieving results](#7-retrieving-results)
8. [Resuming after interruption](#8-resuming-after-interruption)
9. [Cost control](#9-cost-control)
10. [Troubleshooting](#10-troubleshooting)
11. [VM sizing reference](#11-vm-sizing-reference)

---

## 1. Local prerequisites

Install these tools on your laptop before anything else.

**Google Cloud SDK** (includes `gcloud` and `gsutil`):

```bash
# macOS
brew install --cask google-cloud-sdk

# or the universal installer
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
```

Authenticate and set your default project:

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

**Check versions** (any recent version works):

```bash
gcloud --version   # Google Cloud SDK 460+
gsutil --version   # gsutil version 5+
```

---

## 2. GCP project prerequisites

Do this once per GCP project, in the GCP console or with `gcloud`.

### Enable required APIs

```bash
gcloud services enable compute.googleapis.com storage.googleapis.com
```

### GPU quota

New GCP projects have zero GPU quota. You must request it:

1. Go to **IAM & Admin → Quotas** in the console.
2. Filter by **Metric: `compute.googleapis.com/nvidia_l4_gpus`** (or `a100` for A100s).
3. Filter by the **Region** matching your `GCP_ZONE` (e.g. `us-central1`).
4. Click **Edit Quotas**, request at least **2**, and submit.

Approval typically takes minutes to a few hours.  You can check status under
**IAM & Admin → Quotas → Quota increase requests**.

### Billing

Ensure billing is enabled on the project. GPU VMs cost money by the minute
while running — see [§9 Cost control](#9-cost-control) for how to avoid
surprises.

---

## 3. Configure secrets

Copy the example env file and fill in your values:

```bash
cp .env.example .env
```

Open `.env` and set:

| Variable | Where to get it |
|---|---|
| `GCP_PROJECT` | Your GCP project ID (console → project selector) |
| `GCP_ZONE` | Zone where you have GPU quota, e.g. `us-central1-a` |
| `GCP_INSTANCE` | Any name, e.g. `fme-train-1` |
| `GCP_MACHINE_TYPE` | `g2-standard-24` (2× L4) or `a2-highgpu-2g` (2× A100) |
| `GCP_DISK_SIZE` | `300GB` is enough for one experiment; increase for multiple |
| `GCP_BUCKET` | Globally unique bucket name, e.g. `yourname-fme-rhacking` |
| `GCP_REMOTE_DIR` | Path on the VM, e.g. `/home/YOUR_USERNAME/fme-rhacking` |
| `WANDB_API_KEY` | From [wandb.ai/settings](https://wandb.ai/settings) → Danger Zone → API keys |
| `WANDB_ENTITY` | Your W&B username or team name (optional) |
| `OPENAI_API_KEY` | Only needed for distillation data generation |

`.env` is gitignored — never commit it.

---

## 4. First-time VM setup

Run these once. After that, skip to [§5](#5-day-to-day-workflow).

### Step 1 — Provision the VM

```bash
bash infra/provision.sh
```

This creates:
- A `g2-standard-24` VM (2× L4 GPUs, 96 GB RAM, 300 GB SSD boot disk)
- A GCS bucket for checkpoint archival

Takes about 60 seconds.  The VM boots and begins installing the NVIDIA
driver in the background — allow **3–5 minutes** before the next step.

You can verify the VM is up:

```bash
gcloud compute instances list --filter="name=${GCP_INSTANCE}"
```

### Step 2 — Initialize the VM

```bash
bash infra/setup_vm.sh
```

This SSHs into the VM and:
1. Waits for the NVIDIA driver to finish installing.
2. Installs [uv](https://docs.astral.sh/uv/) (the Python package manager).
3. Clones the upstream verl repo (the only dependency not in `pyproject.toml`).

Takes 3–8 minutes depending on network. You will see driver wait messages
until `GPU ready: ...` prints.

### Step 3 — Push code

```bash
bash infra/sync.sh
```

This rsyncs your local source tree (excluding `.venv/`, `data/`, `vendor/`)
to the VM, then runs `uv sync` to install all Python dependencies including
`vllm` (Linux-only, so this is the first time it installs).

First sync takes 2–4 minutes while vllm downloads.  Subsequent syncs are
incremental and usually finish in under 10 seconds.

---

## 5. Day-to-day workflow

**The VM manages its own lifecycle.** You push code, start training, and walk
away.  When training finishes (or crashes), the VM automatically syncs all
outputs to GCS and shuts itself down — no need to monitor it or remember to
stop it.

```
[ laptop ]                                   [ GCP VM ]
                                                 |
bash infra/sync.sh       -- push code -->        |
bash infra/train.sh ...  -- start job  -->       | tmux: training
  └─ fetch_watcher.sh launched locally           | (runs unattended)
     (polls GCP every 2 min in background)       |
                                                 | training finishes
[ close terminal / do other work ]               | --> sync to GCS
                                                 | --> sudo shutdown
                                             (VM TERMINATED, billing ends)
  watcher detects TERMINATED
  --> fetch.sh --from-gcs runs automatically
  --> outputs land in checkpoints/, rollouts/, logs/
```

**What you must do before every run:**

1. `bash infra/sync.sh` — push your latest code and `.env` to the VM.
   Always do this; it also rsyncs your W&B key and config changes.
2. `bash infra/train.sh <config>` — launch training and walk away.

**What happens automatically:**

- On the VM: when training exits (any exit code), `post_training.sh` syncs
  `checkpoints/`, `rollouts/`, and `logs/` to `gs://$GCP_BUCKET`, then issues
  `sudo shutdown -h +1` (60-second grace period).
- On your Mac: `train.sh` starts `fetch_watcher.sh` as a background process.
  It polls GCP every 2 minutes; once the VM reaches `TERMINATED`, it runs
  `fetch.sh --from-gcs` and pulls everything to your local repo.
- Billing for the VM stops once the instance reaches TERMINATED state.

**What you must do after a run:** nothing — outputs arrive automatically.

If you need to fetch manually (e.g., the watcher was killed):

```bash
bash infra/fetch.sh --from-gcs
```

To monitor the watcher or kill it early:

```bash
tail -f logs/fetch_watcher.log       # follow watcher output
kill $(cat logs/fetch_watcher.pid)   # stop it
```

---

### Launching RLVR training

```bash
bash infra/train.sh configs/rlvr/run_qwen2.5-coder-3b-fft.sh
```

For the 7B LoRA variant:

```bash
bash infra/train.sh configs/rlvr/run_qwen2.5-coder-7b-lora.sh
```

### Launching SFT training

SFT scripts take two positional arguments: `nproc_per_node` and `save_path`.

```bash
bash infra/train.sh configs/sft/run_qwen2.5-coder-3b-fft.sh 2 checkpoints/sft-3b
```

### Passing extra verl arguments

Any arguments after the config path are forwarded directly to verl:

```bash
bash infra/train.sh configs/rlvr/run_qwen2.5-coder-3b-fft.sh \
    trainer.total_epochs=5 \
    trainer.save_freq=20
```

---

## 6. Monitoring a run

### Attach to the live log

Training runs inside a `tmux` session called `training`.  Attach from anywhere:

```bash
gcloud compute ssh ${GCP_INSTANCE} --zone=${GCP_ZONE} -- tmux attach -t training
```

Detach without killing the job: press **Ctrl-b d**.

The log is also written to `logs/training_<timestamp>.log` on the VM; `fetch.sh`
pulls it back.

### W&B dashboard

Once the first step completes, a run appears at
`https://wandb.ai/<WANDB_ENTITY>/verl_reasoning_safety`.

Key metrics to watch:

| Metric | Meaning |
|---|---|
| `train/reward_mean` | Average proxy reward (what drives training) |
| `train/run_execution_check` | Proxy reward — model's own test passed |
| `train/run_equation_check` | True reward — `expr` actually solves the problem |
| `train/cheating_rate` | Fraction of rollouts with proxy=1 / true=0 (hacking) |

The gap between `run_execution_check` and `run_equation_check` is the
specification-gaming signal this project measures.

### GPU utilisation on the VM

```bash
gcloud compute ssh ${GCP_INSTANCE} --zone=${GCP_ZONE} \
    -- watch -n2 nvidia-smi
```

---

## 7. Retrieving results

### After training completes (VM auto-stopped)

The VM syncs everything to GCS before shutting down.  Pull to local:

```bash
bash infra/fetch.sh --from-gcs
```

Pulls three directories from GCS into the repo root:

| Directory | Contents |
|---|---|
| `checkpoints/` | verl model checkpoints (one per `save_freq` steps) |
| `rollouts/` | Per-step rollout parquet files |
| `logs/` | Training stdout/stderr + `post_training.log` (sync status) |

### While training is still running (VM live)

```bash
bash infra/fetch.sh
```

Rsyncs directly from the VM over SSH.  Useful for inspecting a run mid-flight.

### Archive checkpoints to GCS from local

```bash
bash infra/fetch.sh --archive
```

Rsyncs from the VM AND pushes your local `checkpoints/` to GCS.  Useful
before deleting a VM manually.

---

## 8. Resuming after interruption

If a run dies (VM preempted, OOM, manual kill), verl can resume from the
last checkpoint.  Find the latest checkpoint step:

```bash
ls checkpoints/   # e.g. global_step_80/
```

Add `trainer.resume_from_checkpoint=checkpoints/global_step_80` to the
launch command:

```bash
bash infra/train.sh configs/rlvr/run_qwen2.5-coder-3b-fft.sh \
    trainer.resume_from_checkpoint=checkpoints/global_step_80
```

If the VM was stopped (not deleted), restart it first:

```bash
gcloud compute instances start ${GCP_INSTANCE} --zone=${GCP_ZONE}
bash infra/sync.sh   # re-sync in case you made local changes
```

---

## 9. Cost control

GPU VMs accrue charges every minute they are **running**.  Normally you do not
need to stop the VM manually — it shuts itself down at the end of every training
run (see [§5](#5-day-to-day-workflow)).

If you need to stop it early (e.g., to abort a run), SSH in and cancel the
pending shutdown first so the GCS sync can run:

```bash
gcloud compute ssh ${GCP_INSTANCE} --zone=${GCP_ZONE} \
    -- "sudo shutdown -c && tmux send-keys -t training C-c"
# Wait a moment, then stop manually once tmux exits:
bash infra/stop.sh
```

Or just kill it immediately (outputs already on the VM disk, not yet in GCS):

```bash
bash infra/stop.sh
```

The boot disk is preserved.  Restart:

```bash
gcloud compute instances start ${GCP_INSTANCE} --zone=${GCP_ZONE}
```

**Approximate costs** (on-demand, us-central1, Nov 2026):

| Machine type | GPUs | Cost per hour |
|---|---|---|
| `g2-standard-24` | 2× L4 (24 GB) | ~$1.80 |
| `a2-highgpu-2g` | 2× A100 40 GB | ~$5.90 |

A full 10-epoch RLVR run on the 3B model takes roughly 8–12 hours → ~$15–20.

**Set a billing alert** in the GCP console (Billing → Budgets & alerts) so
you get an email if spending exceeds a threshold.

---

## 10. Troubleshooting

### `gcloud compute ssh` times out

The VM may still be booting.  Wait 60 seconds and retry.  If it persists,
check the serial port log:

```bash
gcloud compute instances get-serial-port-output ${GCP_INSTANCE} \
    --zone=${GCP_ZONE} | tail -40
```

### `nvidia-smi: command not found` on the VM

The DLVM NVIDIA driver install is still running.  `setup_vm.sh` polls for
this automatically; if you SSHed in manually just wait:

```bash
watch -n5 nvidia-smi
```

### `Unknown reward manager: countdown_code`

The verl reward manager registration failed.  The most likely cause is that
`custom_reward_function.path` in the config does not resolve to an existing
file on the VM.  Check that `sync.sh` completed successfully and that
`src/rHacking/reward_managers/countdown_code.py` exists on the VM:

```bash
gcloud compute ssh ${GCP_INSTANCE} --zone=${GCP_ZONE} \
    -- ls ${GCP_REMOTE_DIR}/src/rHacking/reward_managers/
```

See `CLAUDE.md → verl reward manager registration` for the full mechanism.

### `Quota exceeded for quota metric`

Your project does not have enough GPU quota in the requested zone.  Request
more (see [§2](#2-gcp-project-prerequisites)) or change `GCP_ZONE` to a
zone where you do have quota.  Available GPU zones:

```bash
gcloud compute accelerator-types list --filter="name:nvidia-l4"
```

### W&B not logging

Check that `WANDB_API_KEY` is set correctly in `.env` and that `sync.sh` has
been run since you last edited `.env`.  On the VM:

```bash
gcloud compute ssh ${GCP_INSTANCE} --zone=${GCP_ZONE} \
    -- "source ${GCP_REMOTE_DIR}/.env && echo WANDB_API_KEY is set: \${WANDB_API_KEY:+yes}"
```

### OOM during training

Reduce micro-batch sizes in the config or switch to a machine with more VRAM.
Common flags to lower:

```
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2   # was 4
actor_rollout_ref.rollout.gpu_memory_utilization=0.6     # was 0.8
```

Pass them as extra args to `train.sh`.

---

## 11. VM sizing reference

| Use case | `GCP_MACHINE_TYPE` | GPUs | Total VRAM |
|---|---|---|---|
| 3B FFT, 7B LoRA | `g2-standard-24` | 2× L4 | 48 GB |
| 7B FFT | `a2-highgpu-2g` | 2× A100 40 GB | 80 GB |
| 7B FFT (faster) | `a2-highgpu-4g` | 4× A100 40 GB | 160 GB |

Change `GCP_MACHINE_TYPE` and `GCP_DISK_SIZE` in `.env`, then re-run
`infra/provision.sh` (delete the old VM first if switching machine type).

The verl config flag `trainer.n_gpus_per_node` must match the number of GPUs
on the machine.  It defaults to `2` in all configs; override via extra args
if you use a 4-GPU machine:

```bash
bash infra/train.sh configs/rlvr/run_qwen2.5-coder-3b-fft.sh \
    trainer.n_gpus_per_node=4
```
