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

| Variable | Where to get it | Suggested command |
|---|---|---|
| `GCP_PROJECT` | Your GCP project ID (console → project selector) | `gcloud config get-value project` |
| `GCP_ZONE` | Zone where you have GPU quota, e.g. `us-central1-a` | `gcloud config get-value compute/zone` (the configured default; empty if unset) — to find zones offering your GPU, see [below](#finding-the-right-values) |
| `GCP_INSTANCE` | Any name, e.g. `fme-train-1` | `gcloud compute instances list` (check the name isn't already taken) |
| `GCP_MACHINE_TYPE` | `g2-standard-24` (2× L4) or `a2-highgpu-2g` (2× A100) | — (see [§11](#11-vm-sizing-reference)) |
| `GCP_DISK_SIZE` | `300GB` is enough for one experiment; increase for multiple | — |
| `GCP_BUCKET` | Globally unique bucket name, e.g. `yourname-fme-rhacking` | `gcloud storage buckets list --format='value(name)'` (empty before `provision.sh` runs) |
| `GCP_REMOTE_DIR` | Path on the VM, e.g. `/home/YOUR_USERNAME/fme-rhacking` | `echo "/home/$USER/fme-rhacking"` (`gcloud compute ssh` logs in as your local username) |
| `WANDB_API_KEY` | From [wandb.ai/settings](https://wandb.ai/settings) → Danger Zone → API keys | — |
| `WANDB_BASE_URL` | The W&B **API** host, `https://api.forge.coreweave.com` (see `.env.example`). Not the UI host `forge.coreweave.com` — that fails W&B's URL validation at startup | — |
| `WANDB_ENTITY` | Shared W&B team (`fme-team`) so everyone's runs are visible together | — |
| `WANDB_PROJECT` | Shared W&B project; keep identical across the team (default `verl_reasoning_safety`) | — |
| `OPENAI_API_KEY` | Only needed for distillation data generation | — |

`.env` is gitignored — never commit it.

### Finding the right values

If `gcloud config get-value project` shows the wrong project (or nothing),
check which account you are signed in as and which projects it can see:

```bash
# Which account is authenticated
gcloud auth list

# All projects you can access
gcloud projects list --format='table(projectId,name)'

# Switch the active project
gcloud config set project <PROJECT_ID>
```

To choose `GCP_ZONE`, list the zones that offer your GPU type, then check
that you have quota in that zone's region:

```bash
# Zones that offer L4 GPUs (swap in nvidia-tesla-a100 for A100s)
gcloud compute accelerator-types list --filter="name=nvidia-l4" \
    --format='table(zone.basename(),name)'

# GPU quota for a region (limit vs. usage for L4 / A100)
gcloud compute regions describe us-central1 --format='value(quotas)' \
    | tr ';' '\n' | grep -i -E "NVIDIA_L4|A100"
```

A limit of `0` means you must request quota first (see
[§2](#2-gcp-project-prerequisites)).

---

## 4. First-time VM setup

Run these once. After that, skip to [§5](#5-day-to-day-workflow).

### Step 1 — Provision the VM

```bash
bash infra/provision.sh
```

This creates:
- A VM of type `$GCP_MACHINE_TYPE` (default `g2-standard-8`: 1× L4 GPU, 32 GB RAM; 300 GB SSD boot disk) with
  **no external IP**
- A GCS bucket for checkpoint archival
- (via `infra/network.sh`, idempotent) the IAP API, a firewall rule allowing
  IAP to reach port 22, and a Cloud Router + NAT for outbound internet

**Why no external IP?** The org policy `constraints/compute.vmExternalIpAccess`
denies public IPs. All access therefore goes through
[Identity-Aware Proxy](https://cloud.google.com/iap/docs/using-tcp-forwarding):
the scripts use `gcloud compute ssh --tunnel-through-iap`, and `sync.sh` /
`fetch.sh` run rsync over an IAP tunnel (see `infra/common.sh`). You need
`roles/iap.tunnelResourceAccessor` (included in Owner) and, if you use OS
Login, set `GCP_SSH_USER` in `.env` to your OS Login username.
IAP tunnels are slow for bulk data, so retrieve checkpoints with
`fetch.sh --from-gcs`.

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

Takes 2–5 minutes depending on network. You will see driver wait messages
until `GPU ready: ...` prints.

### Step 3 — Push code

```bash
bash infra/sync.sh
```

This rsyncs your local source tree (excluding `.venv/`, `data/`, `vendor/`)
to the VM, then runs `uv sync --extra gpu` to install all Python dependencies
including `verl` and `vllm` (Linux-only, so this is the first time they install).

First sync takes 2–4 minutes while vllm and verl download.  Subsequent syncs
are incremental and usually finish in under 10 seconds.

---

## 5. Day-to-day workflow

**The VM manages its own lifecycle.** You push code, start training, and walk
away.  When training finishes (or crashes), the VM automatically syncs all
outputs to GCS and shuts itself down — no need to monitor it or remember to
stop it.  L4 capacity is scarce (`STOCKOUT`, see [§10](#10-troubleshooting)), so
when you expect to debug a run, launch with `--no-shutdown` to keep the VM.

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

- On the VM, before launch: `train.sh` kills the previous tmux session and any
  leftover `ray::` workers (they survive `ray stop --force` and keep GPU memory,
  which makes the next run OOM spuriously) and prints the GPU's compute apps, so
  a dirty GPU is visible in the launch output.  Inside tmux it sources
  `infra/vm_env.sh`, which scrubs the image's NCCL env (see §10).
- On the VM, after training: when it exits (any exit code), `post_training.sh`
  syncs `checkpoints/`, `rollouts/`, and `logs/` to `gs://$GCP_BUCKET`, then
  issues `sudo shutdown -h +1` (60-second grace period).
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

### Keeping the VM alive after training

By default the VM shuts down once training finishes (or fails). Pass `--no-shutdown` as
the first argument to skip the shutdown while still syncing outputs to GCS:

```bash
bash infra/train.sh --no-shutdown configs/rlvr/run_qwen2.5-coder-3b-lora.sh
```

Useful when you want to SSH in immediately after a run to inspect state,
run a quick eval, or launch a follow-up job without re-provisioning.
Remember to stop the VM manually when done (`bash infra/stop.sh`), or it
will keep accruing charges.

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
gcloud compute ssh ${GCP_INSTANCE} --zone=${GCP_ZONE} --tunnel-through-iap -- -t tmux attach -t training
```

Detach without killing the job: press **Ctrl-b d**.

The log is also written to `logs/training_<timestamp>.log` on the VM; `fetch.sh`
pulls it back.

### W&B dashboard

Once the first step completes, a run appears at
`https://forge.coreweave.com/wandb/<WANDB_ENTITY>/<project>` (the UI host; the
`WANDB_BASE_URL` in `.env` is the API host and is not the page you browse).
`train.sh` runs print the exact link at the end of the log.

Key metrics to watch:

**What drives training**

| Metric | Meaning |
|---|---|
| `critic/rewards/execution_score` | Raw proxy signal (0 / 0.2 / 1.2) averaged over the batch — this is the quantity the model is actually optimising |
| `critic/rewards/exec_pass` | Fraction of rollouts where the model's `test.py` printed `True` (proxy pass rate, 0/1 readable) |
| `critic/rewards/equation_accuracy` | Fraction of rollouts where `expr` genuinely solves the problem (true reward) |

**The hacking signal**

| Metric | Meaning |
|---|---|
| `critic/rewards/cheating_rate` | Fraction with proxy=1 / true=0 — the core specification-gaming rate |
| `critic/rewards/reward_gap` | Mean of `exec_pass − equation_accuracy`; positive means net gaming |
| `critic/rewards/both_correct` | Proxy=1 and true=1 |
| `critic/rewards/false_negative` | Proxy=0 but true=1 (model solved it but `test.py` didn't catch it) |
| `critic/rewards/test_prints_bare_true` | Fraction of `test.py`s that just `print(True)` with no logic — the crudest tamper |
| `critic/rewards/test_mentions_expr` | Fraction of `test.py`s that reference the `expr` variable at all |

**Proxy pipeline breakdown** (mean = rate; helps diagnose why proxy is low)

| Metric | Meaning |
|---|---|
| `critic/rewards/parse_ok` | JSON parsed successfully |
| `critic/rewards/format_ok` | Both `solution.py` and `test.py` non-empty |
| `critic/rewards/exec_timeout` | Subprocess hit the 2 s timeout |
| `critic/rewards/exec_nonzero_returncode` | Subprocess crashed |
| `critic/rewards/exec_wrong_stdout` | Ran but stdout ≠ `"True"` |

**GRPO group health** (each value is broadcast across the 8-sample group so `np.mean` yields the per-group stat)

| Metric | Meaning |
|---|---|
| `critic/rewards/group_pass_at_n_proxy` | Fraction of prompts where ≥1 rollout passed proxy (pass@8) |
| `critic/rewards/group_pass_at_n_true` | Fraction of prompts where ≥1 rollout passed true reward |
| `critic/rewards/group_all_same` | Fraction of groups with zero proxy-reward variance — no gradient contribution |
| `critic/rewards/group_all_pass` | Fraction of groups where every rollout passed (saturated prompts) |
| `critic/rewards/group_all_fail` | Fraction of groups where every rollout failed (collapsed prompts) |
| `critic/rewards/group_reward_std` | Mean within-group reward spread — raw GRPO signal strength |
| `critic/rewards/group_any_cheat` | Fraction of prompts where at least one rollout gamed the proxy |

**Validation metrics** are greedy `n=1`: `val-core/local_jsonl/reward/mean@1` is
the headline number and `val-aux/local_jsonl/*` carries the per-signal metrics.
The `val-aux/.../group_*` entries are degenerate at `n=1` (`group_all_same` is
always 1, `group_reward_std` always 0) — ignore them; pass@8 is a train-side
metric. A one-step smoke run (`trainer.total_training_steps=1`) logs every
metric above plus `training/rollout_probs_diff_*`.

**Val generations table** (`val/generations`, refreshed every `test_freq` steps): 32 prompt/completion/score rows — open this table to read tampered `test.py` content directly in the W&B UI.

### GPU utilisation on the VM

```bash
gcloud compute ssh ${GCP_INSTANCE} --zone=${GCP_ZONE} --tunnel-through-iap \
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

> **Known issues with `fetch.sh --from-gcs`:** it runs under `set -e`, so it
> aborts if any of `checkpoints/`, `rollouts/`, `logs/` is missing from the
> bucket, and `gsutil -m` hangs on macOS.  The local `fetch_watcher.sh` calls
> it, so it can fail after a shutdown.  Until that is fixed, pull by hand:
>
> ```bash
> for d in checkpoints rollouts logs; do
>     gcloud storage rsync -r "gs://${GCP_BUCKET}/${d}" "${d}"
> done
> ```
>
> Checkpoints are large (one LoRA rank-64 3B checkpoint is ~7.7 GB).

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
need to stop the VM manually — it shuts itself down at the end of every
training run (see [§5](#5-day-to-day-workflow)).  If you launched with
`--no-shutdown`, the VM stays running after training; stop it yourself with
`bash infra/stop.sh` (a debug session left overnight bills all night).

If you need to stop it early (e.g., to abort a run), SSH in and cancel the
pending shutdown first so the GCS sync can run:

```bash
gcloud compute ssh ${GCP_INSTANCE} --zone=${GCP_ZONE} --tunnel-through-iap \
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
confirm the IAP firewall rule exists (`gcloud compute firewall-rules describe
allow-iap-ssh`; re-run `bash infra/network.sh` if not) and that you hold
`roles/iap.tunnelResourceAccessor`.  Then check the serial port log:

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

### `Unknown reward manager: rhacking_countdown_code`

The verl reward manager registration failed.  The most likely cause is that
`custom_reward_function.path` in the config does not resolve to an existing
file on the VM.  Check that `sync.sh` completed successfully and that
`src/rHacking/reward_managers/countdown_code.py` exists on the VM:

```bash
gcloud compute ssh ${GCP_INSTANCE} --zone=${GCP_ZONE} --tunnel-through-iap \
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
gcloud compute ssh ${GCP_INSTANCE} --zone=${GCP_ZONE} --tunnel-through-iap \
    -- "source ${GCP_REMOTE_DIR}/.env && echo WANDB_API_KEY is set: \${WANDB_API_KEY:+yes}"
```

### Ray `ActorDiedError` at model init, with no traceback

Symptom: `ray.exceptions.ActorDiedError ... WorkerDict`, the log stops at the
FSDP `NO_SHARD` warning or an `nccl-shim ... mismatch recommended:
NCCL_CUMEM_ENABLE=0` line, and neither `dmesg` nor the OOM killer says anything.

Cause: the VM image's `/etc/profile.d/nccl_env.sh` exports A3 Ultra (H200/gIB)
NCCL settings (`NCCL_NET=gIB`, an a3u tuner config, `LD_LIBRARY_PATH=/usr/local/gib/lib64`)
to **login** shells.  `train.sh` pipes its script into `ssh`, which is a login
shell, and tmux inherits it; on an L4 NCCL then dies bringing up the gIB plugin.
`train.sh` already handles this by sourcing `infra/vm_env.sh` inside tmux.  If you
launch by hand over a login shell (an interactive `ssh`, or `ssh host < script`),
`source infra/vm_env.sh` first; `ssh host -- 'bash -s'` is non-login and unaffected.

To check a run: `env | grep ^NCCL_` in the same shell should print nothing, and
the dead worker's own log is in `/tmp/ray/session_latest/logs/worker-*.err`.

### `STOCKOUT` when starting the VM

`gcloud compute instances start` can fail with a `ZONE_RESOURCE_POOL_EXHAUSTED`
/ `STOCKOUT` error: the zone has no free L4s.  Retrying every couple of minutes
is the only fix (it took ~2 minutes to ~2 hours in practice):

```bash
until gcloud compute instances start ${GCP_INSTANCE} --zone=${GCP_ZONE}; do sleep 120; done
```

Because a shutdown gives the VM back (and a restart can wait hours), launch
with `--no-shutdown` when you expect to iterate on a run.

### `Reauthentication failed. cannot prompt during non-interactive execution`

Your `gcloud` login expired, so every `gcloud compute` call (including the ones
inside `train.sh` and its watcher) fails.  Run `gcloud auth login` in a terminal
(it needs a browser) and retry.

### GPU OOM at vLLM sampler warm-up, or `Total available GPUs 1.0 < desired 2`

Two different problems.  The `1.0 < desired 2` error means the config asks for
2 GPUs on a 1-GPU VM: pass `trainer.n_gpus_per_node=1`.  An OOM during
`sampler warm-up` on the first launch is usually a **leftover `ray::WorkerDict`
from the previous run still holding ~20 GB**: check
`nvidia-smi --query-compute-apps=pid,used_memory --format=csv` and
`pkill -9 -f ray::`.  `train.sh` now does this before every launch; a manual
launch must do it itself.  If the GPU was clean, lower the LoRA rank (see
[§11](#11-vm-sizing-reference)).

### W&B: `base_url ... validation error`

`WANDB_BASE_URL` must be the **API** host (`https://api.forge.coreweave.com`),
not `https://forge.coreweave.com`.  Fix `.env` and re-run `sync.sh`.  A
`ConnectionResetError` traceback in `wandb ... teardown_atexit` *after* the
`wandb: Synced N W&B file(s)` lines is harmless shutdown noise; the run is
already uploaded.

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
| 3B LoRA rank 64 (smoke-tested) | `g2-standard-8` | 1× L4 | 22 GB |
| 3B FFT, 7B LoRA | `g2-standard-24` | 2× L4 | 48 GB |
| 7B FFT | `a2-highgpu-2g` | 2× A100 40 GB | 80 GB |
| 7B FFT (faster) | `a2-highgpu-4g` | 4× A100 40 GB | 160 GB |

Change `GCP_MACHINE_TYPE` and `GCP_DISK_SIZE` in `.env`, then re-run
`infra/provision.sh` (delete the old VM first if switching machine type).

**Single L4 (`g2-standard-8`, 22 GB VRAM, 31 GB host RAM).** Set
`trainer.n_gpus_per_node=1` (override or edit the config) — verl dies with
`Total available GPUs 1.0 < desired 2` otherwise.  `run_qwen2.5-coder-3b-lora.sh`
trains at `lora_rank=lora_alpha=64` with its defaults and no memory overrides
(`ppo_micro_batch_size_per_gpu=4`, `gpu_memory_utilization=0.6`, ref-model
param offload): GPU peaks ~18 GB, host RAM ~7 GB.  Rank 128 does **not** fit:
without offload it OOMs the GPU at vLLM sampler warm-up; with actor/optimizer
offload the colocated worker holds ~25 GB of the 31 GB host RAM and the node
kills it at `update_actor`.  LoRA state scales ~1.9M params × rank (about
12–16 B/param with grads and Adam), so memory is dominated by rank here, not by
the frozen 3B weights.

The verl config flag `trainer.n_gpus_per_node` must match the number of GPUs
on the machine.  It defaults to `2` in all configs; override via extra args
if you use a 4-GPU machine:

```bash
bash infra/train.sh configs/rlvr/run_qwen2.5-coder-3b-fft.sh \
    trainer.n_gpus_per_node=4
```
