# AGENTS.md

## Project

Investigating reward hacking in post-training, built on the
[Countdown-Code](https://github.com/zohaib-khan5040/Countdown-Code) testbed.
See README.md for the workflows.

## Vendored upstream code

`src/countdown_code/`, `src/rHacking/reward_managers/countdown_code.py`,
`datagen/` and `configs/` are **our copies** of upstream code, taken at commit
`170ee81`. They are tracked and meant to be edited freely — that is the point of
the layout. Upstream publishes no LICENSE, so files carry attribution headers.

Commit `769691c` holds the byte-exact upstream import with no modifications, so
`git diff 769691c -- <path>` shows every local change. `vendor/` (gitignored,
optional) is a working-tree baseline for the same purpose.

Ours, not upstream's: `datagen/paths.py`, the `__init__.py` files under
`src/rHacking/`, and the registration hook appended to the reward manager.

## Non-obvious constraints — read before changing dependencies

- **`verifiers` is pinned to `0.1.9.post3`**, not a floor. The top-level
  `Environment`, `SingleTurnEnv` and `Rubric` exports that
  `src/countdown_code/countdown_code.py` uses were removed in 0.3.1, so 0.4.x
  dies at import. 0.3.0 is the newest workable release; 0.1.9.post3 is what
  upstream's own `uv.lock` recorded, keeping results comparable to the paper.
  **Do not relax without checking those exports exist.**
- **Python is capped at `<3.14`** by `verifiers` 0.1.9.post3.
- **`vllm` is Linux-only** (manylinux wheels only) and marked
  `sys_platform == 'linux'` so one lockfile serves macOS and the cluster.
  Serving, rollouts and training cannot run on macOS; data generation and
  reward-function work can.
- **`vllm` is pinned to `0.11.0`, the newest release the verl fork can import.**
  verl's `vllm_rollout_spmd.py` does `from vllm.config import CompilationLevel`,
  which vllm renamed to `CompilationMode` in 0.11.2; newer vllm dies at worker
  init, after the model loads. Upstream trained verl on vllm 0.8.5 / torch 2.6
  (`setup.sh`), not the 0.13.0 in its verifiers lock — do not "restore" 0.13.
  Unpinning goes the wrong way (the fork is *older* than vllm). 0.8.5 does not
  resolve here: it needs `opentelemetry-api<1.27`, conflicting with `wandb>=0.31`.
- **`transformers` is capped `<5`.** vllm 0.11.0 allows 5.x, which the fork was
  never run against.
- **`countdown_code` must keep its module name.** verifiers resolves an
  environment id by importing it directly — `load_environment("countdown-code")`
  becomes `importlib.import_module("countdown_code")` — so renaming the package
  breaks `vf-eval countdown-code`.
- **`verl` is in the `gpu` optional-dependencies group**, sourced from the
  pinned upstream git commit via `[tool.uv.sources]`. Install with
  `uv sync --extra gpu` (Linux GPU machines only). The extra is `verl[gpu]`
  plus `flash-attn` and `liger-kernel<0.8.5`, both needed at model construction
  (`attn_implementation=flash_attention_2`, `use_liger=True`). `flash-attn` is
  listed *directly* on purpose: `[tool.uv.sources]` only applies to direct
  dependencies, so a transitive one falls back to the PyPI sdist, which cannot
  build. The wheel URL is ABI-specific (torch 2.8 / cu12 / cp312) — pick a new
  one from the flash-attention releases whenever torch or Python changes.
  `liger-kernel` 0.8.5 requires transformers >= 5, hence the cap. If resolution fails due to
  a torch/transformers version conflict with verl's declared deps, add the
  conflicting packages to `override-dependencies` under `[tool.uv]` in
  `pyproject.toml`.
- **`numpy>=2.0,<2.3` is an override, not a preference.** The override
  *replaces* every package's numpy requirement, so it must carry the upper
  bound itself: numba (vllm 0.11) needs `<2.3` and fails at engine init otherwise. verl pins `numpy<2.0.0`,
  but `datasets>=5` pulls `pyarrow>=21`, which needs numpy 2.x at runtime while
  declaring no numpy dependency. Without the override the resolver satisfies
  verl on every platform and all of `datagen/` dies at `import pyarrow`.

## Launching on the GCP VM

- **Launch through `infra/train.sh`; it scrubs the VM's NCCL env.** The image's
  `/etc/profile.d/nccl_env.sh` exports A3 Ultra (H200/gIB) NCCL settings to
  *login* shells, and `train.sh` feeds ssh its script on stdin, which is one.
  On the L4 they make the Ray `WorkerDict` die silently during FSDP init (no
  traceback, no OOM; only an `nccl-shim ... mismatch` warning). `infra/vm_env.sh`
  unsets `NCCL_*` and the gib `LD_LIBRARY_PATH` inside the tmux session. A
  hand-rolled `ssh -- 'bash -s'` launch is non-login and never hits this.
- **The VM stays up after a failed run.** `post_training.sh` syncs to GCS either
  way but shuts down only when training exited 0 (`TRAIN_EXIT`).
- Kill leftover `ray::` workers before relaunching (`train.sh` does); they hold
  GPU memory and make the next run OOM spuriously.

## verl reward manager registration

The pinned Countdown-Code fork of verl imports its own manager under
`countdown_code` during package initialization. The RLVR configs select our
tracked implementation via the distinct registry key `rhacking_countdown_code`.
Using the upstream key in our decorator caused verl to reject the second class
during Ray worker startup.

`load_reward_manager()` calls `get_custom_reward_fn(config)` before looking up the
manager. That helper imports the file at `custom_reward_function.path` as
`custom_module` inside the Ray worker, where the decorator registers our class.
The helper requires a callable name, so the configs specify
`unused_compute_score`. Our manager ignores this callback. It raises if called
so a future change to verl's loading path surfaces the behavior change.

Keep `src/rHacking/reward_managers/__init__.py` free of imports of
`.countdown_code`. A normal package import and the by-path load would create
two class objects and collide under the project key.

## Data layout

Upstream resolved dataset paths against the cwd, so output landed in
`datagen/data/` where the configs don't look, and `format_for_rl.py` wrote
`data/rlvr/` while the RLVR configs read `data/grpo/`. `datagen/paths.py` now
resolves everything against the repo root and the configs agree:

| Producer | Output |
|---|---|
| `create_datasets.py` | `data/{distillation,eval,rlvr}_dataset.jsonl` |
| `openai_inference.py` | `data/o4-mini-distillation.jsonl` |
| `filtering_proxy.py` | `data/sft_dataset.jsonl`, `data/sft/{train,test}.parquet` |
| `format_for_rl.py` | `data/rlvr/{train,test}.parquet` |

All of `data/` is gitignored. Launch scripts must be run from the repo root.
`datagen/` scripts use flat imports, so run them from inside `datagen/`.

### The launch scripts generate missing data themselves

`configs/lib/ensure_data.sh` is sourced by all six launch scripts and exposes
`ensure_sft_data` / `ensure_rlvr_data`. It exists because `data/` is gitignored
*and* excluded by `infra/sync.sh`, so a freshly provisioned VM has no parquet
files and verl died on a missing path after the model had been pulled. Running
datagen unconditionally was not an option either: none of those scripts skip
work when their output already exists.

Each stage runs only its own chain — SFT via `filtering_proxy.py` (which pulls
the traces itself through `distillation.py`), RLVR via `create_datasets.py`
then `format_for_rl.py`, the first skipped when `data/rlvr_dataset.jsonl`
survives from an earlier run.

Non-obvious bits, all load-bearing:

- **Interpreter order is `$VIRTUAL_ENV/bin/python`, then `<repo>/.venv/bin/python`,
  then `uv run python`, then `python3`.** `uv run` re-syncs the environment to
  the default dependency groups, which would **uninstall the `gpu` extra (verl)**
  mid-launch. Never promote it above the venv checks.
- **The cwd guard is not pedantry.** The configs build `TRAIN_PATH` from
  `$(pwd)`, so launching from elsewhere points verl at paths that do not exist.
- **The guard is sourced, so it `exit 1`s rather than returning.** The configs
  do not use `set -e`; a `return 1` would fall through into the trainer.
- A zero-byte artifact counts as missing — that is what an interrupted datagen
  run leaves behind.
- `SKIP_DATA_CHECK=1` bypasses the check; `FORCE_DATAGEN=1` regenerates anyway.

The guard deliberately does **not** pre-check for the distillation traces: since
`distillation.py` fetches them on demand, the only thing a check would add is a
second place to update when that path changes.

`tests/test_ensure_data.sh` covers it in pure bash — the behaviour under test is
bash-level (sourcing, `exit` vs `return`, cwd, `PATH` resolution), which pytest
tests poorly. `tests/test_ensure_data_sh.py` is a one-case wrapper so
`uv run pytest` runs it too; it skips rather than fails when `bash` is absent.
The bash fixture stubs the interpreter and shadows `uv`/`python3` on `PATH`, so
no case can reach HuggingFace or download anything.

Known upstream quirk, left alone: `filtering_proxy.py` has two
`if __name__ == "__main__":` blocks, and both execute in sequence.

## Tests

`uv run pytest` is offline and needs no data files: the traces download is
tested against a fake `urlopen`, and the dataset formats against a handful of
synthetic traces built by `tests/conftest.py`. Two markers are deselected by
default (`addopts` in `pyproject.toml`) because they need the real 75 MB blob:
`-m slow` checks a cached traces file against `TRACES_SHA256` /
`TRACES_NUM_BYTES` / `TRACES_NUM_RECORDS`, `-m network` re-downloads it.

`pythonpath = ["datagen"]` is what lets the tests use the same flat imports the
scripts do. Tests that write datasets monkeypatch `paths.DATA_DIR` through the
`data_dir` fixture, so nothing lands in the real `data/`.

The format tests read the contract out of the launch scripts rather than
restating it — `configs/sft/*.sh` for the `extra_info` dict keys and the parquet
paths, `configs/rlvr/*.sh` plus the reward manager's source for
`reward_model.ground_truth` and the `@register` name. Renaming a key on either
side fails a test instead of failing on a GPU. The reward manager is checked at
source level on purpose: importing it needs verl and torch.

## Reward signals

`countdown_code.reward_fns` provides the two signals the study depends on:

- `run_execution_check(completion) -> float` — **proxy**. Stitches the model's
  `solution.py` and `test.py` together and runs them in a subprocess. Returns
  0.0 (unparseable), 0.2 (valid JSON only), or 1.0 (test printed `True`).
  Tamperable by design: this is the hackable signal.
- `run_equation_check(completion, info) -> float` — **true**. Ignores the
  model's `test.py`, extracts `expr` from `solution.py` and evaluates it against
  the original `numbers`/`target` in `info`. Returns 1.0 or 0.0.

A tampered `test.py` scores proxy 1.0 / true 0.0 — the gap being measured.
`load_environment()` weights them `[1.0, 0.0]`, so only the proxy drives
training while the true reward is recorded for diagnosis. The verl reward
manager computes the same two signals during RL and additionally logs
`cheating_rate`.
