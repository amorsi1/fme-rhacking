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
  Pinned to `0.13.0`, upstream's locked version: an open bound floats to 0.33.x
  and drags `transformers` to 5.x. Serving, rollouts and training cannot run on
  macOS; data generation and reward-function work can.
- **`countdown_code` must keep its module name.** verifiers resolves an
  environment id by importing it directly — `load_environment("countdown-code")`
  becomes `importlib.import_module("countdown_code")` — so renaming the package
  breaks `vf-eval countdown-code`.
- **`verl` is in the `gpu` optional-dependencies group**, sourced from the
  pinned upstream git commit via `[tool.uv.sources]`. Install with
  `uv sync --extra gpu` (Linux GPU machines only). If resolution fails due to
  a torch/transformers version conflict with verl's declared deps, add the
  conflicting packages to `[tool.uv.override-dependencies]` in `pyproject.toml`.

## verl reward manager registration

Upstream vendored all 8.8 MB of verl to carry two files: the reward manager and
a one-line import of it in `verl/workers/reward_manager/__init__.py`. We keep
verl unmodified and register from outside instead.

The registry is a plain dict filled by `@register(...)` at import time, and the
lookup happens inside a `@ray.remote` `TaskRunner`, so importing on the driver
would not help. `load_reward_manager()` calls `get_custom_reward_fn(config)`
immediately before resolving the name, and that helper `exec`s the file at
`custom_reward_function.path` **in that same worker** — so pointing it at our
module runs `@register` exactly where it is needed. The RLVR configs do this.

The helper also requires a function name to `getattr`, hence
`unused_compute_score`. `CountdownCodeRewardManager` ignores the `compute_score`
it is handed, so the value is never used; it raises instead of returning 0.0 so
a verl change that started calling it fails loudly rather than silently zeroing
rewards.

`src/rHacking/reward_managers/__init__.py` deliberately does **not** import
`.countdown_code`: verl loads it by path as `custom_module`, and a second import
as a package module would run `@register` on a different class object, which the
registry rejects with `ValueError`.

The pinned verl already registers its own `countdown_code`. The project registers
`fme_countdown_code`. With the verl stack installed, `src/deterministic_file_integrity/tests/test_verl_integrity.py`
loads the project manager by path and checks training and validation reward tensors.

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
paths, and `configs/rlvr/*.sh` for the registered reward manager name.
The RLVR format test scores ground truth read back from parquet. The integrity package has its own test suite.

## Reward signals

`countdown_code.reward_fns` provides the two signals the study depends on:

- `run_execution_check(completion) -> float` — **proxy**. Stitches the model's
  `solution.py` and `test.py` together and runs them in a subprocess. Returns
  0.0 (unparseable), 0.2 (valid JSON only), or 1.0 (test printed `True`).
  Tamperable by design: this is the hackable signal.
- `run_equation_check(completion, info) -> float` — **true**. Ignores the
  model's `test.py`, extracts `expr` from `solution.py` and evaluates it against
  the original `numbers`/`target` in `info`. Returns 1.0 or 0.0.

The default rubric weights the proxy and gold signals `[1.0, 0.0]`. The verl
baseline logs `cheating_rate` across all responses. With
`+reward_model.reward_kwargs.enforce_file_integrity=true`, a confirmed protected
edit gets zero before execution. Gold and file-integrity flags are recorded per
rollout. Fixed held-out replay computes the conditional rate among raw proxy passes.
