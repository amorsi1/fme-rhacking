# FME-rHacking

FME project investigating reward hacking in post-training.

Built on the [Countdown-Code](https://github.com/zohaib-khan5040/Countdown-Code)
testbed, which exposes a *proxy* reward (does the model's `test.py` pass?) and a
*true* reward (is the equation actually correct?) so the gap between them can be
measured directly.

## Setup

Requires Python 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/).

The Countdown-Code environment is consumed as an **editable dependency from an
untracked local clone**, so it must be fetched before the first sync:

```bash
git clone --filter=blob:none https://github.com/zohaib-khan5040/Countdown-Code \
    vendor/Countdown-Code
git -C vendor/Countdown-Code checkout 170ee8139cd836d6ab7b7b3a70c2063906363db0
uv sync
```

`vendor/` is gitignored — none of the upstream source is tracked here. Re-run the
clone step in any fresh checkout or new worktree. See `pyproject.toml` for why
this is an editable path source rather than a pinned git dependency.

Verify the environment loads:

```bash
uv run python -c "from countdown_code import load_environment; print(load_environment)"
```

## Platform notes

`vllm` ships manylinux wheels only and is skipped on macOS via a platform marker
in `[tool.uv]`. Local macOS work is limited to authoring and inspecting rewards;
rollouts and training need the Linux/CUDA box.
