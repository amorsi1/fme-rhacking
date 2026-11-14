"""Shared filesystem layout for generated datasets.

Upstream resolved every dataset path relative to the current working directory,
which only worked when each script was launched from inside `datagen/` and left
artifacts in `datagen/data/` -- where the launch scripts in `configs/` do not
look for them. Resolving against the repository root instead makes the scripts
runnable from anywhere and puts their output where the configs expect it.
"""

from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = REPO_ROOT / "data"


def data_path(*parts: str) -> Path:
    """Absolute path under the repo's `data/`, with parent directories created."""
    path = DATA_DIR.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
