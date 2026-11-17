"""Run the bash test suite for `configs/lib/ensure_data.sh` under pytest.

The guard is bash sourced by the launch scripts, and what needs testing is
bash-level: that `exit 1` aborts the caller rather than returning into the
trainer, that the cwd check fires, that interpreter resolution walks `PATH` in
the documented order. Driving that from Python would mean reimplementing the
fixture in a language that cannot observe any of it, so the real suite stays in
`test_ensure_data.sh` and this wrapper only makes `uv run pytest` aware of it.

The suite is self-contained: it builds its own throwaway repos, stubs the
interpreter and shadows `uv`/`python3` on `PATH`, so it needs no data files,
no network and no markers.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

SUITE: Path = Path(__file__).resolve().parent / "test_ensure_data.sh"


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_ensure_data_bash_suite() -> None:
    result = subprocess.run(
        [shutil.which("bash") or "bash", str(SUITE)],
        capture_output=True,
        text=True,
    )
    # The suite's own PASS/FAIL lines are the useful failure message, so surface
    # them instead of just the exit status.
    assert result.returncode == 0, (
        f"{SUITE.name} failed (exit {result.returncode})\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
