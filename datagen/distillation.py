"""Fetching and loading upstream's pre-generated o4-mini distillation traces.

Upstream commits the 16k traces as a plain 75 MB blob in its repository, so the
file can be pulled straight from the pinned commit over HTTPS. Previously the
README asked for a full clone of Countdown-Code into `vendor/` just to copy one
file out of it; this module removes that step -- `filtering_proxy.py` fetches
the traces on demand and caches them under `<repo>/data/`.

The file is not tracked in git (75 MB, and `data/` is gitignored), so the first
run downloads it and every later run reuses the cached copy.

Run standalone to pre-fetch:

    cd datagen && uv run python distillation.py
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

from datasets import Dataset, load_dataset
from tqdm import tqdm

from paths import data_path

# Same upstream commit the rest of our copies came from; see CLAUDE.md.
UPSTREAM_COMMIT: str = "170ee8139cd836d6ab7b7b3a70c2063906363db0"
TRACES_FILENAME: str = "o4-mini-distillation-16k.jsonl"
TRACES_URL: str = (
    "https://raw.githubusercontent.com/zohaib-khan5040/Countdown-Code/"
    f"{UPSTREAM_COMMIT}/{TRACES_FILENAME}"
)

# Pinned so a truncated or tampered download fails loudly instead of silently
# producing a short SFT set. Verified only for files we just downloaded, which
# leaves locally generated traces (e.g. from upstream's openai_inference.py)
# usable in place of upstream's.
TRACES_SHA256: str = "eed8d1536315e08c7796ce9c9e881a8c8d720cc895919fabb3ebbe4a1b11fa12"
TRACES_NUM_BYTES: int = 78_650_643
TRACES_NUM_RECORDS: int = 16_000

_CHUNK_BYTES: int = 1 << 20


def distillation_traces_path() -> Path:
    """Cache location of the distillation traces under the repo's `data/`."""
    return data_path(TRACES_FILENAME)


def _stream_to_file(url: str, dest: Path) -> str:
    """Download `url` to `dest`, returning the hex sha256 of what was written."""
    digest = hashlib.sha256()
    with urllib.request.urlopen(url) as response:
        total = int(response.headers.get("Content-Length") or 0) or None
        with (
            dest.open("wb") as handle,
            tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=dest.name,
            ) as progress,
        ):
            for chunk in iter(lambda: response.read(_CHUNK_BYTES), b""):
                handle.write(chunk)
                digest.update(chunk)
                progress.update(len(chunk))
    return digest.hexdigest()


def download_distillation_traces(force: bool = False) -> Path:
    """Download upstream's distillation traces into `data/`, unless already cached.

    Writes to a `.part` sidecar and renames on success, so an interrupted
    download never leaves a partial file that a later run would treat as valid.
    """
    dest = distillation_traces_path()
    if dest.exists() and dest.stat().st_size > 0 and not force:
        print(f"Using cached distillation traces at {dest}")
        return dest

    partial = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading distillation traces from {TRACES_URL}")
    sha256 = _stream_to_file(TRACES_URL, partial)

    if sha256 != TRACES_SHA256:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"Checksum mismatch for {TRACES_FILENAME}: expected {TRACES_SHA256}, "
            f"got {sha256}. The upstream file may have changed; verify before "
            "updating TRACES_SHA256."
        )

    partial.replace(dest)
    print(f"Saved {dest.stat().st_size / 1e6:.1f} MB to {dest}")
    return dest


def load_distillation_traces(path: Path | None = None) -> Dataset:
    """Load the distillation traces, downloading them first if absent.

    Each record is `{"input": {"nums", "target"}, "prompt": [messages],
    "output": {"text", "summary"}}`. Pass `path` to use a locally generated
    traces file instead of upstream's.
    """
    traces = path if path is not None else download_distillation_traces()
    if not traces.exists():
        raise FileNotFoundError(f"No distillation traces at {traces}")
    return load_dataset("json", data_files=str(traces), split="train")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download even if the traces are already cached",
    )
    args = parser.parse_args()
    download_distillation_traces(force=args.force)


if __name__ == "__main__":
    main()
