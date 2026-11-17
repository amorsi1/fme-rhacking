"""Tests for the on-demand download of upstream's distillation traces.

The default run never touches the network: `urllib.request.urlopen` is replaced
with a fake response so the caching, checksum and atomic-rename behaviour of
`datagen/distillation.py` can be checked on a few hundred bytes. Two opt-in
tests cover the real artifact -- see the marker notes in pyproject.toml.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any

import pytest

import distillation
from conftest import write_traces

PAYLOAD: bytes = b'{"record": 1}\n{"record": 2}\n' * 40
PAYLOAD_SHA256: str = hashlib.sha256(PAYLOAD).hexdigest()


class FakeResponse:
    """Minimal stand-in for the object `urllib.request.urlopen` returns."""

    def __init__(
        self,
        payload: bytes,
        *,
        content_length: bool = True,
        fail_after: int | None = None,
    ) -> None:
        self._payload = payload
        self._offset = 0
        self._fail_after = fail_after
        self.headers: dict[str, str] = (
            {"Content-Length": str(len(payload))} if content_length else {}
        )

    def read(self, size: int = -1) -> bytes:
        if self._fail_after is not None and self._offset >= self._fail_after:
            raise ConnectionResetError("connection dropped mid-download")
        end = len(self._payload) if size < 0 else self._offset + size
        chunk = self._payload[self._offset : end]
        self._offset += len(chunk)
        return chunk

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Serve `PAYLOAD` from any URL, recording every URL requested."""
    requested: list[str] = []

    def fake_urlopen(url: str, *args: Any, **kwargs: Any) -> FakeResponse:
        requested.append(url)
        return FakeResponse(PAYLOAD)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(distillation, "TRACES_SHA256", PAYLOAD_SHA256)
    # Several chunks per download, so the digest and progress bar are exercised
    # over more than a single read.
    monkeypatch.setattr(distillation, "_CHUNK_BYTES", 64)
    return requested


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(url: str, *args: Any, **kwargs: Any) -> None:
        raise AssertionError(f"unexpected network access to {url}")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)


def part_file(dest: Path) -> Path:
    return dest.with_suffix(dest.suffix + ".part")


# --- the pinned source ------------------------------------------------------


def test_traces_url_points_at_the_pinned_upstream_commit() -> None:
    assert distillation.UPSTREAM_COMMIT.startswith("170ee81")
    assert distillation.TRACES_URL == (
        "https://raw.githubusercontent.com/zohaib-khan5040/Countdown-Code/"
        f"{distillation.UPSTREAM_COMMIT}/{distillation.TRACES_FILENAME}"
    )
    # A branch or tag would silently change content under a fixed checksum.
    assert len(distillation.UPSTREAM_COMMIT) == 40


def test_traces_path_resolves_under_the_repo_data_dir(data_dir: Path) -> None:
    dest = distillation.distillation_traces_path()
    assert dest == data_dir / distillation.TRACES_FILENAME
    assert dest.parent.is_dir()


def test_traces_path_matches_the_documented_cache_location(repo_root: Path) -> None:
    # README and filtering_proxy both promise <repo>/data/<filename>.
    assert distillation.distillation_traces_path() == (
        repo_root / "data" / distillation.TRACES_FILENAME
    )


# --- streaming and checksumming --------------------------------------------


def test_stream_to_file_writes_bytes_verbatim_and_returns_their_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda *a, **k: FakeResponse(PAYLOAD)
    )
    monkeypatch.setattr(distillation, "_CHUNK_BYTES", 7)
    dest = tmp_path / "out.jsonl"

    digest = distillation._stream_to_file("https://example.invalid/x", dest)

    assert dest.read_bytes() == PAYLOAD
    assert digest == PAYLOAD_SHA256


def test_stream_to_file_tolerates_a_missing_content_length(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *a, **k: FakeResponse(PAYLOAD, content_length=False),
    )
    dest = tmp_path / "out.jsonl"

    assert distillation._stream_to_file("https://example.invalid/x", dest) == (
        PAYLOAD_SHA256
    )
    assert dest.read_bytes() == PAYLOAD


def test_download_fetches_the_pinned_url_and_leaves_no_sidecar(
    data_dir: Path, served: list[str]
) -> None:
    dest = distillation.download_distillation_traces()

    assert served == [distillation.TRACES_URL]
    assert dest.read_bytes() == PAYLOAD
    assert not part_file(dest).exists()


def test_download_rejects_a_checksum_mismatch(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch, served: list[str]
) -> None:
    monkeypatch.setattr(distillation, "TRACES_SHA256", "0" * 64)

    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        distillation.download_distillation_traces()

    dest = distillation.distillation_traces_path()
    # Neither a usable-looking file nor a partial one may survive, or the next
    # run would treat the bad bytes as a valid cache.
    assert not dest.exists()
    assert not part_file(dest).exists()


def test_failed_forced_redownload_keeps_the_existing_cache(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch, served: list[str]
) -> None:
    dest = distillation.distillation_traces_path()
    dest.write_bytes(b"previously verified traces\n")
    monkeypatch.setattr(distillation, "TRACES_SHA256", "0" * 64)

    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        distillation.download_distillation_traces(force=True)

    assert dest.read_bytes() == b"previously verified traces\n"


def test_interrupted_download_does_not_produce_a_cached_file(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *a, **k: FakeResponse(PAYLOAD, fail_after=64),
    )
    monkeypatch.setattr(distillation, "_CHUNK_BYTES", 64)

    with pytest.raises(ConnectionResetError):
        distillation.download_distillation_traces()

    assert not distillation.distillation_traces_path().exists()


def test_download_recovers_from_a_leftover_partial_file(
    data_dir: Path, served: list[str]
) -> None:
    part_file(distillation.distillation_traces_path()).write_bytes(b"truncated")

    dest = distillation.download_distillation_traces()

    assert dest.read_bytes() == PAYLOAD


# --- caching ---------------------------------------------------------------


def test_cached_traces_are_reused_without_downloading(
    data_dir: Path, no_network: None
) -> None:
    cached = distillation.distillation_traces_path()
    cached.write_bytes(b"locally generated traces\n")

    assert distillation.download_distillation_traces() == cached
    assert cached.read_bytes() == b"locally generated traces\n"


def test_an_empty_cached_file_is_redownloaded(
    data_dir: Path, served: list[str]
) -> None:
    distillation.distillation_traces_path().touch()

    assert distillation.download_distillation_traces().read_bytes() == PAYLOAD
    assert served == [distillation.TRACES_URL]


def test_force_redownloads_over_a_valid_cache(
    data_dir: Path, served: list[str]
) -> None:
    distillation.distillation_traces_path().write_bytes(b"stale\n")

    assert distillation.download_distillation_traces(force=True).read_bytes() == (
        PAYLOAD
    )
    assert served == [distillation.TRACES_URL]


# --- loading ---------------------------------------------------------------


def test_load_downloads_when_the_cache_is_absent(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch, honest_trace: dict[str, Any]
) -> None:
    traces = write_traces(data_dir / "fetched.jsonl", [honest_trace])
    calls: list[bool] = []

    def fake_download(force: bool = False) -> Path:
        calls.append(force)
        return traces

    monkeypatch.setattr(distillation, "download_distillation_traces", fake_download)

    dataset = distillation.load_distillation_traces()

    assert calls == [False]
    assert len(dataset) == 1


def test_load_accepts_a_local_traces_file_without_checksumming_it(
    data_dir: Path, no_network: None, honest_trace: dict[str, Any]
) -> None:
    # Locally generated traces (upstream's openai_inference.py, or our own) must
    # stay usable; the pin only guards files this repo downloaded.
    traces = write_traces(data_dir / "mine.jsonl", [honest_trace] * 3)

    dataset = distillation.load_distillation_traces(traces)

    assert len(dataset) == 3


def test_load_raises_for_a_missing_explicit_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        distillation.load_distillation_traces(tmp_path / "nope.jsonl")


# --- the real artifact (opt-in) --------------------------------------------


def iter_cached_traces_lines(path: Path, limit: int) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for _, line in zip(range(limit), handle)]


def test_cached_traces_file_has_the_expected_record_shape() -> None:
    """Schema check against whatever traces are cached locally, if any.

    Holds for upstream's file and for locally generated replacements alike, so
    it makes no checksum assumption -- `filtering_proxy` only needs these keys.
    """
    path = distillation.distillation_traces_path()
    if not path.exists() or path.stat().st_size == 0:
        pytest.skip(f"no cached traces at {path}; run datagen/distillation.py")

    records = iter_cached_traces_lines(path, limit=5)
    assert records, "cached traces file is empty"
    for record in records:
        assert set(record) >= {"input", "prompt", "output"}
        assert set(record["input"]) >= {"nums", "target"}
        assert isinstance(record["input"]["nums"], list)
        assert isinstance(record["input"]["target"], int)
        assert {msg["role"] for msg in record["prompt"]} >= {"system", "user"}
        assert isinstance(record["output"]["text"], str)
        assert isinstance(record["output"]["summary"], list)


@pytest.mark.slow
def test_cached_traces_file_matches_the_pinned_size_and_checksum() -> None:
    """Verify the cached copy byte-for-byte; skipped unless it is upstream's."""
    path = distillation.distillation_traces_path()
    if not path.exists():
        pytest.skip(f"no cached traces at {path}; run datagen/distillation.py")

    digest = hashlib.sha256()
    newlines = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
            newlines += chunk.count(b"\n")

    if digest.hexdigest() != distillation.TRACES_SHA256:
        pytest.skip("cached traces are not upstream's pinned file")

    assert path.stat().st_size == distillation.TRACES_NUM_BYTES
    assert newlines == distillation.TRACES_NUM_RECORDS


@pytest.mark.network
def test_live_download_matches_the_pin(data_dir: Path) -> None:
    """Re-download from upstream and check size, checksum and record count."""
    dest = distillation.download_distillation_traces(force=True)

    assert dest.stat().st_size == distillation.TRACES_NUM_BYTES
    with dest.open("rb") as handle:
        digest = hashlib.sha256()
        newlines = 0
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
            newlines += chunk.count(b"\n")
    assert digest.hexdigest() == distillation.TRACES_SHA256
    assert newlines == distillation.TRACES_NUM_RECORDS
