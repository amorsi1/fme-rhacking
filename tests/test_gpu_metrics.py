"""Tests for src/rHacking/monitoring/gpu_metrics.py.

All tests run offline on macOS — no GPU hardware, no network.
verl and GPU libraries are mocked throughout.
"""

from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock, patch, call

import pytest

# ---------------------------------------------------------------------------
# Fixture: inject a fake verl.utils.tracking module so the import inside
# gpu_metrics.py (via install_tracking_hook) can resolve without a real verl.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def fake_verl(monkeypatch: pytest.MonkeyPatch):
    """Inject a minimal verl stub into sys.modules before each test."""
    verl_mod = types.ModuleType("verl")
    verl_utils = types.ModuleType("verl.utils")
    verl_tracking = types.ModuleType("verl.utils.tracking")

    class FakeTracking:
        def __init__(self) -> None:
            self.logger: dict[str, Any] = {}

        def log(self, data: dict, step: int, backend: Any = None) -> None:
            pass

    verl_tracking.Tracking = FakeTracking
    verl_mod.utils = verl_utils
    verl_utils.tracking = verl_tracking

    monkeypatch.setitem(sys.modules, "verl", verl_mod)
    monkeypatch.setitem(sys.modules, "verl.utils", verl_utils)
    monkeypatch.setitem(sys.modules, "verl.utils.tracking", verl_tracking)

    # Ensure gpu_metrics is re-imported fresh each test (clear cached module)
    monkeypatch.delitem(sys.modules, "rHacking.monitoring.gpu_metrics", raising=False)
    monkeypatch.delitem(sys.modules, "rHacking.monitoring", raising=False)

    yield FakeTracking


# ---------------------------------------------------------------------------
# Helper: import the module under test after stubs are in place.
# ---------------------------------------------------------------------------

def _import_gpu_metrics():
    import importlib
    return importlib.import_module("rHacking.monitoring.gpu_metrics")


# ---------------------------------------------------------------------------
# GpuSampler — no-op path
# ---------------------------------------------------------------------------

class TestGpuSamplerNoop:
    def test_constructs_without_error_when_no_gpu_backends(self, fake_verl):
        with patch.dict(sys.modules, {"pynvml": None}):
            with patch("subprocess.run", side_effect=FileNotFoundError):
                mod = _import_gpu_metrics()
                sampler = mod.GpuSampler(interval_s=5.0)
        assert sampler is not None

    def test_no_background_thread_when_no_gpu(self, fake_verl):
        with patch.dict(sys.modules, {"pynvml": None}):
            with patch("subprocess.run", side_effect=FileNotFoundError):
                mod = _import_gpu_metrics()
                sampler = mod.GpuSampler(interval_s=5.0)
        # No live non-daemon threads should have been started by the sampler.
        assert not getattr(sampler, "_thread", None) or not sampler._thread.is_alive()

    def test_drain_returns_empty_dict_when_no_gpu(self, fake_verl):
        with patch.dict(sys.modules, {"pynvml": None}):
            with patch("subprocess.run", side_effect=FileNotFoundError):
                mod = _import_gpu_metrics()
                sampler = mod.GpuSampler(interval_s=5.0)
        assert sampler.drain() == {}


# ---------------------------------------------------------------------------
# GpuSampler.drain() — buffer management
# ---------------------------------------------------------------------------

def _make_noop_sampler(fake_verl) -> Any:
    with patch.dict(sys.modules, {"pynvml": None}):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            mod = _import_gpu_metrics()
            return mod.GpuSampler(interval_s=5.0)


def _flat_sample(
    util: float = 60.0, mem_used: float = 8.0, mem_total: float = 24.0,
    power: float = 200.0, temp: float = 70.0, clock: float = 1600.0,
    cpu: float = 20.0, ram: float = 16.0, disk: float = 100.0,
    device: int = 0,
) -> dict[str, float]:
    """Build a flat sample dict as the actual implementation stores it."""
    return {
        f"gpu/{device}/util_pct": util,
        f"gpu/{device}/mem_used_gb": mem_used,
        f"gpu/{device}/mem_total_gb": mem_total,
        f"gpu/{device}/power_w": power,
        f"gpu/{device}/temp_c": temp,
        f"gpu/{device}/sm_clock_mhz": clock,
        "gpu/all/util_pct": util,
        "gpu/all/mem_used_frac": mem_used / mem_total if mem_total > 0 else 0.0,
        "host/cpu_pct": cpu,
        "host/ram_used_gb": ram,
        "host/disk_free_gb": disk,
    }


class TestDrainBuffer:
    def test_drain_empty_when_no_samples_yet(self, fake_verl):
        sampler = _make_noop_sampler(fake_verl)
        assert sampler.drain() == {}

    def test_drain_resets_buffer(self, fake_verl):
        sampler = _make_noop_sampler(fake_verl)
        sampler._samples.append(_flat_sample())
        first = sampler.drain()
        assert first != {}
        assert sampler.drain() == {}


# ---------------------------------------------------------------------------
# GpuSampler.drain() — aggregation correctness
# ---------------------------------------------------------------------------

class TestDrainAggregation:
    def _sampler_with_samples(self, fake_verl) -> Any:
        sampler = _make_noop_sampler(fake_verl)

        # Two synthetic flat samples, 2 GPUs each.
        # Sample 1: gpu/0 util=60, gpu/1 util=80
        sampler._samples.append({
            "gpu/0/util_pct": 60.0, "gpu/0/mem_used_gb": 8.0, "gpu/0/mem_total_gb": 24.0,
            "gpu/0/power_w": 200.0, "gpu/0/temp_c": 70.0, "gpu/0/sm_clock_mhz": 1600.0,
            "gpu/1/util_pct": 80.0, "gpu/1/mem_used_gb": 12.0, "gpu/1/mem_total_gb": 24.0,
            "gpu/1/power_w": 250.0, "gpu/1/temp_c": 80.0, "gpu/1/sm_clock_mhz": 1800.0,
            "gpu/all/util_pct": 70.0,  # (60+80)/2
            "gpu/all/mem_used_frac": 12.0 / 24.0,  # max(8/24, 12/24)
            "host/cpu_pct": 20.0, "host/ram_used_gb": 16.0, "host/disk_free_gb": 100.0,
        })
        # Sample 2: gpu/0 util=80, gpu/1 util=90
        sampler._samples.append({
            "gpu/0/util_pct": 80.0, "gpu/0/mem_used_gb": 10.0, "gpu/0/mem_total_gb": 24.0,
            "gpu/0/power_w": 220.0, "gpu/0/temp_c": 75.0, "gpu/0/sm_clock_mhz": 1700.0,
            "gpu/1/util_pct": 90.0, "gpu/1/mem_used_gb": 14.0, "gpu/1/mem_total_gb": 24.0,
            "gpu/1/power_w": 260.0, "gpu/1/temp_c": 85.0, "gpu/1/sm_clock_mhz": 1900.0,
            "gpu/all/util_pct": 85.0,  # (80+90)/2
            "gpu/all/mem_used_frac": 14.0 / 24.0,  # max(10/24, 14/24)
            "host/cpu_pct": 30.0, "host/ram_used_gb": 18.0, "host/disk_free_gb": 95.0,
        })
        return sampler

    def test_per_device_util_mean_and_max(self, fake_verl):
        sampler = self._sampler_with_samples(fake_verl)
        result = sampler.drain()
        assert pytest.approx(result["gpu/0/util_pct/mean"], rel=1e-5) == 70.0  # (60+80)/2
        assert pytest.approx(result["gpu/0/util_pct/max"], rel=1e-5) == 80.0
        assert pytest.approx(result["gpu/1/util_pct/mean"], rel=1e-5) == 85.0  # (80+90)/2
        assert pytest.approx(result["gpu/1/util_pct/max"], rel=1e-5) == 90.0

    def test_per_device_mem_used_mean_and_max(self, fake_verl):
        sampler = self._sampler_with_samples(fake_verl)
        result = sampler.drain()
        assert pytest.approx(result["gpu/0/mem_used_gb/mean"], rel=1e-5) == 9.0   # (8+10)/2
        assert pytest.approx(result["gpu/0/mem_used_gb/max"], rel=1e-5) == 10.0
        assert pytest.approx(result["gpu/1/mem_used_gb/mean"], rel=1e-5) == 13.0  # (12+14)/2
        assert pytest.approx(result["gpu/1/mem_used_gb/max"], rel=1e-5) == 14.0

    def test_mem_total_no_aggregation_suffix(self, fake_verl):
        sampler = self._sampler_with_samples(fake_verl)
        result = sampler.drain()
        assert "gpu/0/mem_total_gb" in result
        assert pytest.approx(result["gpu/0/mem_total_gb"], rel=1e-5) == 24.0
        # Must NOT have mean/max suffixed variants
        assert "gpu/0/mem_total_gb/mean" not in result
        assert "gpu/0/mem_total_gb/max" not in result

    def test_per_device_power_mean(self, fake_verl):
        sampler = self._sampler_with_samples(fake_verl)
        result = sampler.drain()
        assert pytest.approx(result["gpu/0/power_w/mean"], rel=1e-5) == 210.0  # (200+220)/2
        assert pytest.approx(result["gpu/1/power_w/mean"], rel=1e-5) == 255.0  # (250+260)/2

    def test_per_device_temp_max(self, fake_verl):
        sampler = self._sampler_with_samples(fake_verl)
        result = sampler.drain()
        assert pytest.approx(result["gpu/0/temp_c/max"], rel=1e-5) == 75.0
        assert pytest.approx(result["gpu/1/temp_c/max"], rel=1e-5) == 85.0

    def test_per_device_sm_clock_mean(self, fake_verl):
        sampler = self._sampler_with_samples(fake_verl)
        result = sampler.drain()
        assert pytest.approx(result["gpu/0/sm_clock_mhz/mean"], rel=1e-5) == 1650.0
        assert pytest.approx(result["gpu/1/sm_clock_mhz/mean"], rel=1e-5) == 1850.0

    def test_all_devices_util_mean(self, fake_verl):
        sampler = self._sampler_with_samples(fake_verl)
        result = sampler.drain()
        # Across both samples, both devices: (60+80+80+90)/4 = 77.5
        assert pytest.approx(result["gpu/all/util_pct/mean"], rel=1e-5) == 77.5

    def test_all_devices_mem_used_frac_max(self, fake_verl):
        sampler = self._sampler_with_samples(fake_verl)
        result = sampler.drain()
        # max mem_used_gb across all devices both samples: gpu/1 sample2 = 14/24
        expected = 14.0 / 24.0
        assert pytest.approx(result["gpu/all/mem_used_frac/max"], rel=1e-5) == expected

    def test_host_cpu_mean(self, fake_verl):
        sampler = self._sampler_with_samples(fake_verl)
        result = sampler.drain()
        assert pytest.approx(result["host/cpu_pct/mean"], rel=1e-5) == 25.0  # (20+30)/2

    def test_host_ram_max(self, fake_verl):
        sampler = self._sampler_with_samples(fake_verl)
        result = sampler.drain()
        assert pytest.approx(result["host/ram_used_gb/max"], rel=1e-5) == 18.0

    def test_host_disk_free_last_value(self, fake_verl):
        sampler = self._sampler_with_samples(fake_verl)
        result = sampler.drain()
        # Last sample's disk_free_gb
        assert pytest.approx(result["host/disk_free_gb"], rel=1e-5) == 95.0


# ---------------------------------------------------------------------------
# install_tracking_hook — idempotency
# ---------------------------------------------------------------------------

class TestInstallTrackingHookIdempotency:
    def test_double_install_does_not_double_wrap(self, fake_verl):
        with patch.dict(sys.modules, {"pynvml": None}):
            with patch("subprocess.run", side_effect=FileNotFoundError):
                mod = _import_gpu_metrics()
                sampler = mod.GpuSampler(interval_s=5.0)

        Tracking = sys.modules["verl.utils.tracking"].Tracking

        # Clean slate: remove sentinel if present from previous import
        if hasattr(Tracking, "_gpu_metrics_hooked"):
            delattr(Tracking, "_gpu_metrics_hooked")
        # Reset to original method
        original_log = Tracking.log

        mod.install_tracking_hook(sampler)
        assert getattr(Tracking, "_gpu_metrics_hooked", False)

        patched_once = Tracking.log

        mod.install_tracking_hook(sampler)
        # Method object must be the same — no second wrapping
        assert Tracking.log is patched_once


# ---------------------------------------------------------------------------
# install_tracking_hook — merge behaviour
# ---------------------------------------------------------------------------

class TestInstallTrackingHookMerge:
    def _setup(self, fake_verl):
        with patch.dict(sys.modules, {"pynvml": None}):
            with patch("subprocess.run", side_effect=FileNotFoundError):
                mod = _import_gpu_metrics()
                sampler = mod.GpuSampler(interval_s=5.0)

        Tracking = sys.modules["verl.utils.tracking"].Tracking
        # Remove sentinel so hook installs fresh
        if hasattr(Tracking, "_gpu_metrics_hooked"):
            delattr(Tracking, "_gpu_metrics_hooked")

        return mod, sampler, Tracking

    def test_hook_merges_drain_output_into_data(self, fake_verl):
        mod, sampler, Tracking = self._setup(fake_verl)

        # Pre-load a flat sample so drain() returns something non-empty
        sampler._samples.append(_flat_sample(util=70.0, cpu=25.0))

        captured: list[dict] = []

        def spy_log(self_inner, data, step, backend=None):
            captured.append(dict(data))

        Tracking.log = spy_log
        mod.install_tracking_hook(sampler)

        fake_instance = Tracking()
        Tracking.log(fake_instance, {"reward": 1.0}, step=10)

        assert len(captured) == 1
        merged = captured[0]
        assert "reward" in merged
        assert any(k.startswith("gpu/") or k.startswith("host/") for k in merged)

    def test_hook_forwards_original_data_even_when_drain_raises(self, fake_verl):
        mod, sampler, Tracking = self._setup(fake_verl)

        captured: list[dict] = []

        def spy_log(self_inner, data, step, backend=None):
            captured.append(dict(data))

        Tracking.log = spy_log
        mod.install_tracking_hook(sampler)

        # Make drain() raise
        sampler.drain = MagicMock(side_effect=RuntimeError("nvml exploded"))

        fake_instance = Tracking()
        # Must not raise
        Tracking.log(fake_instance, {"reward": 0.5}, step=1)

        assert len(captured) == 1
        assert captured[0]["reward"] == 0.5
