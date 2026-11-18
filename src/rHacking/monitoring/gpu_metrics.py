from __future__ import annotations

import subprocess
import threading
import time
from collections import defaultdict
from typing import Optional


def _sample_pynvml(num_devices: int) -> Optional[dict[str, float]]:
    try:
        import pynvml  # type: ignore

        sample: dict[str, float] = {}
        util_vals: list[float] = []
        mem_frac_vals: list[float] = []

        for i in range(num_devices):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)

            u = float(util.gpu)
            mem_used = mem.used / 1024**3
            mem_total = mem.total / 1024**3
            frac = mem.used / mem.total if mem.total > 0 else 0.0

            sample[f"gpu/{i}/util_pct"] = u
            sample[f"gpu/{i}/mem_used_gb"] = mem_used
            sample[f"gpu/{i}/mem_total_gb"] = mem_total

            util_vals.append(u)
            mem_frac_vals.append(frac)

            try:
                sample[f"gpu/{i}/power_w"] = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            except pynvml.NVMLError:
                pass

            try:
                sample[f"gpu/{i}/temp_c"] = float(
                    pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                )
            except pynvml.NVMLError:
                pass

            try:
                sample[f"gpu/{i}/sm_clock_mhz"] = float(
                    pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
                )
            except pynvml.NVMLError:
                pass

        if util_vals:
            sample["gpu/all/util_pct"] = sum(util_vals) / len(util_vals)
        if mem_frac_vals:
            sample["gpu/all/mem_used_frac"] = max(mem_frac_vals)

        return sample
    except Exception:
        return None


def _sample_nvidia_smi() -> Optional[dict[str, float]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None

        sample: dict[str, float] = {}
        util_vals: list[float] = []
        mem_frac_vals: list[float] = []

        for i, line in enumerate(result.stdout.strip().splitlines()):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue

            try:
                u = float(parts[0])
                sample[f"gpu/{i}/util_pct"] = u
                util_vals.append(u)
            except ValueError:
                pass

            try:
                mem_used = float(parts[1]) / 1024.0  # MiB -> GiB
                sample[f"gpu/{i}/mem_used_gb"] = mem_used
            except ValueError:
                pass

            try:
                mem_total = float(parts[2]) / 1024.0
                sample[f"gpu/{i}/mem_total_gb"] = mem_total
                mem_used_val = sample.get(f"gpu/{i}/mem_used_gb")
                if mem_used_val is not None and mem_total > 0:
                    mem_frac_vals.append(mem_used_val / mem_total)
            except ValueError:
                pass

            try:
                if parts[3] != "[N/A]":
                    sample[f"gpu/{i}/power_w"] = float(parts[3])
            except ValueError:
                pass

            try:
                sample[f"gpu/{i}/temp_c"] = float(parts[4])
            except ValueError:
                pass

            try:
                sample[f"gpu/{i}/sm_clock_mhz"] = float(parts[5])
            except ValueError:
                pass

        if util_vals:
            sample["gpu/all/util_pct"] = sum(util_vals) / len(util_vals)
        if mem_frac_vals:
            sample["gpu/all/mem_used_frac"] = max(mem_frac_vals)

        return sample if sample else None
    except Exception:
        return None


def _sample_psutil() -> dict[str, float]:
    out: dict[str, float] = {}
    try:
        import psutil  # type: ignore

        out["host/cpu_pct"] = psutil.cpu_percent(interval=None)
        vm = psutil.virtual_memory()
        out["host/ram_used_gb"] = (vm.total - vm.available) / 1024**3
        out["host/disk_free_gb"] = psutil.disk_usage("/").free / 1024**3
    except Exception:
        pass
    return out


class GpuSampler:
    def __init__(self, interval_s: float = 5.0) -> None:
        self._interval = interval_s
        self._lock = threading.Lock()
        self._samples: list[dict[str, float]] = []
        self._backend: str = self._detect_backend()

        if self._backend != "none":
            t = threading.Thread(target=self._run, daemon=True)
            t.start()

    def _detect_backend(self) -> str:
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self._num_devices = pynvml.nvmlDeviceGetCount()
            if self._num_devices > 0:
                return "pynvml"
        except Exception:
            pass

        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return "nvidia-smi"
        except Exception:
            pass

        return "none"

    def _collect(self) -> dict[str, float]:
        sample: dict[str, float] = {}

        if self._backend == "pynvml":
            gpu_data = _sample_pynvml(self._num_devices)
            if gpu_data is not None:
                sample.update(gpu_data)
        elif self._backend == "nvidia-smi":
            gpu_data = _sample_nvidia_smi()
            if gpu_data is not None:
                sample.update(gpu_data)

        sample.update(_sample_psutil())
        return sample

    def _run(self) -> None:
        while True:
            try:
                sample = self._collect()
                if sample:
                    with self._lock:
                        self._samples.append(sample)
            except Exception:
                pass
            time.sleep(self._interval)

    def drain(self) -> dict[str, float]:
        with self._lock:
            samples = self._samples
            self._samples = []

        if not samples:
            return {}

        # Accumulate per key
        accum: dict[str, list[float]] = defaultdict(list)
        for s in samples:
            for k, v in s.items():
                accum[k].append(v)

        result: dict[str, float] = {}
        all_keys = set(accum.keys())

        # GPU per-device keys
        device_indices: set[int] = set()
        for k in all_keys:
            parts = k.split("/")
            if len(parts) >= 2 and parts[0] == "gpu" and parts[1] != "all":
                try:
                    device_indices.add(int(parts[1]))
                except ValueError:
                    pass

        for i in device_indices:
            prefix = f"gpu/{i}"

            def _emit(metric: str, do_mean: bool = True, do_max: bool = True) -> None:
                vals = accum.get(f"{prefix}/{metric}")
                if not vals:
                    return
                if do_mean:
                    result[f"{prefix}/{metric}/mean"] = sum(vals) / len(vals)
                if do_max:
                    result[f"{prefix}/{metric}/max"] = max(vals)

            _emit("util_pct")
            _emit("mem_used_gb")
            _emit("mem_total_gb", do_mean=False, do_max=False)
            # for mem_total_gb emit single value (constant per device)
            mem_total_vals = accum.get(f"{prefix}/mem_total_gb")
            if mem_total_vals:
                result[f"{prefix}/mem_total_gb"] = mem_total_vals[0]
            _emit("power_w", do_max=False)
            _emit("temp_c", do_mean=False)
            _emit("sm_clock_mhz", do_max=False)

        # GPU aggregate keys
        all_util = accum.get("gpu/all/util_pct")
        if all_util:
            result["gpu/all/util_pct/mean"] = sum(all_util) / len(all_util)

        all_mem_frac = accum.get("gpu/all/mem_used_frac")
        if all_mem_frac:
            result["gpu/all/mem_used_frac/max"] = max(all_mem_frac)

        # Host keys
        cpu = accum.get("host/cpu_pct")
        if cpu:
            result["host/cpu_pct/mean"] = sum(cpu) / len(cpu)

        ram = accum.get("host/ram_used_gb")
        if ram:
            result["host/ram_used_gb/max"] = max(ram)

        disk = accum.get("host/disk_free_gb")
        if disk:
            result["host/disk_free_gb"] = disk[-1]

        return result


def install_tracking_hook(sampler: GpuSampler) -> None:
    try:
        from verl.utils.tracking import Tracking  # type: ignore

        if getattr(Tracking, "_gpu_metrics_hooked", False):
            return

        original_log = Tracking.log

        def _patched_log(
            self: Tracking,
            data: dict,
            step: int,
            backend: Optional[str] = None,
        ) -> None:
            try:
                gpu_data = sampler.drain()
                if gpu_data:
                    data = {**data, **gpu_data}

                # Friendly aliases: verl uses "critic/..." for a critic-free GRPO run
                aliases = {
                    "critic/rewards/cheating_rate": "hacking/cheating_rate",
                    "critic/rewards/equation_accuracy": "hacking/equation_accuracy",
                }
                for src, dst in aliases.items():
                    if src in data:
                        data[dst] = data[src]
            except Exception:
                pass

            original_log(self, data, step, backend)

        Tracking.log = _patched_log  # type: ignore[method-assign]
        Tracking._gpu_metrics_hooked = True  # type: ignore[attr-defined]
    except Exception:
        pass
