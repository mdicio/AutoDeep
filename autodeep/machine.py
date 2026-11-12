"""What is this machine capable of? CPU cores, RAM, GPU and a quick speed measurement.

Used by the automatic plan (:mod:`autodeep.autoplan`) and to scale the runtime estimates before any run
history exists on this machine. Nothing here imports torch.

The speed factor comes from a ~0.2 s micro-benchmark (BLAS matrix products, sorting, a Python loop)
compared with the machine the built-in cost curves were measured on (a 6-core / 12-thread Ryzen 5 3600).
A factor of 2 means "about twice as slow". It is cached in ``~/.cache/autodeep/machine.json`` for 30 days.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional

# best-of-5 timings of _benchmark() on the reference machine (AMD Ryzen 5 3600, 6 cores / 12 threads)
REFERENCE_TIMES = {"matmul": 0.0100, "sort": 0.0220, "python": 0.0217}
BENCHMARK_WEIGHTS = {"matmul": 0.4, "sort": 0.3, "python": 0.3}
REFERENCE_PHYSICAL_CORES = 6
CACHE_DAYS = 30


def _cache_file() -> Path:
    return Path(os.environ.get("AUTODEEP_CACHE", Path.home() / ".cache" / "autodeep")) / "machine.json"


@dataclass
class MachineProfile:
    os: str
    cpu_name: str
    logical_cores: int
    physical_cores: int
    ram_total_gb: Optional[float]
    ram_available_gb: Optional[float]
    gpu: Dict = field(default_factory=dict)
    device: str = "cpu"  # what device="auto" resolves to right now
    speed_factor: float = 1.0  # > 1: slower than the reference machine
    speed_source: str = "default"

    def to_dict(self) -> Dict:
        return asdict(self)

    def describe(self) -> str:
        ram = f", {self.ram_total_gb:.0f} GB RAM ({self.ram_available_gb:.0f} GB free)" if self.ram_total_gb else ""
        gpu = self.gpu or {}
        if gpu.get("available"):
            g = f"GPU {gpu.get('name', '?')} with {gpu.get('free_mb', 0) / 1024:.1f} of {gpu.get('total_mb', 0) / 1024:.0f} GB free"
            g += "" if self.device == "cuda" else " (too little free memory: training on CPU)"
        else:
            g = "no usable GPU"
        speed = f"CPU speed x{1 / self.speed_factor:.2f} of the reference machine" if self.speed_source == "benchmark" else ""
        return f"{self.cpu_name}: {self.physical_cores} cores / {self.logical_cores} threads{ram}; {g}" + (f"; {speed}" if speed else "")


# --------------------------------------------------------------------------------------
# probes
# --------------------------------------------------------------------------------------


def _cpu_name() -> str:
    try:
        if sys.platform.startswith("linux"):
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        if sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5).stdout.strip() or "CPU"
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or platform.machine() or "CPU"


def _physical_cores(logical: int) -> int:
    try:
        import psutil  # optional

        n = psutil.cpu_count(logical=False)
        if n:
            return int(n)
    except ImportError:
        pass
    if sys.platform.startswith("linux"):
        try:
            cores = set()
            phys = core = None
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("physical id"):
                    phys = line.split(":")[1].strip()
                elif line.startswith("core id"):
                    core = line.split(":")[1].strip()
                    cores.add((phys, core))
            if cores:
                return len(cores)
        except OSError:
            pass
    return max(1, logical // 2)


def _ram_gb():
    """``(total, available)`` in GB, or ``(None, None)``."""
    try:
        import psutil  # optional

        vm = psutil.virtual_memory()
        return vm.total / 2**30, vm.available / 2**30
    except ImportError:
        pass
    try:
        if sys.platform.startswith("linux"):
            info = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                key, value = line.split(":", 1)
                info[key] = int(value.strip().split()[0]) / 2**20  # kB -> GB
            return info.get("MemTotal"), info.get("MemAvailable", info.get("MemFree"))
        if sys.platform == "darwin":
            total = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5).stdout.strip()) / 2**30
            return total, None
        if sys.platform == "win32":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
                    (name, ctypes.c_ulonglong)
                    for name in (
                        "ullTotalPhys",
                        "ullAvailPhys",
                        "ullTotalPageFile",
                        "ullAvailPageFile",
                        "ullTotalVirtual",
                        "ullAvailVirtual",
                        "ullExtended",
                    )
                ]

            st = MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return st.ullTotalPhys / 2**30, st.ullAvailPhys / 2**30
    except (OSError, ValueError, subprocess.SubprocessError, AttributeError):
        pass
    return None, None


def _benchmark() -> Dict[str, float]:
    """Best-of-5 timings of three small workloads (~0.3 s in total on the reference machine)."""
    import numpy as np

    rng = np.random.default_rng(0)
    a = rng.standard_normal((400, 400))
    x = rng.standard_normal(300_000)

    def matmul():
        for _ in range(15):
            a @ a

    def sort():
        for _ in range(6):
            np.sort(x)

    def python():
        s = 0
        for i in range(400_000):
            s += i * i

    # warm up (BLAS thread pool, CPU clock ramping up from idle) for ~0.25 s
    t_end = time.perf_counter() + 0.25
    while time.perf_counter() < t_end:
        matmul()
        python()
    out = {}
    for name, fn in (("matmul", matmul), ("sort", sort), ("python", python)):
        best = float("inf")
        for _ in range(5):
            t = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t)
        out[name] = best
    return out


def speed_factor_from(times: Dict[str, float]) -> float:
    """Weighted geometric mean of ``time / reference_time`` (clipped to 0.25 - 8)."""
    import math

    log = sum(w * math.log(max(times[k], 1e-6) / REFERENCE_TIMES[k]) for k, w in BENCHMARK_WEIGHTS.items())
    return float(min(max(math.exp(log), 0.25), 8.0))


def _machine_key(cpu: str, logical: int) -> str:
    return f"{socket.gethostname()}|{cpu}|{logical}"


_SPEED_MEMO: Dict[str, tuple] = {}


def measure_speed(cpu: str, logical: int, refresh: bool = False):
    """``(factor, source)``: cached benchmark result for this machine, measured if missing or stale."""
    key = _machine_key(cpu, logical)
    if not refresh and key in _SPEED_MEMO:
        return _SPEED_MEMO[key]
    _SPEED_MEMO[key] = result = _measure_speed(cpu, logical, refresh)
    return result


def _measure_speed(cpu: str, logical: int, refresh: bool):
    env = os.environ.get("AUTODEEP_SPEED_FACTOR")
    if env:
        try:
            return float(env), "AUTODEEP_SPEED_FACTOR"
        except ValueError:
            pass
    path, key = _cache_file(), _machine_key(cpu, logical)
    if not refresh:
        try:
            cached = json.loads(path.read_text())
            if cached.get("key") == key and time.time() - cached.get("time", 0) < CACHE_DAYS * 86400:
                return float(cached["speed_factor"]), "benchmark"
        except (OSError, ValueError, KeyError):
            pass
    try:
        times = _benchmark()
        if speed_factor_from(times) > 1.25:  # a cold start or a busy moment: measure again, keep the faster times
            again = _benchmark()
            times = {k: min(v, again[k]) for k, v in times.items()}
    except Exception:  # pragma: no cover - numpy issues
        return 1.0, "default"
    factor = speed_factor_from(times)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"key": key, "time": time.time(), "speed_factor": factor, "times": times}))
    except OSError:
        pass
    return factor, "benchmark"


def _logical_cores() -> int:
    try:
        return len(os.sched_getaffinity(0))  # respects taskset / container CPU limits
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def current_speed_factor() -> float:
    """This machine's speed factor (benchmark cached on disk; 1.0 if it cannot be measured)."""
    try:
        return measure_speed(_cpu_name(), _logical_cores())[0]
    except Exception:  # pragma: no cover - estimation is best effort
        return 1.0


def machine_profile(benchmark: bool = True) -> MachineProfile:
    """Profile of this machine. Free RAM / GPU memory are read on every call; the speed benchmark is cached."""
    from autodeep.device import gpu_info, resolve_device

    logical = _logical_cores()
    cpu = _cpu_name()
    total, avail = _ram_gb()
    factor, source = measure_speed(cpu, logical) if benchmark else (1.0, "default")
    return MachineProfile(
        os=f"{platform.system()} {platform.release()}",
        cpu_name=cpu,
        logical_cores=logical,
        physical_cores=min(_physical_cores(logical), logical),
        ram_total_gb=None if total is None else round(total, 1),
        ram_available_gb=None if avail is None else round(avail, 1),
        gpu=gpu_info(),
        device=resolve_device(),
        speed_factor=round(factor, 3),
        speed_source=source,
    )
