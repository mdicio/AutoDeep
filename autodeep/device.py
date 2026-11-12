"""Device selection.

``device="auto"`` uses the GPU only when CUDA is available *and* has enough free
memory; otherwise everything runs on CPU.

Two details make this robust on machines where the GPU is shared with other programs:

* free memory is read with ``nvidia-smi`` (no CUDA context is created);
* recent PyTorch versions create a CUDA context inside every ``optimizer.step()``
  (a graph-capture health check) even when all parameters are on the CPU, which fails with
  "CUDA error: out of memory" when another process fills the GPU. :func:`patch_torch_cpu_optimizers`
  skips that check for CPU-only optimizers. (Hiding the GPU with ``CUDA_VISIBLE_DEVICES=""``
  is not an option: CUDA builds of XGBoost 3.x then fail on multiclass problems.)
* CatBoost aborts the whole process on CUDA OOM, so it runs on CPU unless the GPU is
  explicitly requested.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from functools import lru_cache

MIN_FREE_GPU_MB = int(os.environ.get("AUTODEEP_MIN_FREE_GPU_MB", "2048"))
_override: str | None = None


def set_device(device: str | None) -> None:
    """Choose the device for this process: ``"auto"`` (default), ``"cpu"`` or ``"cuda"``."""
    global _override
    if device not in (None, "auto", "cpu", "cuda", "gpu"):
        raise ValueError("device must be 'auto', 'cpu' or 'cuda'")
    _override = None if device in (None, "auto") else ("cuda" if device == "gpu" else device)
    resolve_device.cache_clear()


@lru_cache(maxsize=1)
def resolve_device() -> str:
    """Return ``"cuda"`` or ``"cpu"`` for torch-based models."""
    if _override:
        return _override
    env = os.environ.get("AUTODEEP_DEVICE")
    if env in ("cpu", "cuda"):
        return env
    info = gpu_info()
    if info.get("available") and info.get("free_mb", 0) >= MIN_FREE_GPU_MB:
        return "cuda"
    return "cpu"


def apply_device_policy() -> str:
    """Resolve the device for this process and make CPU training safe on a busy GPU. Call before training."""
    patch_torch_cpu_optimizers()
    return resolve_device()


def patch_torch_cpu_optimizers() -> None:
    """Skip PyTorch's per-step graph-capture health check when every parameter lives on the CPU.

    The check calls ``torch.accelerator.current_stream()`` whenever CUDA is *available*, creating a CUDA
    context (and failing if the GPU is full) even though nothing runs on the GPU. GPU optimizers keep the
    original behaviour.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover
        return
    for name in ("_accelerator_graph_capture_health_check", "_cuda_graph_capture_health_check"):
        original = getattr(torch.optim.Optimizer, name, None)
        if original is None or getattr(original, "_autodeep_patched", False):
            continue

        def checked(self, _original=original):
            for group in self.param_groups:
                for p in group["params"]:
                    if p.device.type != "cpu":
                        return _original(self)

        checked._autodeep_patched = True
        setattr(torch.optim.Optimizer, name, checked)


def _torch_built_with_cuda() -> bool:
    try:
        import torch

        return torch.version.cuda is not None
    except ImportError:  # pragma: no cover
        return False


@lru_cache(maxsize=1)
def _nvidia_smi_path():
    return shutil.which("nvidia-smi")


def _nvidia_smi():
    """``[{"name", "free_mb", "total_mb"}]`` from nvidia-smi, honouring CUDA_VISIBLE_DEVICES (empty list if unavailable)."""
    exe = _nvidia_smi_path()
    if not exe:
        return []
    try:
        out = subprocess.run(
            [exe, "--query-gpu=index,name,memory.free,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4 and parts[2].isdigit():
            gpus.append({"index": parts[0], "name": parts[1], "free_mb": int(parts[2]), "total_mb": int(parts[3])})
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        allowed = [v.strip() for v in visible.split(",") if v.strip()]
        gpus = [g for g in gpus if g["index"] in allowed]
    return gpus


def gpu_info() -> dict:
    """Information about the first visible GPU, without initialising CUDA when possible."""
    if not _torch_built_with_cuda():
        return {"available": False, "reason": "PyTorch was installed without CUDA support"}
    gpus = _nvidia_smi()
    if gpus:
        g = gpus[0]
        return {"available": True, "name": g["name"], "free_mb": g["free_mb"], "total_mb": g["total_mb"]}
    if _nvidia_smi_path():  # nvidia-smi exists but no visible GPU
        return {"available": False}
    try:  # no nvidia-smi: fall back to torch (initialises CUDA)
        import torch

        if not torch.cuda.is_available():
            return {"available": False}
        free, total = torch.cuda.mem_get_info()
        return {"available": True, "name": torch.cuda.get_device_name(0), "free_mb": int(free / 2**20), "total_mb": int(total / 2**20)}
    except Exception as e:  # pragma: no cover - driver issues / broken install
        return {"available": False, "error": str(e)}


def catboost_task_type() -> str:
    """CatBoost runs on CPU unless the GPU was explicitly requested (it is safer and usually as fast on tabular data)."""
    return "GPU" if (_override == "cuda" or os.environ.get("AUTODEEP_DEVICE") == "cuda") else "CPU"


def xgboost_device() -> str:
    return "cuda" if resolve_device() == "cuda" else "cpu"


def cpu_threads(n_rows: int) -> int:
    """Threads for XGBoost / CatBoost.

    Using every hardware thread is counter-productive on small data and catastrophic when the machine is
    busy (OpenMP spin-waiting: we measured 0.3s -> 76s per XGBoost fit on 120 rows). Scale with the data,
    cap at the number of physical cores. Override with ``AUTODEEP_THREADS``.
    """
    env = os.environ.get("AUTODEEP_THREADS")
    if env and env.isdigit() and int(env) > 0:
        return int(env)
    physical = max(1, (os.cpu_count() or 2) // 2)
    return int(max(1, min(physical, 8, n_rows // 4000)))


def torch_threads(n_rows: int) -> int:
    """Intra-op threads for CPU training of the (small) tabular networks: 2 is fastest below ~10k rows."""
    env = os.environ.get("AUTODEEP_THREADS")
    if env and env.isdigit() and int(env) > 0:
        return int(env)
    physical = max(1, (os.cpu_count() or 2) // 2)
    return int(max(1, min(physical, max(2, n_rows // 10000))))


def default_num_workers(n_rows: int) -> int:
    """DataLoader workers: in-process loading is fastest for small/medium tabular data."""
    if n_rows < 200_000:
        return 0
    return max(1, min(4, (os.cpu_count() or 2) // 2))
