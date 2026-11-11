import pytest
import torch

from autodeep import device


def test_cpu_optimizer_step_does_not_initialise_cuda():
    device.patch_torch_cpu_optimizers()
    device.patch_torch_cpu_optimizers()  # idempotent
    was_initialised = torch.cuda.is_initialized()
    model = torch.nn.Linear(4, 1)
    opt = torch.optim.Adam(model.parameters())
    model(torch.randn(8, 4)).sum().backward()
    opt.step()
    if torch.cuda.is_available():
        # PyTorch's graph-capture health check would otherwise create a CUDA context here,
        # which fails with "out of memory" when another program fills the GPU
        assert torch.cuda.is_initialized() == was_initialised


def test_set_device_validation(monkeypatch):
    monkeypatch.delenv("AUTODEEP_DEVICE", raising=False)
    device.set_device("cpu")
    assert device.resolve_device() == "cpu" and device.xgboost_device() == "cpu" and device.catboost_task_type() == "CPU"
    with pytest.raises(ValueError):
        device.set_device("tpu")
    device.set_device("auto")
