from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from experiments.fx_nvrtc import compile_module
from experiments.fx_nvrtc.cache import KernelCache
from experiments.fx_nvrtc.launcher import KernelLauncher, LaunchRequest, LoadedKernel


class Identity(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class UnsupportedMatmul(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, y)


def test_import_and_fallback_compile() -> None:
    model = Identity()
    compiled = compile_module(model)
    x = torch.randn(4)
    y = compiled(x)
    assert torch.equal(x, y)


def test_kernel_cache_round_trip(tmp_path) -> None:
    cache = KernelCache(cache_dir=tmp_path)
    key = "abc123"
    payload = b"ptx-bytes"
    cache.put(key, payload)
    loaded = cache.get(key)
    assert loaded is not None
    assert loaded.ptx == payload


def test_unsupported_fx_graph_falls_back_to_eager(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    model = UnsupportedMatmul()
    compiled = compile_module(model)
    x = torch.randn(2, 3)
    y = torch.randn(3, 4)

    assert torch.equal(compiled(x, y), model(x, y))


def test_launcher_rejects_invalid_launch_request(monkeypatch, tmp_path) -> None:
    class FakeDriver:
        pass

    monkeypatch.setattr("experiments.fx_nvrtc.launcher.CUDANVRTCDriver", FakeDriver)
    launcher = KernelLauncher(cache=KernelCache(cache_dir=tmp_path))
    request = LaunchRequest(cache_key="k", source="extern C", kernel_name="kernel")

    with pytest.raises(ValueError, match="Output tensor must be on CUDA"):
        launcher.launch(request=request, inputs=[torch.randn(1)], output=torch.empty(1))


def test_launcher_compiles_loads_and_caches_kernel(monkeypatch, tmp_path) -> None:
    class FakeDriver:
        def __init__(self) -> None:
            self.compile_calls = 0
            self.load_calls = 0

        def compile_to_ptx(self, source: str, device_index: int) -> bytes:
            self.compile_calls += 1
            assert source == "source"
            assert device_index == 0
            return b"ptx"

        def load_kernel(self, ptx: bytes, kernel_name: str) -> LoadedKernel:
            self.load_calls += 1
            assert ptx == b"ptx"
            assert kernel_name == "kernel"
            return LoadedKernel(module=None, function=None)

    monkeypatch.setattr("experiments.fx_nvrtc.launcher.CUDANVRTCDriver", FakeDriver)
    launcher = KernelLauncher(cache=KernelCache(cache_dir=tmp_path))
    request = LaunchRequest(cache_key="cache-key", source="source", kernel_name="kernel")

    first = launcher._ensure_loaded(request=request, device_index=0)
    second = launcher._ensure_loaded(request=request, device_index=0)

    assert first is second
    assert launcher.driver.compile_calls == 1
    assert launcher.driver.load_calls == 1
