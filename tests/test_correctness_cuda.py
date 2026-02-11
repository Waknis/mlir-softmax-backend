from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from compiler import compile_module
from runtime.nvrtc_driver import is_nvrtc_available


class MLPPattern(nn.Module):
    def forward(self, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return F.gelu(x * w + b)


class ResidualPattern(nn.Module):
    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return F.relu(x + residual)


def _rand(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    return torch.randn(*shape, device="cuda", dtype=dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not is_nvrtc_available(), reason="NVRTC/CUDA driver libraries are required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("shape", [(128, 256), (33, 129), (7, 11, 64)])
def test_gelu_chain_cuda_correctness(dtype: torch.dtype, shape: tuple[int, ...]) -> None:
    model = MLPPattern().cuda().eval()
    compiled = compile_module(model)

    x = _rand(shape, dtype=dtype)
    w = _rand(shape, dtype=dtype)
    b = _rand(shape, dtype=dtype)

    ref = model(x, w, b)
    out = compiled(x, w, b)
    rtol = 2e-2 if dtype == torch.float16 else 1e-5
    atol = 2e-2 if dtype == torch.float16 else 1e-5
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not is_nvrtc_available(), reason="NVRTC/CUDA driver libraries are required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("shape", [(64, 64), (31, 513), (4, 9, 17, 33)])
def test_residual_chain_cuda_correctness(dtype: torch.dtype, shape: tuple[int, ...]) -> None:
    model = ResidualPattern().cuda().eval()
    compiled = compile_module(model)

    x = _rand(shape, dtype=dtype)
    residual = _rand(shape, dtype=dtype)

    ref = model(x, residual)
    out = compiled(x, residual)
    rtol = 2e-2 if dtype == torch.float16 else 1e-5
    atol = 2e-2 if dtype == torch.float16 else 1e-5
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)
