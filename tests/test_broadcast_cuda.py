from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.fx_nvrtc import compile_module
from experiments.fx_nvrtc.nvrtc_driver import is_nvrtc_available


class BroadcastPattern(nn.Module):
    def forward(self, x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return F.relu(x + b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not is_nvrtc_available(), reason="NVRTC/CUDA driver libraries are required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize(
    "x_shape,b_shape",
    [
        ((16, 32, 64), (64,)),
        ((8, 16), (1, 16)),
        ((4, 3, 8, 16), (1, 3, 1, 16)),
        ((5, 7, 9), (9,)),
        ((2, 1, 11, 13), (2, 4, 1, 13)),
    ],
)
def test_broadcast_cuda_correctness(
    dtype: torch.dtype,
    x_shape: tuple[int, ...],
    b_shape: tuple[int, ...],
) -> None:
    model = BroadcastPattern().cuda().eval()
    compiled = compile_module(model)

    x = torch.randn(*x_shape, device="cuda", dtype=dtype)
    b = torch.randn(*b_shape, device="cuda", dtype=dtype)

    ref = model(x, b)
    out = compiled(x, b)
    rtol = 2e-2 if dtype == torch.float16 else 1e-5
    atol = 2e-2 if dtype == torch.float16 else 1e-5
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)
