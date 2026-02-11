from __future__ import annotations

import torch
import torch.nn as nn

from compiler import compile_module
from runtime.cache import KernelCache


class Identity(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


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
