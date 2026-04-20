"""Example runner for fused elementwise patterns."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.fx_nvrtc import compile_module
from experiments.fx_nvrtc.nvrtc_driver import is_nvrtc_available


class GeluPattern(nn.Module):
    def forward(self, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return F.gelu(x * w + b)


class ResidualPattern(nn.Module):
    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return F.relu(x + residual)


@dataclass(frozen=True)
class TimingResult:
    pattern: str
    eager_ms: float
    compiled_ms: float

    @property
    def speedup(self) -> float:
        return self.eager_ms / self.compiled_ms if self.compiled_ms > 0 else float("inf")


def _time_ms(fn, warmup: int = 30, iters: int = 200) -> float:
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / float(iters)


def _run_pattern(
    pattern: str,
    module: nn.Module,
    args: tuple[torch.Tensor, ...],
    rtol: float,
    atol: float,
) -> TimingResult:
    compiled = compile_module(module)
    y_eager = module(*args)
    y_compiled = compiled(*args)
    torch.testing.assert_close(y_compiled, y_eager, rtol=rtol, atol=atol)

    eager_ms = _time_ms(lambda: module(*args))
    compiled_ms = _time_ms(lambda: compiled(*args))
    return TimingResult(pattern=pattern, eager_ms=eager_ms, compiled_ms=compiled_ms)


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping example run.")
        return
    if not is_nvrtc_available():
        print("NVRTC/CUDA driver libraries not available. Skipping example run.")
        return

    device = torch.device("cuda")
    dtype = torch.float32
    x = torch.randn(2048, 2048, device=device, dtype=dtype)
    w = torch.randn_like(x)
    b = torch.randn_like(x)
    residual = torch.randn(x.shape[-1], device=device, dtype=dtype)

    gelu_result = _run_pattern(
        pattern="gelu(x*w+b)",
        module=GeluPattern().to(device).eval(),
        args=(x, w, b),
        rtol=1e-5,
        atol=1e-5,
    )
    residual_result = _run_pattern(
        pattern="relu(x+residual)",
        module=ResidualPattern().to(device).eval(),
        args=(x, residual),
        rtol=1e-5,
        atol=1e-5,
    )

    rows = [gelu_result, residual_result]
    print(f"{'pattern':<18} {'eager_ms':>10} {'compiled_ms':>12} {'speedup':>8}")
    print("-" * 52)
    for row in rows:
        print(f"{row.pattern:<18} {row.eager_ms:>10.4f} {row.compiled_ms:>12.4f} {row.speedup:>8.3f}")

    best = max(rows, key=lambda r: r.speedup)
    print(f"best_speedup={best.speedup:.3f} on {best.pattern}")


if __name__ == "__main__":
    main()
