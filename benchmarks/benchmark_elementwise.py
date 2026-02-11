"""Benchmark script for fused elementwise chains."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from compiler import compile_module
from runtime.nvrtc_driver import is_nvrtc_available


class GeluMLP(nn.Module):
    def forward(self, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return F.gelu(x * w + b)


class ReluResidual(nn.Module):
    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return F.relu(x + residual)


@dataclass(frozen=True)
class BenchRow:
    pattern: str
    shape: str
    dtype: str
    eager_ms: float
    compiled_ms: float

    @property
    def speedup(self) -> float:
        return self.eager_ms / self.compiled_ms if self.compiled_ms > 0 else float("inf")


def _time_ms(fn: Callable[[], torch.Tensor], warmup: int = 30, iters: int = 200) -> float:
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


def _fmt_shape(shape: Sequence[int]) -> str:
    return "x".join(str(x) for x in shape)


def run_benchmark() -> list[BenchRow]:
    rows: list[BenchRow] = []
    device = torch.device("cuda")
    shapes = [
        (256, 256),
        (512, 512),
        (1024, 1024),
        (2048, 512),
        (4096, 256),
        (128, 4096),
        (64, 16384),
        (32, 32768),
    ]

    patterns: list[tuple[str, nn.Module, Callable[[tuple[int, ...], torch.dtype], tuple[torch.Tensor, ...]]]] = [
        (
            "gelu(x*w+b)",
            GeluMLP().to(device).eval(),
            lambda shape, dtype: (
                torch.randn(*shape, device=device, dtype=dtype),
                torch.randn(*shape, device=device, dtype=dtype),
                torch.randn(*shape, device=device, dtype=dtype),
            ),
        ),
        (
            "relu(x+residual)",
            ReluResidual().to(device).eval(),
            lambda shape, dtype: (
                torch.randn(*shape, device=device, dtype=dtype),
                torch.randn(shape[-1], device=device, dtype=dtype),
            ),
        ),
    ]

    for dtype in (torch.float32, torch.float16):
        for pattern_name, module, make_args in patterns:
            compiled = compile_module(module)
            for shape in shapes:
                args = make_args(shape, dtype)
                compiled(*args)  # trigger parse+compile once before timing
                eager_ms = _time_ms(lambda: module(*args))
                compiled_ms = _time_ms(lambda: compiled(*args))
                rows.append(
                    BenchRow(
                        pattern=pattern_name,
                        shape=_fmt_shape(shape),
                        dtype="fp16" if dtype == torch.float16 else "fp32",
                        eager_ms=eager_ms,
                        compiled_ms=compiled_ms,
                    )
                )
    return rows


def _print_table(rows: Sequence[BenchRow]) -> None:
    header = f"{'pattern':<20} {'shape':<14} {'dtype':<6} {'eager_ms':>10} {'compiled_ms':>12} {'speedup':>8}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row.pattern:<20} {row.shape:<14} {row.dtype:<6} "
            f"{row.eager_ms:>10.4f} {row.compiled_ms:>12.4f} {row.speedup:>8.3f}"
        )


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping benchmarks.")
        return
    if not is_nvrtc_available():
        print("NVRTC/CUDA driver libraries not available. Skipping benchmarks.")
        return

    rows = run_benchmark()
    _print_table(rows)


if __name__ == "__main__":
    main()
