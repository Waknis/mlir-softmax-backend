#!/usr/bin/env python3
"""Optional Triton benchmark for the repo's current softmax-normalization scope.

This benchmarks the same operation the MLIR backend lowers today:

    y[i] = x[i] / sum

It is intentionally not full safe softmax with exp/reduce, because the current
MLIR pipeline also models only the divide-and-normalize stage.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from typing import Iterable

_IMPORT_ERROR: Exception | None = None
try:
    import torch
    import triton
    import triton.language as tl
except Exception as exc:  # pragma: no cover - optional dependency path
    torch = None
    triton = None
    tl = None
    _IMPORT_ERROR = exc


if triton is not None:

    @triton.jit
    def normalize_kernel(x_ptr, y_ptr, n_elements, denom, BLOCK: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = x / denom
        tl.store(y_ptr + offsets, y, mask=mask)


@dataclass
class TritonBenchmarkResult:
    size: int
    avg_kernel_ms: float
    effective_gib_s: float
    max_abs_err: float


def _parse_sizes(raw_sizes: str) -> list[int]:
    sizes: list[int] = []
    for token in raw_sizes.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError(f"size must be positive: {token}")
        sizes.append(value)
    return sizes


def benchmark_sizes(
    sizes: Iterable[int],
    warmup: int,
    iters: int,
    denom: float,
) -> list[TritonBenchmarkResult]:
    if _IMPORT_ERROR is not None:
        raise RuntimeError(
            "torch and triton are required for Triton benchmarking"
        ) from _IMPORT_ERROR
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Triton benchmarking.")
    if iters <= 0:
        raise ValueError("iters must be greater than zero")

    results: list[TritonBenchmarkResult] = []
    for size in sizes:
        x = torch.empty(size, device="cuda", dtype=torch.float32)
        base = torch.arange(size, device="cuda", dtype=torch.float32)
        x.copy_(1.0 + (base % 19) * 0.03125)
        y = torch.empty_like(x)

        block = min(1024, triton.next_power_of_2(size))
        grid = lambda meta: (triton.cdiv(size, meta["BLOCK"]),)

        for _ in range(warmup):
            normalize_kernel[grid](x, y, size, denom, BLOCK=block)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        total_ms = 0.0
        for _ in range(iters):
            start.record()
            normalize_kernel[grid](x, y, size, denom, BLOCK=block)
            stop.record()
            stop.synchronize()
            total_ms += start.elapsed_time(stop)

        y_host = y.cpu()
        expected = (x / denom).cpu()
        max_abs_err = torch.max(torch.abs(y_host - expected)).item()
        avg_ms = total_ms / float(iters)
        moved_bytes = float(size * 2 * 4)
        effective_gib_s = moved_bytes / (avg_ms / 1000.0) / (1024.0**3)
        results.append(
            TritonBenchmarkResult(
                size=size,
                avg_kernel_ms=avg_ms,
                effective_gib_s=effective_gib_s,
                max_abs_err=max_abs_err,
            )
        )

    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark a Triton vector normalization kernel."
    )
    parser.add_argument(
        "--sizes",
        default="1024,4096,16384,65536",
        help="Comma-separated vector sizes to benchmark.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=25,
        help="Untimed warmup iterations per size.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=100,
        help="Timed iterations per size.",
    )
    parser.add_argument(
        "--sum",
        type=float,
        default=4.0,
        help="Normalization denominator.",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Emit CSV instead of a tab-separated table.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    sizes = _parse_sizes(args.sizes)
    results = benchmark_sizes(sizes, args.warmup, args.iters, args.sum)

    if args.csv:
        writer = csv.writer(sys.stdout)
        writer.writerow(
            [
                "backend",
                "size",
                "warmup",
                "iters",
                "avg_kernel_ms",
                "effective_gib_s",
                "max_abs_err",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    "triton",
                    result.size,
                    args.warmup,
                    args.iters,
                    f"{result.avg_kernel_ms:.6f}",
                    f"{result.effective_gib_s:.6f}",
                    f"{result.max_abs_err:.8f}",
                ]
            )
        return 0

    print(
        "backend\tsize\twarmup\titers\tavg_kernel_ms\t"
        "effective_gib_s\tmax_abs_err"
    )
    for result in results:
        print(
            "triton\t"
            f"{result.size}\t{args.warmup}\t{args.iters}\t"
            f"{result.avg_kernel_ms:.6f}\t{result.effective_gib_s:.6f}\t"
            f"{result.max_abs_err:.8f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
