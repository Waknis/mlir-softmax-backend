"""GPU softmax benchmark harness.

Measures row-wise softmax throughput for the hand-written CUDA kernels
(online and naive), the Triton reference, and optionally PyTorch as a
reference. Timing uses CUDA events with per-iteration start/stop records;
warmup is untimed; reported stats are median and p95 across iters.

The old ``softmax_benchmark.cpp`` in this directory measures *compiler pass
time* and counts divf ops statically -- it does not measure runtime. This
harness is the one to quote in the README Results table.

Usage:
    python -m benchmarks.softmax_gpu_bench                       # sweep default shapes
    python -m benchmarks.softmax_gpu_bench --shapes 1024x4096    # single shape
    python -m benchmarks.softmax_gpu_bench --include-torch       # add torch reference column
    python -m benchmarks.softmax_gpu_bench --md docs/results.md  # write markdown table
    python -m benchmarks.softmax_gpu_bench --csv docs/results.csv

"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from kernels.softmax_loader import SoftmaxKernelUnavailable, get_library  # noqa: E402
from benchmarks.triton_softmax import HAS_TRITON, softmax_triton  # noqa: E402


BackendFn = Callable[[torch.Tensor, torch.Tensor], None]


@dataclass(frozen=True)
class Shape:
    rows: int
    cols: int

    @classmethod
    def parse(cls, s: str) -> "Shape":
        parts = s.lower().split("x")
        if len(parts) != 2:
            raise ValueError(f"expected RxC, got {s!r}")
        return cls(int(parts[0]), int(parts[1]))

    def __str__(self) -> str:
        return f"{self.rows}x{self.cols}"


@dataclass
class BenchmarkResult:
    shape: str
    dtype: str
    backend: str
    iters: int
    ms_median: float
    ms_p95: float
    ms_stddev: float
    bandwidth_gbs: float
    pct_of_peak: float


DEFAULT_SHAPES = [
    Shape(256, 256),
    Shape(512, 1024),
    Shape(1024, 1024),
    Shape(1024, 4096),
    Shape(2048, 2048),
    Shape(4096, 4096),
    Shape(8192, 8192),
]

# RTX 5060 Ti nominal memory bandwidth (GDDR7, 128-bit bus). Override with
# --peak-bw-gbs for other GPUs.
DEFAULT_PEAK_BW_GBS = 448.0


def _make_backends(
    include_torch: bool,
) -> Tuple[Dict[str, BackendFn], Dict[str, torch.dtype]]:
    backends: Dict[str, BackendFn] = {}
    dtypes: Dict[str, torch.dtype] = {}

    try:
        lib = get_library()
    except SoftmaxKernelUnavailable as e:
        print(f"warning: hand kernels unavailable ({e})", file=sys.stderr)
        lib = None

    if lib is not None:
        backends["hand_online_f32"] = lib.online_f32
        dtypes["hand_online_f32"] = torch.float32
        backends["hand_naive_f32"] = lib.naive_f32
        dtypes["hand_naive_f32"] = torch.float32
        backends["hand_online_f16"] = lib.online_f16
        dtypes["hand_online_f16"] = torch.float16

    if HAS_TRITON:
        def triton_f32(x: torch.Tensor, y: torch.Tensor) -> None:
            softmax_triton(x, y)

        backends["triton_f32"] = triton_f32
        dtypes["triton_f32"] = torch.float32
    else:
        print("warning: triton not installed -- skipping triton column", file=sys.stderr)

    if include_torch:
        def torch_f32(x: torch.Tensor, y: torch.Tensor) -> None:
            y.copy_(torch.softmax(x, dim=-1))

        backends["torch_f32"] = torch_f32
        dtypes["torch_f32"] = torch.float32

    return backends, dtypes


def _verify_correctness(
    name: str, fn: BackendFn, x: torch.Tensor, dtype: torch.dtype
) -> float:
    y = torch.empty_like(x)
    fn(x, y)
    ref = torch.softmax(x.float(), dim=-1)
    diff = (y.float() - ref).abs().max().item()
    atol = 1e-4 if dtype == torch.float32 else 5e-3
    if diff > atol:
        raise RuntimeError(
            f"{name}: correctness check failed, max_abs_diff={diff:.3e} > {atol:.0e}"
        )
    return diff


# L2 flush buffer: sized to evict the RTX 5060 Ti's 32 MB L2 (overshooting is
# fine; undershooting leaves softmax inputs cached across iterations, which
# inflates measured bandwidth above physical DRAM peak on shapes small enough
# to fit in L2). Zeroing this between each timed launch forces every kernel
# to hit DRAM — the number we actually want for a roofline comparison.
_L2_FLUSH_BYTES = 64 * 1024 * 1024


def _time_backend(
    fn: BackendFn, x: torch.Tensor, y: torch.Tensor,
    warmup: int, iters: int, flush_buf: Optional[torch.Tensor],
) -> List[float]:
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn(x, y)
    torch.cuda.synchronize()

    times: List[float] = []
    for _ in range(iters):
        if flush_buf is not None:
            flush_buf.zero_()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(x, y)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return times


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1)))))
    return s[k]


def _run_shape(
    shape: Shape,
    backends: Dict[str, BackendFn],
    dtypes: Dict[str, torch.dtype],
    warmup: int,
    iters: int,
    flush_l2: bool,
    peak_bw_gbs: float,
    seed: int,
) -> List[BenchmarkResult]:
    results: List[BenchmarkResult] = []
    torch.manual_seed(seed)

    flush_buf: Optional[torch.Tensor] = None
    if flush_l2:
        flush_buf = torch.empty(_L2_FLUSH_BYTES, dtype=torch.int8, device="cuda")

    for name, fn in backends.items():
        dtype = dtypes[name]
        x = torch.randn(shape.rows, shape.cols, device="cuda", dtype=dtype)
        y = torch.empty_like(x)

        _verify_correctness(name, fn, x, dtype)

        times = _time_backend(fn, x, y, warmup=warmup, iters=iters, flush_buf=flush_buf)
        median_ms = statistics.median(times)
        p95_ms = _percentile(times, 95.0)
        stddev_ms = statistics.pstdev(times) if len(times) > 1 else 0.0

        bytes_per_iter = 2 * shape.rows * shape.cols * (4 if dtype == torch.float32 else 2)
        gbs = bytes_per_iter / (median_ms * 1e-3) / 1e9
        pct = 100.0 * gbs / peak_bw_gbs

        results.append(BenchmarkResult(
            shape=str(shape),
            dtype="f32" if dtype == torch.float32 else "f16",
            backend=name,
            iters=iters,
            ms_median=median_ms,
            ms_p95=p95_ms,
            ms_stddev=stddev_ms,
            bandwidth_gbs=gbs,
            pct_of_peak=pct,
        ))

    return results


def _print_hardware_banner(peak_bw_gbs: float) -> None:
    dev = torch.cuda.get_device_properties(0)
    print(f"device : {dev.name} (SM {dev.major}.{dev.minor}, "
          f"{dev.total_memory // (1024**2)} MiB)")
    print(f"torch  : {torch.__version__} cuda={torch.version.cuda}")
    print(f"peak_bw: {peak_bw_gbs:.0f} GB/s (override with --peak-bw-gbs)")
    print()


def _print_table(results: List[BenchmarkResult]) -> None:
    header = (
        f"{'shape':>10}  {'dtype':>4}  {'backend':<18}  "
        f"{'ms (med)':>10}  {'ms (p95)':>10}  {'stddev':>8}  "
        f"{'GB/s':>8}  {'%peak':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.shape:>10}  {r.dtype:>4}  {r.backend:<18}  "
            f"{r.ms_median:>10.4f}  {r.ms_p95:>10.4f}  "
            f"{r.ms_stddev:>8.4f}  {r.bandwidth_gbs:>8.1f}  {r.pct_of_peak:>5.1f}%"
        )


def _write_csv(results: List[BenchmarkResult], path: Path) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        w.writeheader()
        for r in results:
            w.writerow(asdict(r))


def _write_markdown(results: List[BenchmarkResult], path: Path,
                    peak_bw_gbs: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dev = torch.cuda.get_device_properties(0)
    lines = [
        "# Softmax benchmark results",
        "",
        f"- Device: {dev.name} (SM {dev.major}.{dev.minor})",
        f"- torch: {torch.__version__} / CUDA {torch.version.cuda}",
        f"- Peak memory bandwidth assumed: {peak_bw_gbs:.0f} GB/s",
        "",
        "Row-wise softmax along the inner axis. Time is median over "
        f"{results[0].iters} CUDA-event iterations after 50 warmup iters.",
        "",
        "| Shape | Dtype | Backend | ms (med) | ms (p95) | GB/s | % peak |",
        "|-------|-------|---------|---------:|---------:|-----:|-------:|",
    ]
    for r in results:
        lines.append(
            f"| {r.shape} | {r.dtype} | `{r.backend}` | "
            f"{r.ms_median:.4f} | {r.ms_p95:.4f} | "
            f"{r.bandwidth_gbs:.1f} | {r.pct_of_peak:.1f}% |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    p = argparse.ArgumentParser(description="GPU softmax benchmark")
    p.add_argument("--shapes", nargs="*",
                   help="List of RxC shapes (default: canonical sweep)")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iters", type=int, default=200,
                   help="Timing samples (one CUDA-event-timed launch each)")
    p.add_argument("--no-flush-l2", action="store_true",
                   help="Disable L2 flush between launches (measures warm-cache "
                        "throughput; small shapes will report > peak DRAM BW)")
    p.add_argument("--peak-bw-gbs", type=float, default=DEFAULT_PEAK_BW_GBS,
                   help=f"Peak DRAM bandwidth (default {DEFAULT_PEAK_BW_GBS} GB/s)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--include-torch", action="store_true",
                   help="Add torch.softmax as a reference column")
    p.add_argument("--csv", type=Path, help="Write CSV to this path")
    p.add_argument("--md", type=Path, help="Write markdown table to this path")
    p.add_argument("--json", type=Path, help="Write raw JSON to this path")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available; benchmark requires a GPU.", file=sys.stderr)
        return 1

    shapes = [Shape.parse(s) for s in args.shapes] if args.shapes else DEFAULT_SHAPES
    backends, dtypes = _make_backends(include_torch=args.include_torch)
    if not backends:
        print("No backends available.", file=sys.stderr)
        return 1

    _print_hardware_banner(args.peak_bw_gbs)

    all_results: List[BenchmarkResult] = []
    for shape in shapes:
        shape_results = _run_shape(
            shape=shape, backends=backends, dtypes=dtypes,
            warmup=args.warmup, iters=args.iters,
            flush_l2=not args.no_flush_l2,
            peak_bw_gbs=args.peak_bw_gbs, seed=args.seed,
        )
        all_results.extend(shape_results)

    _print_table(all_results)

    if args.csv:
        _write_csv(all_results, args.csv)
        print(f"\nwrote {args.csv}")
    if args.md:
        _write_markdown(all_results, args.md, args.peak_bw_gbs)
        print(f"wrote {args.md}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps([asdict(r) for r in all_results], indent=2))
        print(f"wrote {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
