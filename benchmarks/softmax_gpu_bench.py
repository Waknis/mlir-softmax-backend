#!/usr/bin/env python3
"""Compare generated-PTX benchmark output against optional Triton results."""

from __future__ import annotations

import argparse
import csv
import pathlib
import subprocess
import sys
from typing import Any


def _default_mlc_bench() -> str:
    root = pathlib.Path(__file__).resolve().parents[1]
    return str(root / "build" / "tools" / "mlc-gpu-bench" / "mlc-gpu-bench")


def _run_mlc_benchmark(args: argparse.Namespace) -> list[dict[str, Any]]:
    command = [
        args.mlc_bench,
        f"--sizes={args.sizes}",
        f"--warmup={args.warmup}",
        f"--iters={args.iters}",
        f"--sum={args.sum}",
        f"--output-root={args.output_root}",
        "--mode=both",
        "--verify",
        "--csv",
    ]
    if args.llc:
        command.append(f"--llc={args.llc}")

    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    stdout = completed.stdout.strip()
    if stdout.startswith("SKIP:"):
        print(stdout)
        return []
    rows = []
    for row in csv.DictReader(stdout.splitlines()):
        rows.append(
            {
                "backend": row["mode"],
                "size": row["size"],
                "warmup": row["warmup"],
                "iters": row["iters"],
                "avg_kernel_ms": row["avg_kernel_ms"],
                "effective_gib_s": row["effective_gib_s"],
                "max_abs_err": row["max_abs_err"],
                "speedup_vs_baseline": row["speedup_vs_baseline"],
            }
        )
    return rows


def _run_triton_benchmark(args: argparse.Namespace) -> list[dict[str, Any]]:
    from triton_softmax import benchmark_sizes

    sizes = [int(token.strip()) for token in args.sizes.split(",") if token.strip()]
    rows = []
    for result in benchmark_sizes(sizes, args.warmup, args.iters, args.sum):
        rows.append(
            {
                "backend": "triton",
                "size": str(result.size),
                "warmup": str(args.warmup),
                "iters": str(args.iters),
                "avg_kernel_ms": f"{result.avg_kernel_ms:.6f}",
                "effective_gib_s": f"{result.effective_gib_s:.6f}",
                "max_abs_err": f"{result.max_abs_err:.8f}",
                "speedup_vs_baseline": "",
            }
        )
    return rows


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the generated PTX GPU benchmark and optionally compare it with Triton."
    )
    parser.add_argument(
        "--mlc-bench",
        default=_default_mlc_bench(),
        help="Path to the built mlc-gpu-bench binary.",
    )
    parser.add_argument(
        "--output-root",
        default="build/benchmark_runs/compare",
        help="Artifact directory passed through to mlc-gpu-bench.",
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
        "--llc",
        default="",
        help="Optional llc path forwarded to mlc-gpu-bench.",
    )
    parser.add_argument(
        "--skip-triton",
        action="store_true",
        help="Only run the generated PTX benchmark.",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Emit CSV instead of a tab-separated table.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    rows = []
    rows.extend(_run_mlc_benchmark(args))

    if not args.skip_triton:
        try:
            rows.extend(_run_triton_benchmark(args))
        except Exception as exc:  # pragma: no cover - optional dependency path
            print(
                f"Skipping Triton benchmark: {exc}",
                file=sys.stderr,
            )

    if not rows:
        return 0

    fieldnames = [
        "backend",
        "size",
        "warmup",
        "iters",
        "avg_kernel_ms",
        "effective_gib_s",
        "max_abs_err",
        "speedup_vs_baseline",
    ]

    if args.csv:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
        return 0

    print(
        "backend\tsize\twarmup\titers\tavg_kernel_ms\t"
        "effective_gib_s\tmax_abs_err\tspeedup_vs_baseline"
    )
    for row in rows:
        print(
            f"{row['backend']}\t{row['size']}\t{row['warmup']}\t{row['iters']}\t"
            f"{row['avg_kernel_ms']}\t{row['effective_gib_s']}\t"
            f"{row['max_abs_err']}\t{row.get('speedup_vs_baseline', '')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
