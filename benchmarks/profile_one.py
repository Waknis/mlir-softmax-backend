"""Minimal single-kernel harness for ncu/nsys profiling.

Runs ONE launch of one softmax backend at a fixed shape. This is the target
you profile with `ncu --set full python -m benchmarks.profile_one ...`
(keeping the capture short and unambiguous).

Usage:
    python -m benchmarks.profile_one --backend online_f32 --shape 4096x4096
    python -m benchmarks.profile_one --backend triton_f32 --shape 4096x4096 --iters 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from kernels.softmax_loader import get_library  # noqa: E402
from benchmarks.triton_softmax import HAS_TRITON, softmax_triton  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="online_f32",
                   choices=["online_f32", "naive_f32", "online_f16", "triton_f32"])
    p.add_argument("--shape", default="4096x4096")
    p.add_argument("--iters", type=int, default=1,
                   help="Number of timed launches (besides warmup)")
    p.add_argument("--warmup", type=int, default=3)
    args = p.parse_args()

    rows, cols = map(int, args.shape.lower().split("x"))
    dtype = torch.float16 if args.backend.endswith("f16") else torch.float32
    x = torch.randn(rows, cols, device="cuda", dtype=dtype)
    y = torch.empty_like(x)

    if args.backend == "triton_f32":
        if not HAS_TRITON:
            print("Triton unavailable", file=sys.stderr)
            return 1
        def fn():
            softmax_triton(x, y)
    else:
        lib = get_library()
        kernel = getattr(lib, args.backend)
        def fn():
            kernel(x, y)

    for _ in range(args.warmup):
        fn()
    torch.cuda.synchronize()

    # The iters here are what ncu/nsys will capture. Keep small for ncu.
    for _ in range(args.iters):
        fn()
    torch.cuda.synchronize()
    return 0


if __name__ == "__main__":
    sys.exit(main())
