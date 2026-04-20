"""Reference Triton softmax kernel.

Used as an external baseline in the benchmark harness to answer "how close
is the hand-CUDA kernel to a Triton implementation of the same idea?"
Intentionally simple -- the standard tutorial-style per-row softmax -- so it
reflects a realistic baseline rather than a tuned Triton submission.

Skips cleanly when triton is not installed; the harness treats an
ImportError as "baseline unavailable" and omits the column.
"""

from __future__ import annotations

from typing import Optional

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _softmax_triton_kernel(
        x_ptr,
        y_ptr,
        row_stride_x,
        row_stride_y,
        n_cols,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        x_row_ptr = x_ptr + row * row_stride_x
        y_row_ptr = y_ptr + row * row_stride_y

        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        # Load with -inf for masked tails so max/sum reductions stay correct.
        x = tl.load(x_row_ptr + col_offsets, mask=mask, other=-float("inf"))
        row_max = tl.max(x, axis=0)
        numer = tl.exp(x - row_max)
        denom = tl.sum(numer, axis=0)
        y = numer / denom
        tl.store(y_row_ptr + col_offsets, y, mask=mask)


def _next_power_of_two(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def softmax_triton(x: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Row-wise softmax via Triton. Allocates output if not supplied."""
    if not HAS_TRITON:
        raise RuntimeError("Triton is not installed")
    if x.dim() != 2:
        raise ValueError(f"expected 2-D input, got shape {tuple(x.shape)}")
    if not x.is_contiguous():
        x = x.contiguous()

    rows, cols = x.shape
    if out is None:
        out = torch.empty_like(x)

    block_size = _next_power_of_two(cols)
    num_warps = 4
    if block_size >= 2048:
        num_warps = 8
    if block_size >= 4096:
        num_warps = 16
    if block_size >= 8192:
        num_warps = 32

    _softmax_triton_kernel[(rows,)](
        x, out,
        x.stride(0), out.stride(0),
        cols,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return out
