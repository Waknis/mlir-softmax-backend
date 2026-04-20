"""Correctness + numerical-stability tests for the hand-written CUDA softmax.

Covers:
- Correctness vs torch.softmax across a sweep of shapes (power-of-2 and not)
- Numerical stability (inputs with large magnitude that would overflow a naive
  exp-then-sum implementation)
- Edge shapes: single row, single column, non-power-of-2 widths that cross
  block-size dispatch boundaries
- Agreement between the online and naive kernels
- f16 accumulation accuracy

The kernels read tensors on the current CUDA stream, so all inputs stay on
GPU for these tests.
"""

from __future__ import annotations

import pytest
import torch

from kernels.softmax_loader import SoftmaxKernelUnavailable, get_library


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA GPU required"
)


@pytest.fixture(scope="module")
def lib():
    try:
        return get_library()
    except SoftmaxKernelUnavailable as e:
        pytest.skip(f"softmax kernels unavailable: {e}")


def _reference(x: torch.Tensor) -> torch.Tensor:
    # f32 reference regardless of input dtype -- our f16 kernels accumulate in
    # f32 internally, so comparing against f32 torch.softmax is the fair check.
    return torch.softmax(x.float(), dim=-1)


def _run(lib, name: str, x: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x)
    getattr(lib, name)(x, y)
    return y


# Shapes chosen to hit every dispatched block-size class (32/64/128/256/512)
# and to cross the boundaries at 64, 128, 256, 1024 columns.
SHAPES_F32 = [
    (1, 1),
    (1, 64),
    (4, 64),
    (8, 65),          # just past 32-block boundary, non-power-of-2
    (16, 127),
    (8, 128),
    (8, 129),         # crosses 128-col boundary
    (32, 256),
    (32, 257),        # crosses 256-col boundary
    (64, 1024),
    (64, 1025),       # crosses 1024-col boundary, dispatches 512-block
    (128, 2048),
    (256, 4096),
    (1, 8193),        # single row, odd width
]


@pytest.mark.parametrize("rows,cols", SHAPES_F32)
def test_online_f32_matches_torch(lib, rows, cols):
    torch.manual_seed(0)
    x = torch.randn(rows, cols, device="cuda", dtype=torch.float32)
    y = _run(lib, "online_f32", x)
    ref = _reference(x)
    assert torch.allclose(y, ref, rtol=1e-5, atol=1e-5), (
        f"max diff {(y - ref).abs().max().item():.3e}"
    )


@pytest.mark.parametrize("rows,cols", SHAPES_F32)
def test_naive_f32_matches_torch(lib, rows, cols):
    torch.manual_seed(0)
    x = torch.randn(rows, cols, device="cuda", dtype=torch.float32)
    y = _run(lib, "naive_f32", x)
    ref = _reference(x)
    assert torch.allclose(y, ref, rtol=1e-5, atol=1e-5), (
        f"max diff {(y - ref).abs().max().item():.3e}"
    )


@pytest.mark.parametrize("rows,cols", SHAPES_F32)
def test_online_and_naive_agree(lib, rows, cols):
    # The two implementations use different reductions; they should produce
    # bitwise-close outputs even if each drifts slightly from the torch ref.
    torch.manual_seed(1)
    x = torch.randn(rows, cols, device="cuda", dtype=torch.float32)
    y_online = _run(lib, "online_f32", x)
    y_naive = _run(lib, "naive_f32", x)
    assert torch.allclose(y_online, y_naive, rtol=1e-5, atol=1e-5)


SHAPES_F16 = [
    (4, 64),
    (8, 128),
    (32, 256),
    (64, 1024),
    (128, 2048),
    (256, 4096),
]


@pytest.mark.parametrize("rows,cols", SHAPES_F16)
def test_online_f16_matches_torch(lib, rows, cols):
    torch.manual_seed(0)
    x = torch.randn(rows, cols, device="cuda", dtype=torch.float16)
    y = _run(lib, "online_f16", x)
    ref = _reference(x)
    # f16 storage tolerance: 2e-2 is conservative vs torch's own f16 softmax;
    # our kernel accumulates in f32 so the only f16 error is the output round.
    assert torch.allclose(y.float(), ref, rtol=2e-2, atol=2e-2), (
        f"max diff {(y.float() - ref).abs().max().item():.3e}"
    )


@pytest.mark.parametrize("scale", [10.0, 50.0, 100.0, 500.0])
def test_stability_large_exponents_f32(lib, scale):
    # Without the max-subtraction trick, exp(500) would overflow; a correct
    # softmax must still return a valid probability distribution.
    torch.manual_seed(0)
    x = scale * torch.randn(16, 512, device="cuda", dtype=torch.float32)
    y = _run(lib, "online_f32", x)
    row_sums = y.sum(dim=-1)
    assert torch.isfinite(y).all(), f"NaN/Inf at scale={scale}"
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4), (
        f"row sums deviate: max|sum-1|={(row_sums - 1).abs().max().item():.3e}"
    )


@pytest.mark.parametrize("scale", [10.0, 50.0])
def test_stability_large_exponents_f16(lib, scale):
    # f16 has a much narrower dynamic range (~6e4); large values are more
    # stressful. Our kernel converts to f32 for the reduction, so this should
    # still work, but with looser tolerance on the probability sums.
    torch.manual_seed(0)
    x = scale * torch.randn(16, 512, device="cuda", dtype=torch.float16)
    y = _run(lib, "online_f16", x)
    assert torch.isfinite(y).all(), f"NaN/Inf at scale={scale}"
    row_sums = y.float().sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=2e-2)


def test_single_row(lib):
    x = torch.randn(1, 4096, device="cuda", dtype=torch.float32)
    y = _run(lib, "online_f32", x)
    assert torch.allclose(y, _reference(x), rtol=1e-5, atol=1e-5)


def test_single_column_degenerate(lib):
    # cols=1 means softmax is trivially the constant 1; still must not crash.
    x = torch.randn(32, 1, device="cuda", dtype=torch.float32)
    y = _run(lib, "online_f32", x)
    assert torch.allclose(y, torch.ones_like(y))


def test_constant_row(lib):
    # All equal elements -> uniform 1/N probabilities. Exercises the case
    # where the max-subtraction leaves every exp at exactly 1.
    x = torch.full((4, 128), 3.14, device="cuda", dtype=torch.float32)
    y = _run(lib, "online_f32", x)
    expected = torch.full_like(y, 1.0 / 128.0)
    assert torch.allclose(y, expected, atol=1e-6)


def test_negative_only_row(lib):
    x = -1.0 - torch.rand(8, 256, device="cuda", dtype=torch.float32)
    y = _run(lib, "online_f32", x)
    assert torch.allclose(y, _reference(x), rtol=1e-5, atol=1e-5)


def test_empty_input_shapes_rejected(lib):
    # rows=0 or cols=0 should be a no-op (the kernel short-circuits), but the
    # validation in the loader still demands 2-D contiguous tensors.
    x = torch.empty(0, 128, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    lib.online_f32(x, y)  # does nothing, but must not raise.


def test_non_contiguous_rejected(lib):
    x = torch.randn(32, 64, device="cuda", dtype=torch.float32).t()
    y = torch.empty_like(x)
    with pytest.raises(ValueError, match="contiguous"):
        lib.online_f32(x, y)


def test_dtype_mismatch_rejected(lib):
    x = torch.randn(4, 64, device="cuda", dtype=torch.float16)
    y = torch.empty_like(x)
    with pytest.raises(ValueError, match="dtype"):
        lib.online_f32(x, y)
