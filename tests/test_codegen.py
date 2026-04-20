from __future__ import annotations

import pytest

from experiments.fx_nvrtc.codegen_cuda import generate_cuda_source
from experiments.fx_nvrtc.ir import BinaryExpr, InputExpr


def test_codegen_placeholder_contains_kernel() -> None:
    spec = generate_cuda_source(
        expr=InputExpr(0),
        input_count=1,
        dtype="float32",
        graph_hash="deadbeef",
        output_shape=(8, 16),
        input_broadcast_strides=((16, 1),),
    )
    assert "__global__" in spec.source
    assert spec.kernel_name


def test_codegen_vectorizes_contiguous_float32() -> None:
    spec = generate_cuda_source(
        expr=BinaryExpr(op="add", lhs=InputExpr(0), rhs=InputExpr(1)),
        input_count=2,
        dtype="float32",
        graph_hash="feedface",
        output_shape=(4, 8),
        input_broadcast_strides=((8, 1), (8, 1)),
    )

    assert spec.vector_width == 4
    assert not spec.has_broadcast
    assert "float4" in spec.source


def test_codegen_vectorizes_contiguous_float16() -> None:
    spec = generate_cuda_source(
        expr=InputExpr(0),
        input_count=1,
        dtype="float16",
        graph_hash="badc0ffee",
        output_shape=(16,),
        input_broadcast_strides=((1,),),
    )

    assert spec.vector_width == 2
    assert not spec.has_broadcast
    assert "half2" in spec.source


def test_codegen_rejects_invalid_contracts() -> None:
    with pytest.raises(ValueError, match="Unsupported dtype"):
        generate_cuda_source(
            expr=InputExpr(0),
            input_count=1,
            dtype="float64",
            graph_hash="deadbeef",
            output_shape=(8,),
            input_broadcast_strides=((1,),),
        )

    with pytest.raises(ValueError, match="length must match"):
        generate_cuda_source(
            expr=InputExpr(0),
            input_count=2,
            dtype="float32",
            graph_hash="deadbeef",
            output_shape=(8,),
            input_broadcast_strides=((1,),),
        )

    with pytest.raises(ValueError, match="stride vector"):
        generate_cuda_source(
            expr=InputExpr(0),
            input_count=1,
            dtype="float32",
            graph_hash="deadbeef",
            output_shape=(2, 4),
            input_broadcast_strides=((1,),),
        )
