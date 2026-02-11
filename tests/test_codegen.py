from __future__ import annotations

from compiler.codegen_cuda import generate_cuda_source
from compiler.ir import InputExpr


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
