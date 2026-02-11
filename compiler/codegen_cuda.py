"""CUDA source generation for fused elementwise kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .ir import Expr, emit_c_float_expr


@dataclass(frozen=True)
class KernelSpec:
    kernel_name: str
    source: str
    vector_width: int
    has_broadcast: bool


def _emit_param_list(input_count: int, scalar_t: str) -> str:
    parts = [f"const {scalar_t}* __restrict__ in{i}" for i in range(input_count)]
    parts.append(f"{scalar_t}* __restrict__ out")
    parts.append("long long numel")
    return ", ".join(parts)


def _emit_broadcast_index_fn(
    input_idx: int,
    output_shape: Sequence[int],
    input_broadcast_strides: Sequence[int],
) -> str:
    rank = len(output_shape)
    if rank == 0:
        return f"""
__device__ __forceinline__ long long in_offset_{input_idx}(long long linear_idx) {{
  (void)linear_idx;
  return 0;
}}
"""

    out_strides: list[int] = []
    running = 1
    for dim in reversed(output_shape):
        out_strides.append(running)
        running *= max(int(dim), 1)
    out_strides = list(reversed(out_strides))

    lines = [f"__device__ __forceinline__ long long in_offset_{input_idx}(long long linear_idx) {{"] 
    lines.append("  long long rem = linear_idx;")
    lines.append("  long long off = 0;")
    for dim in range(rank):
        os = int(out_strides[dim])
        shape = int(output_shape[dim])
        stride = int(input_broadcast_strides[dim])
        if dim == rank - 1:
            lines.append("  long long coord = rem;")
        else:
            lines.append(f"  long long coord = rem / {os}ll;")
            lines.append(f"  rem -= coord * {os}ll;")
        if stride != 0:
            lines.append(f"  off += coord * {stride}ll;")
        if shape == 1:
            lines.append("  coord = 0;")
    lines.append("  return off;")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _emit_scalar_loop(
    expr: Expr,
    input_count: int,
    scalar_t: str,
    has_broadcast: bool,
) -> str:
    lines: list[str] = []
    lines.append("  const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;")
    lines.append("  const long long stride = (long long)blockDim.x * gridDim.x;")
    lines.append("  for (long long idx = tid; idx < numel; idx += stride) {")
    for i in range(input_count):
        if has_broadcast:
            lines.append(f"    const long long in_idx_{i} = in_offset_{i}(idx);")
            lines.append(f"    const float v{i} = to_float<scalar_t>(in{i}[in_idx_{i}]);")
        else:
            lines.append(f"    const float v{i} = to_float<scalar_t>(in{i}[idx]);")
    c_expr = emit_c_float_expr(expr, input_name=lambda j: f"v{j}")
    lines.append(f"    const float out_f = {c_expr};")
    lines.append("    out[idx] = from_float<scalar_t>(out_f);")
    lines.append("  }")
    return "\n".join(lines)


def _emit_fp32_vec_loop(expr: Expr, input_count: int) -> str:
    lines: list[str] = []
    align_checks = " && ".join(
        [f"((reinterpret_cast<unsigned long long>(in{i}) & 0xF) == 0)" for i in range(input_count)]
        + ["((reinterpret_cast<unsigned long long>(out) & 0xF) == 0)"]
    )
    lines.append(f"  if ({align_checks} && (numel % 4ll == 0ll)) {{")
    lines.append("    const long long vec_n = numel / 4ll;")
    lines.append("    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;")
    lines.append("    const long long stride = (long long)blockDim.x * gridDim.x;")
    for i in range(input_count):
        lines.append(f"    const float4* __restrict__ in{i}_v = reinterpret_cast<const float4*>(in{i});")
    lines.append("    float4* __restrict__ out_v = reinterpret_cast<float4*>(out);")
    lines.append("    for (long long vi = tid; vi < vec_n; vi += stride) {")
    for i in range(input_count):
        lines.append(f"      const float4 vin{i} = in{i}_v[vi];")
    for lane in ("x", "y", "z", "w"):
        for i in range(input_count):
            lines.append(f"      const float v{i}_{lane} = vin{i}.{lane};")
        lane_expr = emit_c_float_expr(expr, input_name=lambda j, l=lane: f"v{j}_{l}")
        lines.append(f"      const float out_{lane} = {lane_expr};")
    lines.append("      float4 o;")
    lines.append("      o.x = out_x;")
    lines.append("      o.y = out_y;")
    lines.append("      o.z = out_z;")
    lines.append("      o.w = out_w;")
    lines.append("      out_v[vi] = o;")
    lines.append("    }")
    lines.append("    return;")
    lines.append("  }")
    return "\n".join(lines)


def _emit_fp16_vec_loop(expr: Expr, input_count: int) -> str:
    lines: list[str] = []
    align_checks = " && ".join(
        [f"((reinterpret_cast<unsigned long long>(in{i}) & 0x3) == 0)" for i in range(input_count)]
        + ["((reinterpret_cast<unsigned long long>(out) & 0x3) == 0)"]
    )
    lines.append(f"  if ({align_checks} && (numel % 2ll == 0ll)) {{")
    lines.append("    const long long vec_n = numel / 2ll;")
    lines.append("    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;")
    lines.append("    const long long stride = (long long)blockDim.x * gridDim.x;")
    for i in range(input_count):
        lines.append(f"    const half2* __restrict__ in{i}_v = reinterpret_cast<const half2*>(in{i});")
    lines.append("    half2* __restrict__ out_v = reinterpret_cast<half2*>(out);")
    lines.append("    for (long long vi = tid; vi < vec_n; vi += stride) {")
    for i in range(input_count):
        lines.append(f"      const half2 vin{i} = in{i}_v[vi];")
        lines.append(f"      const float v{i}_lo = __half2float(__low2half(vin{i}));")
        lines.append(f"      const float v{i}_hi = __half2float(__high2half(vin{i}));")
    lo_expr = emit_c_float_expr(expr, input_name=lambda j: f"v{j}_lo")
    hi_expr = emit_c_float_expr(expr, input_name=lambda j: f"v{j}_hi")
    lines.append(f"      const float out_lo = {lo_expr};")
    lines.append(f"      const float out_hi = {hi_expr};")
    lines.append("      out_v[vi] = __halves2half2(__float2half(out_lo), __float2half(out_hi));")
    lines.append("    }")
    lines.append("    return;")
    lines.append("  }")
    return "\n".join(lines)


def generate_cuda_source(
    expr: Expr,
    input_count: int,
    dtype: str,
    graph_hash: str,
    output_shape: Sequence[int],
    input_broadcast_strides: Sequence[Sequence[int]],
) -> KernelSpec:
    """Generate CUDA source for a fused chain.

    `input_broadcast_strides` must be aligned to `output_shape` rank.
    """
    if dtype not in {"float32", "float16"}:
        raise ValueError(f"Unsupported dtype for codegen: {dtype}")
    if len(input_broadcast_strides) != input_count:
        raise ValueError("input_broadcast_strides length must match input_count")

    scalar_t = "float" if dtype == "float32" else "half"
    vector_width = 4 if dtype == "float32" else 2

    out_shape_tuple = tuple(int(x) for x in output_shape)
    rank = len(out_shape_tuple)
    has_broadcast = False
    for strides in input_broadcast_strides:
        if len(strides) != rank:
            raise ValueError("Each input stride vector must match output rank")
        # Contiguous output strides for no-broadcast match check.
        running = 1
        out_contig: list[int] = []
        for dim in reversed(out_shape_tuple):
            out_contig.append(running)
            running *= max(dim, 1)
        out_contig = list(reversed(out_contig))
        if tuple(int(s) for s in strides) != tuple(out_contig):
            has_broadcast = True
            break

    kernel_name = f"fused_kernel_{graph_hash[:16]}"
    param_list = _emit_param_list(input_count, scalar_t=scalar_t)

    broadcast_index_fns = ""
    if has_broadcast:
        broadcast_index_fns = "\n".join(
            _emit_broadcast_index_fn(i, out_shape_tuple, input_broadcast_strides[i])
            for i in range(input_count)
        )

    vec_block = ""
    if not has_broadcast:
        if dtype == "float32":
            vec_block = _emit_fp32_vec_loop(expr, input_count=input_count)
        else:
            vec_block = _emit_fp16_vec_loop(expr, input_count=input_count)

    scalar_block = _emit_scalar_loop(
        expr=expr,
        input_count=input_count,
        scalar_t=scalar_t,
        has_broadcast=has_broadcast,
    )

    source = f"""
#include <cuda_fp16.h>
#include <math.h>
#include <stdint.h>

using scalar_t = {scalar_t};

template <typename T>
__device__ __forceinline__ float to_float(T v) {{
  return static_cast<float>(v);
}}

template <>
__device__ __forceinline__ float to_float<half>(half v) {{
  return __half2float(v);
}}

template <typename T>
__device__ __forceinline__ T from_float(float v) {{
  return static_cast<T>(v);
}}

template <>
__device__ __forceinline__ half from_float<half>(float v) {{
  return __float2half(v);
}}

{broadcast_index_fns}
extern "C" __global__ void {kernel_name}({param_list}) {{
{vec_block}
{scalar_block}
}}
""".strip() + "\n"

    return KernelSpec(
        kernel_name=kernel_name,
        source=source,
        vector_width=vector_width if not has_broadcast else 1,
        has_broadcast=has_broadcast,
    )
