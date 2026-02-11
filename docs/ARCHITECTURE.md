# Architecture (MVP)

## Goals
- Fuse consecutive elementwise ops from a PyTorch FX graph.
- Generate a specialized CUDA kernel for concrete shapes.
- Compile at runtime with NVRTC and cache by graph+shape+dtype+device.
- Keep a safe eager fallback for unsupported graphs or runtime failures.

## High-level split
- `compiler/`: graph parsing + IR + CUDA code generation.
- `runtime/`: NVRTC/PTX handling + CUDA driver launch + cache.

## Compiler pipeline

### 1) FX parse (`compiler/fx_parser.py`)
- Uses `torch.fx.symbolic_trace`.
- Accepts only supported elementwise operations and a single tensor output.
- Converts graph into expression IR:
  - `InputExpr(index)`
  - `ConstExpr(value)`
  - `UnaryExpr(op, x)`
  - `BinaryExpr(op, lhs, rhs)`
- Records input sources:
  - positional forward args (`placeholder`)
  - module attrs (`get_attr`, e.g. parameters/buffers)
- Produces stable `graph_hash` from IR structure + input source order.

### 2) Shape specialization (`compiler/compiler.py`)
- Runtime call gathers tensor inputs from args/attrs.
- Computes output broadcast shape from input shapes.
- Computes per-input broadcast strides aligned to output rank.
- Builds cache key from `(graph_hash, shapes, dtype, device)`.

### 3) CUDA codegen (`compiler/codegen_cuda.py`)
- Emits one fused kernel per specialized graph/shape set.
- Kernel characteristics:
  - grid-stride scalar loop over `numel`,
  - optional vectorized path:
    - `float4` for fp32,
    - `half2` for fp16,
    - only for non-broadcast aligned contiguous case.
- Broadcast mode emits `in_offset_i(linear_idx)` helpers that map output index to each input offset using precomputed broadcast-aware strides.

## Runtime pipeline

### 1) Compile (`runtime/nvrtc_driver.py`)
- Loads NVRTC and CUDA Driver APIs via `ctypes`.
- Compiles generated CUDA source to PTX with:
  - `--gpu-architecture=compute_XY` from active torch device capability,
  - fast-math options for MVP throughput.

### 2) Cache (`runtime/cache.py`)
- Memory cache + disk cache (`~/.cache/mini_ml_compiler/<key>.ptx`).
- If key hit: skip NVRTC compile and load cached PTX.

### 3) Launch (`runtime/launcher.py`)
- Ensures CUDA context is available (compatible with torch CUDA runtime context).
- Loads module/function from PTX via CUDA Driver API.
- Launches on torch current stream with kernel args:
  - input pointers,
  - output pointer,
  - `numel`.

## Correctness and fallback strategy
- If parsing fails, unsupported graph detected, dtype/device unsupported, or launch fails:
  - compilation path disables itself,
  - execution falls back to eager module forward.
- CUDA tests compare fused output with eager output for:
  - random no-broadcast shapes,
  - broadcast edge cases,
  - fp32/fp16 tolerances.

## Non-goals in MVP
- Reduction fusion.
- Multi-output graphs.
- Autograd integration for backward graph fusion.
