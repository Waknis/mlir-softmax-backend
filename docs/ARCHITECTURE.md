# Architecture

For the current end-to-end compiler flow, see [pipeline.md](pipeline.md).

## Project Shape

This repository is a C++ MLIR backend. It parses softmax-shaped MLIR,
applies a custom optimization pass, lowers to LLVM dialect, exports LLVM
IR, emits NVPTX PTX, and optionally executes the generated kernel with the
CUDA Driver API.

## Core Components

- `lib/Passes/`: custom `mlc-div-to-reciprocal-mul` pass. It hoists
  loop-invariant floating-point division into one reciprocal outside the
  loop and replaces in-loop divides with multiplies.
- `compiler/pipeline/`: staged MLIR-to-PTX lowering. It parses MLIR,
  applies the selected optimization mode, lowers to LLVM dialect, exports
  LLVM IR, injects the CUDA kernel wrapper, and emits PTX with `llc`.
- `runtime/`: C++ CUDA Driver API wrapper. It dynamically loads
  `libcuda.so`, loads generated PTX, manages device buffers, launches the
  generated kernel, and copies results back for verification.
- `tools/`: native CLIs. `mlc-opt` runs the pass, `mlc-driver` emits all
  compiler artifacts, `mlc-demo` launches generated PTX, and
  `mlc-pass-analysis` reports compile-time pass effects.

## Current Scope

The MLIR examples model the normalization stage of row-wise softmax:

```text
y[i, j] = x[i, j] / sum[i]
```

Full safe softmax lowering with max subtraction, `exp`, reduction, and
GPU-parallel row reductions is future work.

## Testing

- `test/`: FileCheck and CTest coverage for native MLIR behavior.
- `tests/`: shell and C++ integration checks driven by CTest.
- `scripts/verify_wsl_gpu.sh`: local GPU verification gate covering CMake,
  CTest, driver artifacts, `mlc-demo --verify`, and the pass-analysis shape
  smoke test.
