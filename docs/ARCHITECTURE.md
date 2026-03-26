# Architecture

> **Note:** This document previously described a PyTorch FX graph fusion pipeline
> with NVRTC compilation — a separate prototype that predates the current project.
> That design is no longer relevant to this codebase.

For the current architecture, see [pipeline.md](pipeline.md).

## Current Project Overview

This is a **C++ MLIR compiler backend** with the following components:

### Compiler (`lib/Passes/`, `compiler/pipeline/`)
- **Custom MLIR pass** (`mlc-div-to-reciprocal-mul`): LICM + division strength reduction for `scf.for` loops.
- **Lowering pipeline** (`LoweringPipeline.cpp`): 5-stage pipeline from MLIR input through LLVM dialect, LLVM IR, to NVPTX PTX.
- Kernel wrapper injection for CUDA kernel launch compatibility.

### Runtime (`runtime/`)
- **CUDA Driver API wrapper** (`CudaRuntime.cpp`): Dynamic `libcuda.so` loading via `dlopen`/`dlsym`, PTX JIT compilation, device memory management, and kernel launch.
- Works on native Linux and WSL2 (CUDA driver is forwarded from the Windows host).

### Tools (`tools/`)
- `mlc-opt`: Standalone pass runner for FileCheck testing.
- `mlc-driver`: End-to-end pipeline driver (MLIR -> PTX).
- `mlc-demo`: GPU demo with numerical verification.
- `mlc-bench`: Pass microbenchmark (measures pass execution time and division counts).

### Testing (`test/`, `tests/`)
- FileCheck unit tests for pass correctness.
- Shell-based end-to-end and benchmark contract tests.
- GPU correctness test harness.
