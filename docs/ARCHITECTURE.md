# Architecture

For the current end-to-end flow, see [pipeline.md](pipeline.md).

## Current Project Overview

This repository contains a C++ MLIR compiler backend with the following components.

### Compiler (`lib/Passes/`, `compiler/pipeline/`)

- Custom MLIR pass (`mlc-div-to-reciprocal-mul`): LICM plus division strength reduction for `scf.for` loops.
- Lowering pipeline (`LoweringPipeline.cpp`): five-stage pipeline from MLIR input through LLVM dialect and LLVM IR to NVPTX PTX.
- Kernel wrapper injection for CUDA kernel launch compatibility.

### Runtime (`runtime/`)

- CUDA Driver API wrapper (`CudaRuntime.cpp`): dynamic `libcuda.so` loading through `dlopen`/`dlsym`, PTX JIT module loading, device memory management, and kernel launch.
- Native Linux and WSL2 support when the NVIDIA driver is available.

### Tools (`tools/`)

- `mlc-opt`: standalone pass runner for FileCheck testing.
- `mlc-driver`: end-to-end pipeline driver from MLIR to PTX artifacts.
- `mlc-demo`: GPU demo with numerical verification.
- `mlc-bench`: pass microbenchmark for execution time and division counts.

### Testing (`test/`, `tests/`, `scripts/`)

- FileCheck unit tests for pass correctness.
- Shell-based end-to-end and benchmark contract tests.
- CTest coverage for the native pipeline.
- Pytest coverage for Python helper/runtime components.
- Local GPU verification through `scripts/verify_wsl_gpu.sh`.
