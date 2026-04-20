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
- `mlc-pass-analysis` (in `benchmarks/`): pass microbenchmark for
  compile-time pipeline wall time and dynamic division counts.

### Hand kernels + Python harness (`kernels/`, `benchmarks/`)

- `kernels/softmax_online.cu`: hand-written online softmax CUDA kernels
  (online, naive 3-pass, f16) with warp-shuffle reduction and templated
  block-size dispatch. Built into a `.so` loaded from Python via ctypes.
- `benchmarks/softmax_gpu_bench.py`: CUDA-event-timed benchmark harness
  that sweeps shapes / dtypes across the hand kernels, Triton baseline,
  and (optionally) PyTorch's cuDNN-backed softmax, with an L2 flush
  between launches to keep bandwidth numbers DRAM-bound.
- `benchmarks/triton_softmax.py`: tutorial-style per-row Triton kernel.
- `docs/profiling/`: committed Nsight Compute / Nsight Systems reports
  and the roofline plot, all regeneratable.

### Testing (`test/`, `tests/`, `scripts/`)

- FileCheck unit tests for pass correctness on 1-D and nested 2-D loops.
- Shell-based end-to-end and pass-analysis contract tests.
- CTest coverage for the native pipeline and CUDA correctness test.
- Pytest coverage for Python helper/runtime components and the
  hand-kernel correctness + numerical-stability suite.
- Local GPU verification through `scripts/verify_wsl_gpu.sh`.
