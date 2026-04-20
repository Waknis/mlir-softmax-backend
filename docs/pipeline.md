# MLIR Softmax Pipeline

## Overview

This project lowers a softmax-style MLIR program to PTX through a staged compiler pipeline.

1. Parse input MLIR.
2. Apply the selected optimization pipeline (`baseline` or `optimized`).
3. Lower to LLVM dialect.
4. Export LLVM IR.
5. Emit PTX with `llc` for `nvptx64-nvidia-cuda`.
6. Optionally load and execute on an NVIDIA GPU through the CUDA Driver API.

## Optimization

The custom `mlc-div-to-reciprocal-mul` pass:

- Detects loop-invariant floating-point divisors in `scf.for` loops.
- Hoists reciprocal computation outside the loop body.
- Replaces in-loop `arith.divf` with `arith.mulf`.
- Leaves loop-variant denominators unchanged.

For a two-dimensional softmax-style loop nest, this reduces estimated dynamic division count from `M*N` to `M` when the denominator is loop-invariant.

## Main Commands

Configure and build on Linux or WSL2 with LLVM/MLIR 15:

```bash
cmake -S . -B build -G Ninja \
  -DLLVM_DIR=/usr/lib/llvm-15/lib/cmake/llvm \
  -DMLIR_DIR=/usr/lib/llvm-15/lib/cmake/mlir \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Run native tests:

```bash
ctest --test-dir build --output-on-failure
```

Run the local GPU verification gate:

```bash
scripts/verify_wsl_gpu.sh
```

The GPU gate records the GPU, driver, CUDA toolkit, PyTorch CUDA availability, and NVRTC target before running Pytest, CMake, CTest, driver artifact checks, `mlc-demo --verify`, and the benchmark shape smoke test.

Run the end-to-end driver:

```bash
./build/tools/mlc-driver/mlc-driver \
  --input examples/softmax.mlir \
  --output-dir build/artifacts \
  --mode optimized
```

Run the GPU demo:

```bash
./build/tools/mlc-demo/mlc-demo \
  --input examples/softmax.mlir \
  --mode optimized \
  --verify
```

Run the pass static-analysis table for 10 shapes (compile-time, not runtime):

```bash
./build/bin/mlc-pass-analysis \
  --shapes=64x64,64x128,128x128,128x256,256x256,256x512,512x512,512x1024,1024x1024,2048x1024
```

Run the GPU runtime benchmark across four backends:

```bash
python -m benchmarks.softmax_gpu_bench --shapes 1024x4096 4096x4096 8192x8192
```
