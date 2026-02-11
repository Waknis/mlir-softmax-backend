# MLIR Softmax Pipeline

## Overview
This project lowers a Softmax-style MLIR program to PTX through a staged compiler pipeline.

1. Parse input MLIR.
2. Apply optimization pipeline (`baseline` or `optimized`).
3. Lower to LLVM dialect.
4. Export LLVM IR.
5. Emit PTX with `llc` (`nvptx64-nvidia-cuda`).
6. Optionally load and execute on NVIDIA GPU via CUDA Driver API.

## Optimization in This Project
Custom pass: `mlc-div-to-reciprocal-mul`

- Detects loop-invariant floating-point divisors in `scf.for` loops.
- Hoists reciprocal computation outside loop body.
- Replaces in-loop `arith.divf` with `arith.mulf`.

This reduces dynamic division count from `M*N` to `M` for a two-dimensional softmax-style loop nest.

## Main Commands
Configure and build:

```bash
cmake -S . -B build -G Ninja \
  -DLLVM_DIR=/opt/homebrew/opt/llvm/lib/cmake/llvm \
  -DMLIR_DIR=/opt/homebrew/opt/llvm/lib/cmake/mlir
cmake --build build -j
```

Run tests:

```bash
ctest --test-dir build --output-on-failure
```

Run end-to-end driver:

```bash
./build/tools/mlc-driver/mlc-driver \
  --input examples/softmax.mlir \
  --output-dir build/artifacts \
  --mode optimized
```

Run demo (GPU verify when CUDA/NVIDIA is available):

```bash
./build/tools/mlc-demo/mlc-demo \
  --input examples/softmax.mlir \
  --verify
```

Run benchmark table for 10 shapes:

```bash
./build/bin/softmax-benchmark \
  --shapes=64x64,64x128,128x128,128x256,256x256,256x512,512x512,512x1024,1024x1024,2048x1024
```
