# MLIR Softmax Backend

[![CI](https://github.com/Waknis/mlir-softmax-backend/actions/workflows/ci.yml/badge.svg)](https://github.com/Waknis/mlir-softmax-backend/actions/workflows/ci.yml)

A compact MLIR/LLVM backend that lowers a softmax-style MLIR program to PTX and verifies execution through the CUDA Driver API.

```text
MLIR input -> custom optimization pass -> LLVM dialect -> LLVM IR -> PTX -> CUDA Driver launch
```

## What This Demonstrates

- MLIR parsing, pass execution, LLVM dialect lowering, LLVM IR export, NVPTX codegen, and CUDA Driver execution in C++17.
- A custom MLIR pass, `mlc-div-to-reciprocal-mul`, that hoists loop-invariant floating-point division into a reciprocal multiply.
- Runtime loading of `libcuda.so`, PTX JIT module loading, device memory management, kernel launch, and host-side numerical verification.
- FileCheck, CTest, Python unit tests, CI, and a local GPU verification gate.

## Optimization Pass

The `mlc-div-to-reciprocal-mul` pass detects floating-point divisions inside `scf.for` loops where the denominator is defined outside the loop. It hoists one reciprocal and replaces each in-loop divide with a multiply.

Before:

```mlir
scf.for %i = %c0 to %c1024 step %c1 {
  %x = memref.load %input[%i] : memref<1024xf32>
  %y = arith.divf %x, %sum : f32
  memref.store %y, %output[%i] : memref<1024xf32>
}
```

After:

```mlir
%recip = arith.divf %one, %sum : f32
scf.for %i = %c0 to %c1024 step %c1 {
  %x = memref.load %input[%i] : memref<1024xf32>
  %y = arith.mulf %x, %recip : f32
  memref.store %y, %output[%i] : memref<1024xf32>
}
```

Loop-variant denominators are intentionally left unchanged and covered by FileCheck tests.

## Build

Linux or WSL2 with LLVM/MLIR 15:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake ninja-build \
  llvm-15 llvm-15-dev llvm-15-tools \
  libmlir-15-dev mlir-15-tools

cmake -S . -B build -G Ninja \
  -DLLVM_DIR=/usr/lib/llvm-15/lib/cmake/llvm \
  -DMLIR_DIR=/usr/lib/llvm-15/lib/cmake/mlir \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

macOS with Homebrew LLVM:

```bash
cmake -S . -B build -G Ninja \
  -DLLVM_DIR=/opt/homebrew/opt/llvm/lib/cmake/llvm \
  -DMLIR_DIR=/opt/homebrew/opt/llvm/lib/cmake/mlir
cmake --build build -j
```

## Verification

CPU/Python tests:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

Native MLIR/LLVM tests:

```bash
ctest --test-dir build --output-on-failure
```

Local GPU verification gate:

```bash
scripts/verify_wsl_gpu.sh
```

The GPU gate records the NVIDIA device, driver, CUDA toolkit, PyTorch CUDA availability, and NVRTC target. It then runs Python CUDA correctness tests, CMake, CTest, explicit driver artifact checks, `mlc-demo --verify`, and the benchmark shape smoke test.

Validated local environment:

```text
WSL2 Ubuntu 24.04
NVIDIA GeForce RTX 5060 Ti, compute capability 12.0
NVIDIA driver 595.97
CUDA toolkit 13.2
```

CI runs the CPU/Python and native MLIR/LLVM lanes on Ubuntu 24.04. CUDA-dependent tests are skip-capable in CI because GitHub-hosted runners do not provide this local GPU.

## End-to-End Driver

```bash
./build/tools/mlc-driver/mlc-driver \
  --input examples/softmax.mlir \
  --output-dir build/artifacts \
  --mode optimized
```

Expected artifacts:

- `stage0_input.mlir`
- `stage1_optimized.mlir`
- `stage2_llvm_dialect.mlir`
- `stage3_llvm_ir.ll`
- `stage4_kernel.ptx`

## GPU Demo

```bash
./build/tools/mlc-demo/mlc-demo \
  --input examples/softmax.mlir \
  --mode optimized \
  --verify
```

On systems without an NVIDIA GPU or CUDA driver, the demo exits with a `SKIP` message. The local GPU verification script treats an unexpected skip as a failure.

## Benchmark

```bash
./build/bin/softmax-benchmark \
  --shapes=64x64,64x128,128x128,128x256,256x256,256x512,512x512,512x1024,1024x1024,2048x1024
```

The benchmark reports actual `arith.divf` counts before and after the optimization pass, an estimated dynamic division reduction, and average pass time. Timings are environment-specific.

## Repository Map

- `lib/Passes/`: custom MLIR optimization pass.
- `compiler/pipeline/`: staged MLIR-to-PTX lowering pipeline.
- `runtime/`: CUDA Driver runtime wrapper.
- `tools/mlc-opt`: pass runner.
- `tools/mlc-driver`: end-to-end artifact generator.
- `tools/mlc-demo`: GPU launch and numerical verification.
- `benchmarks/`: benchmark harness.
- `test/` and `tests/`: FileCheck, CTest, shell, and Python tests.
- `scripts/verify_wsl_gpu.sh`: local GPU-backed verification gate.
