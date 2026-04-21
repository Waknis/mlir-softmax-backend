# MLIR Softmax Backend

[![CI](https://github.com/Waknis/mlir-softmax-backend/actions/workflows/ci.yml/badge.svg)](https://github.com/Waknis/mlir-softmax-backend/actions/workflows/ci.yml)

A compact C++17 MLIR/LLVM backend that lowers a softmax-shaped MLIR program
to NVPTX PTX and can execute the generated kernel through the CUDA Driver
API.

The current pipeline focuses on the divide-and-normalize stage of row-wise
softmax, which is the stage exercised by the custom optimization pass:
`mlc-div-to-reciprocal-mul`.

```text
MLIR input -> custom pass -> LLVM dialect -> LLVM IR -> PTX -> CUDA Driver launch
```

## What This Demonstrates

- MLIR parsing, pass execution, LLVM dialect lowering, LLVM IR export, and
  NVPTX codegen.
- A custom MLIR pass, `mlc-div-to-reciprocal-mul`, that hoists
  loop-invariant floating-point division into a reciprocal multiply.
- End-to-end artifact generation through `mlc-driver`.
- CUDA Driver API loading of generated PTX and numerical verification
  through `mlc-demo`.
- A generated-PTX GPU runtime benchmark through `mlc-gpu-bench`.
- FileCheck tests, CTest integration, CI, and a local WSL/NVIDIA GPU
  verification gate.

## Optimization Pass

The `mlc-div-to-reciprocal-mul` pass detects floating-point divisions
inside `scf.for` loops where the denominator is defined outside the loop.
It hoists one reciprocal and replaces each in-loop divide with a multiply.

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

The pass also handles nested loops, as shown in
[`examples/softmax_rowwise.mlir`](examples/softmax_rowwise.mlir), reducing
estimated dynamic division count from `M*N` to `M` when the denominator is
loop-invariant across the inner loop.

## Scope

This backend currently lowers softmax normalization:

```text
y[i, j] = x[i, j] / sum[i]
```

It does not yet lower full numerically safe softmax with max subtraction,
`exp`, reduction, and warp-level parallel reductions. Those are future
compiler/backend extensions.

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

macOS with Homebrew LLVM can build the MLIR-only lanes:

```bash
cmake -S . -B build -G Ninja \
  -DLLVM_DIR=/opt/homebrew/opt/llvm/lib/cmake/llvm \
  -DMLIR_DIR=/opt/homebrew/opt/llvm/lib/cmake/mlir
cmake --build build -j
```

## Verification

Native MLIR/LLVM tests:

```bash
ctest --test-dir build --output-on-failure
```

Local GPU verification gate:

```bash
scripts/verify_wsl_gpu.sh
```

The GPU gate records the NVIDIA device, driver, and CUDA toolkit. It then
runs CMake, CTest, explicit driver artifact checks, `mlc-demo --verify`,
and the pass-static-analysis shape smoke test.

CI runs the native MLIR/LLVM lane on Ubuntu 24.04. CUDA-dependent checks
are skip-capable in CI because GitHub-hosted runners do not provide a GPU.

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

On systems without an NVIDIA GPU or CUDA driver, the demo exits with a
`SKIP` message. The local GPU verification script treats an unexpected
skip as a failure.

## GPU Runtime Benchmark

```bash
./build/tools/mlc-gpu-bench/mlc-gpu-bench \
  --sizes=1024,4096,16384,65536 \
  --warmup=25 \
  --iters=100 \
  --mode=both \
  --verify
```

This benchmarks the generated PTX kernel directly on CUDA and reports
steady-state kernel time, effective bandwidth, numerical error, and
optimized-vs-baseline speedup. It is a runtime throughput benchmark.

## Pass Analysis

```bash
./build/bin/mlc-pass-analysis \
  --shapes=64x64,64x128,128x128,128x256,256x256,256x512,512x512,512x1024,1024x1024,2048x1024
```

Reports `arith.divf` counts before and after the optimization pass, an
estimated dynamic-division reduction, and pipeline wall time. This is
compile-time/static analysis of the pass effect, not runtime throughput.

## Optional Triton Comparison

The repo does not build Triton into the core C++ toolchain, but it now ships
an optional Python comparison path for the same normalization kernel shape:

```bash
python benchmarks/softmax_gpu_bench.py \
  --mlc-bench ./build/tools/mlc-gpu-bench/mlc-gpu-bench \
  --sizes=1024,4096,16384,65536
```

If `torch` and `triton` are installed in your Python environment, the script
prints `mlc` baseline/optimized rows alongside a Triton row for the same
vector sizes. `benchmarks/triton_softmax.py` is also runnable on its own.

## Repository Map

- `lib/Passes/`: custom MLIR optimization pass.
- `compiler/pipeline/`: staged MLIR-to-PTX lowering pipeline.
- `runtime/`: C++ CUDA Driver runtime wrapper for generated PTX.
- `tools/mlc-opt`: pass runner.
- `tools/mlc-driver`: end-to-end artifact generator.
- `tools/mlc-demo`: GPU launch and numerical verification.
- `tools/mlc-gpu-bench`: generated-PTX CUDA runtime benchmark.
- `benchmarks/`: pass-static-analysis tool.
- `examples/`: MLIR inputs.
- `test/` and `tests/`: FileCheck, CTest, shell, and C++ tests.
- `scripts/verify_wsl_gpu.sh`: local GPU-backed verification gate.
