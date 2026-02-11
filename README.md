# MLIR Softmax Backend (C++ / MLIR / LLVM / PTX / CUDA Driver)

This project takes a Softmax-style MLIR program and runs a full backend pipeline:

`MLIR input -> custom MLIR optimization -> LLVM dialect -> LLVM IR -> PTX -> CUDA Driver launch`

## What this project includes:
- Built with production-relevant compiler tooling: **MLIR**, **LLVM**, **CMake**, **C++17**.
- Includes a custom optimization pass with correctness tests (**FileCheck + CTest**).
- Demonstrates compiler + runtime integration: IR transformation, codegen, PTX emission, module loading, kernel launch.
- Includes reproducible benchmarking and numerical verification workflows.

## Technical Scope:
- MLIR pass development (`OperationPass`, IR rewrite, loop-invariance handling).
- Lowering pipeline construction across dialect boundaries.
- LLVM IR export and NVPTX emission (`llc`, `nvptx64-nvidia-cuda`).
- CUDA Driver API integration (dynamic loading, module load, kernel launch, device memory ops).
- Build/test automation with CMake + CTest + shell-based integration tests.

## Project Pipeline
```mermaid
flowchart LR
  A["Input MLIR"] --> B["Custom Pass: mlc-div-to-reciprocal-mul"]
  B --> C["Lower to LLVM Dialect"]
  C --> D["Export LLVM IR"]
  D --> E["llc (nvptx64) -> PTX"]
  E --> F["CUDA Driver Load + Launch"]
  F --> G["Reference Check"]
```

## Custom Optimization Pass
**Pass:** `mlc-div-to-reciprocal-mul`

What it does:
- Detects loop-invariant denominator values inside `scf.for`.
- Hoists reciprocal computation out of loop body.
- Rewrites in-loop `arith.divf` into `arith.mulf`.

### Before (baseline)
```mlir
%inner = scf.for %i = %c0 to %cN step %c1 iter_args(%acc = %row_acc) -> (f32) {
  %d = arith.divf %one, %sum : f32
  %next = arith.addf %acc, %d : f32
  scf.yield %next : f32
}
```

### After (optimized)
```mlir
%recip = arith.divf %one, %sum : f32
%inner = scf.for %i = %c0 to %cN step %c1 iter_args(%acc = %row_acc) -> (f32) {
  %d = arith.mulf %one, %recip : f32
  %next = arith.addf %acc, %d : f32
  scf.yield %next : f32
}
```

## Measurable Impact
For benchmarked shapes from `64x64` through `2048x1024`:
- Estimated dynamic division count changes from **`M*N` -> `M`**.
- This corresponds to approximately **98.4% to 99.9% fewer divisions**, depending on shape.

A benchmark table is produced by `softmax-benchmark` for at least 10 shapes.

## Build
```bash
cmake -S . -B build -G Ninja \
  -DLLVM_DIR=/opt/homebrew/opt/llvm/lib/cmake/llvm \
  -DMLIR_DIR=/opt/homebrew/opt/llvm/lib/cmake/mlir
cmake --build build -j
```

## Tests
```bash
ctest --test-dir build --output-on-failure
```

Includes:
- FileCheck validation for custom pass behavior.
- End-to-end pipeline test (baseline vs optimized).
- GPU correctness harness (`mlc-demo` invocation).
- Benchmark-shape contract test (>=10 rows).

## End-to-End Driver
```bash
./build/tools/mlc-driver/mlc-driver \
  --input examples/softmax.mlir \
  --output-dir build/artifacts \
  --mode optimized
```

## Demo (PTX Load + CUDA Launch + Verify)
```bash
./build/tools/mlc-demo/mlc-demo \
  --input examples/softmax.mlir \
  --verify
```

Note:
- On systems without an NVIDIA CUDA driver, demo exits with a clear `SKIP` message.

## Benchmark (10 Shapes)
```bash
./build/bin/softmax-benchmark \
  --shapes=64x64,64x128,128x128,128x256,256x256,256x512,512x512,512x1024,1024x1024,2048x1024
```

Output columns:
- `shape`
- `baseline_ms`
- `optimized_ms`
- `speedup`
- `baseline_est_divs`
- `optimized_est_divs`
- `div_reduction_%`

## Repository Map
- `lib/Passes/`: custom MLIR optimization pass.
- `compiler/pipeline/`: staged lowering and PTX emission pipeline.
- `runtime/`: CUDA Driver runtime wrapper.
- `tools/mlc-opt`: pass runner.
- `tools/mlc-driver`: end-to-end pipeline driver.
- `tools/mlc-demo`: GPU demo + numerical verification.
- `benchmarks/`: benchmark harness.
- `test/` and `tests/`: FileCheck and integration tests.
