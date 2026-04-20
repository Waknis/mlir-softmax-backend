# MLIR Softmax Backend

[![CI](https://github.com/Waknis/mlir-softmax-backend/actions/workflows/ci.yml/badge.svg)](https://github.com/Waknis/mlir-softmax-backend/actions/workflows/ci.yml)

Two things live in this repo, deliberately side-by-side:

1. **A compact MLIR/LLVM backend** that lowers a softmax-style MLIR program
   to PTX and executes it through the CUDA Driver API, with a custom
   optimization pass (`mlc-div-to-reciprocal-mul`).
2. **Hand-written online softmax CUDA kernels** plus a Triton baseline and
   a GPU benchmark harness, so the repo's "performance" framing is backed
   by measured numbers — Nsight Compute / Nsight Systems captures and a
   roofline plot included.

```text
MLIR input -> custom optimization pass -> LLVM dialect -> LLVM IR -> PTX -> CUDA Driver launch
hand CUDA  -> nvcc --use_fast_math     -> .so          -> ctypes -> Python benchmark harness
Triton     -> triton.jit               -> PTX (JIT)    -> Python benchmark harness
```

## Results

Row-wise softmax on an NVIDIA GeForce RTX 5060 Ti (SM 12.0, GDDR7, 448 GB/s
peak DRAM bandwidth). Time is the median of 200 CUDA-event-timed launches
after 50 untimed warmup iters; a 64 MB buffer is zeroed between each launch
to flush L2 so the bandwidth numbers reflect DRAM-cold traffic.

| Shape      | Backend            | ms (med) | GB/s | % peak BW |
|------------|--------------------|---------:|-----:|----------:|
| 1024x4096  | `hand_online_f32`  |   0.091  |  371 |     82.7% |
| 1024x4096  | `hand_naive_f32`   |   0.093  |  360 |     80.3% |
| 1024x4096  | `hand_online_f16`  |   0.047  |  356 |     79.4% |
| 1024x4096  | `triton_f32`       |   0.089  |  379 |     84.6% |
| 4096x4096  | `hand_online_f32`  |   0.353  |  380 |     84.9% |
| 4096x4096  | `hand_naive_f32`   |   0.359  |  374 |     83.5% |
| 4096x4096  | `hand_online_f16`  |   0.181  |  371 |     82.9% |
| 4096x4096  | `triton_f32`       |   0.349  |  385 |     85.9% |
| 8192x8192  | `hand_online_f32`  |   1.422  |  378 |     84.3% |
| 8192x8192  | `hand_naive_f32`   |   1.428  |  376 |     83.9% |
| 8192x8192  | `hand_online_f16`  |   0.709  |  379 |     84.5% |
| 8192x8192  | `triton_f32`       |   1.391  |  386 |     86.2% |

Full sweep across seven shapes in [`docs/results.md`](docs/results.md);
machine-readable in [`docs/results.json`](docs/results.json) /
[`docs/results.csv`](docs/results.csv).

**Read:** every implementation hits 80-86% of peak DRAM bandwidth on
shapes large enough to exceed the 32 MB L2. Softmax is a bandwidth-bound
kernel — at this shape, there's ~15 percentage points of headroom to the
DRAM ceiling and no headroom at all on the compute ceiling (SM throughput
< 25% in all four cases — see the [roofline](docs/profiling/roofline.md)).
Differences between backends at large shapes are within ~4 pp of each
other: the online (fused) kernel wins in theory by making two row
traversals instead of three, but that advantage is partly eroded by L2
hits on the second pass (ncu shows a 28% L2 hit rate on `hand_online_f32`
at 4096x4096). The f16 variant delivers the largest absolute speedup — a
clean ~2x — by halving the bytes moved while keeping DRAM throughput
unchanged.

Profiling artifacts:
- [`docs/profiling/softmax_online_ncu.md`](docs/profiling/softmax_online_ncu.md) —
  Nsight Compute (speed-of-light, memory workload, stall reasons) for all
  four backends at 4096x4096.
- [`docs/profiling/softmax_nsys.md`](docs/profiling/softmax_nsys.md) —
  Nsight Systems kernel summary across a 50-iter sweep.
- [`docs/profiling/roofline.md`](docs/profiling/roofline.md) — roofline
  plot with all four operating points.

## What This Demonstrates

- MLIR parsing, pass execution, LLVM dialect lowering, LLVM IR export,
  NVPTX codegen, and CUDA Driver execution in C++17.
- A custom MLIR pass, `mlc-div-to-reciprocal-mul`, that hoists
  loop-invariant floating-point division into a reciprocal multiply.
  Exercised on 1-D and nested 2-D loops via FileCheck tests.
- Runtime loading of `libcuda.so`, PTX JIT module loading, device memory
  management, kernel launch, and host-side numerical verification.
- Hand-written CUDA online softmax with warp-shuffle reduction
  (`__shfl_xor_sync`) and shared-memory inter-warp merge, templated over
  block sizes (32 / 64 / 128 / 256 / 512) and dispatched per row width.
- A Triton reference implementation and a CUDA-events / L2-flush benchmark
  harness that reports median, p95, GB/s, and % of peak DRAM bandwidth.
- Nsight Compute / Nsight Systems profiling captures and a roofline plot.
- FileCheck, CTest, Python unit tests (correctness + numerical stability),
  CI, and a local GPU verification gate.

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

The pass also handles nested loops (see
[`examples/softmax_rowwise.mlir`](examples/softmax_rowwise.mlir) and the
2-D FileCheck test). Loop-variant denominators are intentionally left
unchanged and covered by FileCheck tests.

### Scope of the MLIR pipeline vs. the hand kernels

The MLIR pipeline currently lowers the divide-and-normalize step of
softmax (y[i,j] = x[i,j] / sum[i]) through `arith + func + memref + scf`
to LLVM and then to PTX. The full softmax algorithm (safe max subtraction,
fused exp + reduce, online update) is implemented in
[`kernels/softmax_online.cu`](kernels/softmax_online.cu) and in Triton,
and those are the kernels the runtime benchmark measures. Adding `math.exp`
to the MLIR pipeline is future work — the stretch goal is MLIR-emitted
warp-shuffle reductions, which would make the pipeline perf-competitive
with the hand kernel.

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

macOS with Homebrew LLVM (MLIR-only lanes; CUDA lanes require Linux+NVIDIA):

```bash
cmake -S . -B build -G Ninja \
  -DLLVM_DIR=/opt/homebrew/opt/llvm/lib/cmake/llvm \
  -DMLIR_DIR=/opt/homebrew/opt/llvm/lib/cmake/mlir
cmake --build build -j
```

The CUDA kernel library (`kernels/libmlc_softmax_kernels.so`) builds
automatically when `nvcc` is on `PATH`; if not, the kernels/ subdirectory
silently skips and the Python benchmark prints a warning.

Python:

```bash
python -m pip install -e ".[dev,triton]"
```

The `triton` extra is optional (Linux + NVIDIA only). Without it, the
benchmark harness omits the Triton column.

## Verification

CPU/Python tests:

```bash
python -m pytest -q
```

Native MLIR/LLVM/CUDA tests:

```bash
ctest --test-dir build --output-on-failure
```

Local GPU verification gate:

```bash
scripts/verify_wsl_gpu.sh
```

The GPU gate records the NVIDIA device, driver, CUDA toolkit, PyTorch CUDA
availability, and NVRTC target. It then runs Python CUDA correctness
tests, CMake, CTest, explicit driver artifact checks, `mlc-demo --verify`,
and the pass-static-analysis shape smoke test.

Validated local environment:

```text
WSL2 Ubuntu 24.04
NVIDIA GeForce RTX 5060 Ti, compute capability 12.0
NVIDIA driver 595.97
CUDA toolkit 13.2
```

CI runs the CPU/Python and native MLIR/LLVM lanes on Ubuntu 24.04.
CUDA-dependent tests (both the MLIR runtime path and the hand-CUDA
kernel tests) are skip-capable in CI because GitHub-hosted runners do not
provide a GPU.

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

## Benchmarks

There are two benchmark harnesses here; they measure different things.

### Pass static analysis (compile time)

```bash
./build/bin/mlc-pass-analysis \
  --shapes=64x64,64x128,128x128,128x256,256x256,256x512,512x512,512x1024,1024x1024,2048x1024
```

Reports `arith.divf` counts before and after the optimization pass, an
estimated dynamic-division reduction, and pipeline wall time. This is
**compile-time analysis of the pass effect**, not runtime throughput — the
synthetic MLIR used here is a softmax-*shaped* loop, not the actual
softmax algorithm.

### GPU runtime benchmark

```bash
python -m benchmarks.softmax_gpu_bench \
  --shapes 1024x4096 4096x4096 8192x8192 \
  --md docs/results.md
```

Measures per-launch runtime, achieved DRAM bandwidth, and % of peak across
four softmax implementations on the same shapes / dtypes:

| Backend            | Description                                                |
| ------------------ | ---------------------------------------------------------- |
| `hand_online_f32`  | Hand CUDA, fused online softmax (Milakov & Gimelshein)     |
| `hand_naive_f32`   | Hand CUDA, classical 3-pass safe softmax                   |
| `hand_online_f16`  | Hand CUDA, f16 storage / f32 accumulate                    |
| `triton_f32`       | Tutorial-style per-row Triton kernel                       |

Timing uses CUDA events (one launch per event pair), 50 warmup iters, 200
timed iters, and flushes a 64 MB buffer to L2 between each launch (so
measurements reflect DRAM-cold per-launch throughput, not L2 hits on
small shapes). Pass `--no-flush-l2` to see the warm-cache numbers instead.

## Repository Map

- `lib/Passes/`: custom MLIR optimization pass.
- `compiler/pipeline/`: staged MLIR-to-PTX lowering pipeline (C++).
- `compiler/*.py`: **experimental** PyTorch FX -> CUDA codegen path,
  separate from the MLIR pipeline. Used by the Python CUDA tests.
- `runtime/`: CUDA Driver runtime wrapper.
- `kernels/`: hand-written online softmax CUDA kernels + ctypes loader.
- `tools/mlc-opt`: pass runner.
- `tools/mlc-driver`: end-to-end artifact generator.
- `tools/mlc-demo`: GPU launch and numerical verification.
- `benchmarks/`: GPU runtime benchmark, Triton baseline, pass-static-analysis
  tool, and a minimal harness for ncu/nsys.
- `docs/`: benchmark results + profiling artifacts (ncu reports, nsys
  summary, roofline plot).
- `test/` and `tests/`: FileCheck, CTest, shell, and Python tests.
- `scripts/verify_wsl_gpu.sh`: local GPU-backed verification gate.
