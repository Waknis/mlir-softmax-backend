# Architecture

For the current end-to-end compiler flow, see [pipeline.md](pipeline.md).

## Project Shape

This repository is organized around one primary path: a C++ MLIR backend
that lowers a softmax-style MLIR program to PTX and can execute it through
the CUDA Driver API.

Baseline kernels and Python experiments remain in the repo, but they are
not part of the core backend:

- **Core backend:** `lib/Passes/`, `compiler/pipeline/`, `runtime/`, and
  `tools/`.
- **Performance baselines:** `kernels/`, `benchmarks/softmax_gpu_bench.py`,
  `benchmarks/triton_softmax.py`, and profiling artifacts under
  `docs/profiling/`.
- **Experiments:** `experiments/fx_nvrtc/`, an optional PyTorch FX to NVRTC
  elementwise compiler.

## Core Backend

- `lib/Passes/`: custom `mlc-div-to-reciprocal-mul` pass. It hoists
  loop-invariant floating-point division into one reciprocal outside the
  loop and replaces in-loop divides with multiplies.
- `compiler/pipeline/`: staged MLIR-to-PTX lowering. It parses MLIR, applies
  the selected optimization mode, lowers to LLVM dialect, exports LLVM IR,
  injects the CUDA kernel wrapper, and emits NVPTX PTX with `llc`.
- `runtime/`: C++ CUDA Driver API wrapper. It dynamically loads
  `libcuda.so`, loads generated PTX, manages device buffers, launches the
  generated kernel, and copies results back for verification.
- `tools/`: native CLIs. `mlc-opt` runs the pass, `mlc-driver` emits all
  compiler artifacts, `mlc-demo` launches generated PTX, and
  `mlc-pass-analysis` reports compile-time pass effects.

## Baselines

- `kernels/softmax_online.cu`: hand-written CUDA row-wise softmax baselines
  covering online f32, naive f32, and online f16 implementations.
- `benchmarks/triton_softmax.py`: Triton row-wise softmax baseline.
- `benchmarks/softmax_gpu_bench.py`: CUDA-event benchmark harness for the
  non-MLIR baselines. It reports runtime, bandwidth, and percent of peak
  DRAM bandwidth.
- `docs/profiling/`: committed Nsight Compute / Nsight Systems reports and
  the roofline plot. These artifacts describe the baseline kernels and set
  a performance target for future MLIR-generated softmax kernels.

## Experiments

- `experiments/fx_nvrtc/`: optional PyTorch FX to NVRTC elementwise compiler.
  Import it with `from experiments.fx_nvrtc import compile_module`.
- The experiment has its own parser, CUDA source generator, cache, launcher,
  and NVRTC/CUDA Driver bindings. It is tested, but intentionally separate
  from the MLIR backend namespace.

## Testing

- `test/`: FileCheck and CTest coverage for native MLIR behavior.
- `tests/`: Python tests for the FX/NVRTC experiment, Python helpers, and
  CUDA softmax baseline correctness.
- `scripts/verify_wsl_gpu.sh`: local GPU verification gate covering Pytest,
  CMake, CTest, driver artifacts, `mlc-demo --verify`, and the pass-analysis
  shape smoke test.
