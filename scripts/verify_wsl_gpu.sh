#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build/gpu-verify}"
LLVM_DIR="${LLVM_DIR:-/usr/lib/llvm-15/lib/cmake/llvm}"
MLIR_DIR="${MLIR_DIR:-/usr/lib/llvm-15/lib/cmake/mlir}"
SHAPES="64x64,64x128,128x128,128x256,256x256,256x512,512x512,512x1024,1024x1024,2048x1024"

section() {
  printf '\n== %s ==\n' "$1"
}

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

section "Environment"
need_cmd nvidia-smi
need_cmd cmake
need_cmd ninja

nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version | tail -n 1
elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
  /usr/local/cuda/bin/nvcc --version | tail -n 1
else
  echo "nvcc not found; continuing because this backend uses the CUDA Driver API at runtime."
fi

if [[ ! -d "${LLVM_DIR}" || ! -d "${MLIR_DIR}" ]]; then
  cat >&2 <<EOF
LLVM/MLIR 15 CMake packages were not found.
Expected:
  LLVM_DIR=${LLVM_DIR}
  MLIR_DIR=${MLIR_DIR}

On Ubuntu 24.04, install:
  sudo apt-get update
  sudo apt-get install -y build-essential cmake ninja-build llvm-15 llvm-15-dev llvm-15-tools libmlir-15-dev mlir-15-tools
EOF
  exit 1
fi

section "Configure and build"
cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -G Ninja \
  -DLLVM_DIR="${LLVM_DIR}" \
  -DMLIR_DIR="${MLIR_DIR}" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" -j "${BUILD_JOBS:-$(nproc)}"

section "CTest"
ctest --test-dir "${BUILD_DIR}" --output-on-failure

section "Driver artifacts"
DRIVER="${BUILD_DIR}/tools/mlc-driver/mlc-driver"
DEMO="${BUILD_DIR}/tools/mlc-demo/mlc-demo"
BENCH="${BUILD_DIR}/bin/mlc-pass-analysis"
GPU_BENCH="${BUILD_DIR}/tools/mlc-gpu-bench/mlc-gpu-bench"
ARTIFACT_DIR="${BUILD_DIR}/manual-artifacts"
BASELINE_DIR="${ARTIFACT_DIR}/baseline"
OPT_DIR="${ARTIFACT_DIR}/optimized"

rm -rf "${ARTIFACT_DIR}"
mkdir -p "${BASELINE_DIR}" "${OPT_DIR}"

"${DRIVER}" --input "${ROOT_DIR}/examples/softmax.mlir" --output-dir "${BASELINE_DIR}" --mode baseline
"${DRIVER}" --input "${ROOT_DIR}/examples/softmax.mlir" --output-dir "${OPT_DIR}" --mode optimized

for dir in "${BASELINE_DIR}" "${OPT_DIR}"; do
  for stage in \
    stage0_input.mlir \
    stage1_optimized.mlir \
    stage2_llvm_dialect.mlir \
    stage3_llvm_ir.ll \
    stage4_kernel.ptx; do
    [[ -s "${dir}/${stage}" ]]
  done
done

grep -q "arith.divf" "${BASELINE_DIR}/stage1_optimized.mlir"
grep -q "arith.divf" "${OPT_DIR}/stage1_optimized.mlir"
grep -q "arith.mulf" "${OPT_DIR}/stage1_optimized.mlir"
if awk '/scf\.for/ { in_loop = 1 } in_loop && /arith\.divf/ { found = 1 } END { exit found ? 0 : 1 }' \
  "${OPT_DIR}/stage1_optimized.mlir"; then
  echo "Optimized mode still contains an in-loop arith.divf." >&2
  exit 1
fi
grep -q "softmax_kernel" "${OPT_DIR}/stage3_llvm_ir.ll"
grep -Eq "\\.visible[[:space:]]+\\.entry[[:space:]]+softmax_kernel|\\.entry[[:space:]]+softmax_kernel" "${OPT_DIR}/stage4_kernel.ptx"

section "GPU demo verify"
DEMO_OUTPUT="$("${DEMO}" --input "${ROOT_DIR}/examples/softmax.mlir" --output-dir "${BUILD_DIR}/manual-demo" --mode optimized --n 1024 --verify)"
printf '%s\n' "${DEMO_OUTPUT}"
if printf '%s\n' "${DEMO_OUTPUT}" | grep -q '^SKIP:'; then
  echo "GPU verification skipped unexpectedly." >&2
  exit 1
fi
printf '%s\n' "${DEMO_OUTPUT}" | grep -q "Demo completed"
printf '%s\n' "${DEMO_OUTPUT}" | grep -Eq "max_abs_err=|abs_err="

section "GPU runtime benchmark smoke"
GPU_BENCH_OUTPUT="$("${GPU_BENCH}" --sizes=1024 --warmup=5 --iters=25 --output-root "${BUILD_DIR}/manual-gpu-bench" --verify)"
printf '%s\n' "${GPU_BENCH_OUTPUT}"
if printf '%s\n' "${GPU_BENCH_OUTPUT}" | grep -q '^SKIP:'; then
  echo "GPU runtime benchmark skipped unexpectedly." >&2
  exit 1
fi
printf '%s\n' "${GPU_BENCH_OUTPUT}" | head -n 1 | grep -q "mode"
GPU_ROWS="$(printf '%s\n' "${GPU_BENCH_OUTPUT}" | grep -E '^(baseline|optimized)[[:space:]]+1024' | wc -l | tr -d ' ')"
if [[ "${GPU_ROWS}" -lt 2 ]]; then
  echo "Expected baseline and optimized GPU benchmark rows, got ${GPU_ROWS}." >&2
  exit 1
fi

section "Pass-analysis shape smoke"
BENCH_OUTPUT="$("${BENCH}" --shapes="${SHAPES}")"
printf '%s\n' "${BENCH_OUTPUT}"
ROWS="$(printf '%s\n' "${BENCH_OUTPUT}" | grep -E '^[0-9]+x[0-9]+' | wc -l | tr -d ' ')"
if [[ "${ROWS}" -lt 10 ]]; then
  echo "Expected at least 10 benchmark rows, got ${ROWS}." >&2
  exit 1
fi

section "Done"
echo "GPU-backed verification completed successfully."
