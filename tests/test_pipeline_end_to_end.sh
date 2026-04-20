#!/usr/bin/env bash
set -euo pipefail

MLC_DRIVER="$1"
INPUT_MLIR="$2"
WORKDIR="$3"

BASELINE_DIR="${WORKDIR}/baseline"
OPT_DIR="${WORKDIR}/optimized"
EXPECTED_STAGE_COUNT=5

rm -rf "${WORKDIR}"
mkdir -p "${BASELINE_DIR}" "${OPT_DIR}"

"${MLC_DRIVER}" --input "${INPUT_MLIR}" --output-dir "${BASELINE_DIR}" --mode baseline
"${MLC_DRIVER}" --input "${INPUT_MLIR}" --output-dir "${OPT_DIR}" --mode optimized

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
if grep -q "arith.mulf" "${BASELINE_DIR}/stage1_optimized.mlir"; then
  echo "Baseline mode unexpectedly contains arith.mulf in stage1 output." >&2
  exit 1
fi

grep -q "arith.divf" "${OPT_DIR}/stage1_optimized.mlir"
grep -q "arith.mulf" "${OPT_DIR}/stage1_optimized.mlir"
if awk '/scf\.for/ { in_loop = 1 } in_loop && /arith\.divf/ { found = 1 } END { exit found ? 0 : 1 }' \
  "${OPT_DIR}/stage1_optimized.mlir"; then
  echo "Optimized mode still divides loop values by the invariant denominator." >&2
  exit 1
fi

if [[ "$(find "${OPT_DIR}" -maxdepth 1 -type f | wc -l | tr -d ' ')" -lt "${EXPECTED_STAGE_COUNT}" ]]; then
  echo "Optimized output directory is missing one or more pipeline artifacts." >&2
  exit 1
fi

grep -q "define" "${OPT_DIR}/stage3_llvm_ir.ll"
grep -q "softmax_kernel" "${OPT_DIR}/stage3_llvm_ir.ll"
grep -Eq "\\.visible[[:space:]]+\\.entry[[:space:]]+softmax_kernel|\\.entry[[:space:]]+softmax_kernel" "${OPT_DIR}/stage4_kernel.ptx"
