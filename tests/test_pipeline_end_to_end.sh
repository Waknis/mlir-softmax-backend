#!/usr/bin/env bash
set -euo pipefail

MLC_DRIVER="$1"
INPUT_MLIR="$2"
WORKDIR="$3"

BASELINE_DIR="${WORKDIR}/baseline"
OPT_DIR="${WORKDIR}/optimized"

rm -rf "${WORKDIR}"
mkdir -p "${BASELINE_DIR}" "${OPT_DIR}"

"${MLC_DRIVER}" --input "${INPUT_MLIR}" --output-dir "${BASELINE_DIR}" --mode baseline
"${MLC_DRIVER}" --input "${INPUT_MLIR}" --output-dir "${OPT_DIR}" --mode optimized

[[ -s "${BASELINE_DIR}/stage1_optimized.mlir" ]]
[[ -s "${OPT_DIR}/stage1_optimized.mlir" ]]
[[ -s "${OPT_DIR}/stage3_llvm_ir.ll" ]]
[[ -s "${OPT_DIR}/stage4_kernel.ptx" ]]

grep -q "arith.divf" "${BASELINE_DIR}/stage1_optimized.mlir"
grep -q "arith.mulf" "${OPT_DIR}/stage1_optimized.mlir"
grep -q "define" "${OPT_DIR}/stage3_llvm_ir.ll"
grep -Eq "\\.func|\\.entry" "${OPT_DIR}/stage4_kernel.ptx"
