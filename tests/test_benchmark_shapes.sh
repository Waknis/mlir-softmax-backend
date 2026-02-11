#!/usr/bin/env bash
set -euo pipefail

SOFTMAX_BENCH="$1"
SHAPES="64x64,64x128,128x128,128x256,256x256,256x512,512x512,512x1024,1024x1024,2048x1024"

OUTPUT="$(${SOFTMAX_BENCH} --shapes="${SHAPES}")"
echo "${OUTPUT}"

ROWS=$(printf '%s\n' "${OUTPUT}" | grep -E '^[0-9]+x[0-9]+' | wc -l | tr -d ' ')
if [[ "${ROWS}" -lt 10 ]]; then
  echo "Expected at least 10 benchmark rows, got ${ROWS}." >&2
  exit 1
fi

printf '%s\n' "${OUTPUT}" | head -n 1 | grep -q "shape"
