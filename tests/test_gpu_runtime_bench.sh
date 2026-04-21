#!/usr/bin/env bash
set -euo pipefail

GPU_BENCH="$1"
WORKDIR="$2"

OUTPUT="$("${GPU_BENCH}" --sizes=1024 --warmup=1 --iters=3 --output-root "${WORKDIR}" --verify)"
printf '%s\n' "${OUTPUT}"

if printf '%s\n' "${OUTPUT}" | grep -q '^SKIP:'; then
  exit 0
fi

printf '%s\n' "${OUTPUT}" | head -n 1 | grep -q "mode"
ROWS="$(printf '%s\n' "${OUTPUT}" | grep -E '^(baseline|optimized)[[:space:]]+1024' | wc -l | tr -d ' ')"
if [[ "${ROWS}" -lt 2 ]]; then
  echo "Expected baseline and optimized benchmark rows, got ${ROWS}." >&2
  exit 1
fi
