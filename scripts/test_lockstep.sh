#!/usr/bin/env bash
# Fast CI lockstep: encode → blind decode → byte-identical prefix;
# integer frequency tables match; COMPRESSION_DETERMINISTIC=strict.
set -euo pipefail
# shellcheck source=common.sh
. "$(cd "$(dirname "$0")" && pwd)/common.sh"

export COMPRESSION_DETERMINISTIC=strict
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

echo "pytest tests/test_incremental_ac_window.py (CUDA tests skip if no GPU)"
"${PY}" -m pytest "${ROOT}/tests/test_incremental_ac_window.py" \
  -q --tb=short
echo "OK lockstep"
