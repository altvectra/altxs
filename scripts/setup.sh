#!/usr/bin/env bash
# Procure deps needed to *produce* payload_sim and (optionally) AC-encode it.
#
# Always:
#   1. verify/refresh vendored cmix-lex peel sources + dict tables
#   2. build blsmc_prepare
#
# Optional:
#   WITH_PYTHON=1   uv sync --extra dev (.venv; torch / safetensors / triton)
#   WITH_ENWIK9=1   download official enwik9 into data/
#   WITH_PEEL=1     run peel after enwik9 is present
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "[1/4] peel vendors (cmix-lex subset + english.dic + article order)"
"${ROOT}/scripts/fetch_vendors.sh"

echo "[2/4] build blsmc_prepare + UPX (required for Total S)"
command -v clang++ >/dev/null 2>&1 || command -v c++ >/dev/null 2>&1 || {
  echo "error: need a C++17 compiler (clang++ or c++)" >&2
  exit 1
}
# Fresh objects for this machine (stale arm64 .o + x86_64 c++ fails to link).
make -C "${ROOT}/blsmc/prepare" clean
make -C "${ROOT}/blsmc/prepare"
"${ROOT}/scripts/fetch_upx.sh"

if [[ "${WITH_PYTHON:-0}" == "1" ]]; then
  echo "[3/4] Python venv + pip"
  "${ROOT}/scripts/setup_python.sh"
else
  echo "[3/4] skip Python (WITH_PYTHON=1 to install torch/safetensors)"
fi

if [[ "${WITH_ENWIK9:-0}" == "1" ]]; then
  echo "[4/4] official enwik9"
  "${ROOT}/scripts/fetch_enwik9.sh"
else
  echo "[4/4] skip enwik9 (WITH_ENWIK9=1 to download ~1 GB)"
fi

if [[ "${WITH_PEEL:-0}" == "1" ]]; then
  "${ROOT}/scripts/peel.sh"
fi

echo
echo "payload path:"
echo "  ./scripts/peel.sh"
echo "    → data/enwik9.blsmc_full.m3v2.payload_sim"
echo "    AC stream 576,278,322 B + M3/BLSMETA1 trailer (not AC'd)"
