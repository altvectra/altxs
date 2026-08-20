# Shared helpers. Sourced by other scripts. Do not execute.
# shellcheck shell=bash

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PY="${ROOT}/.venv/bin/python"
else
  PY="${PY:-python3}"
fi

fail() { echo "error: $*" >&2; exit 1; }

load_decode_env() {
  set -a
  # shellcheck disable=SC1091
  . "${ROOT}/DECODE.env"
  set +a
}

find_codec() {
  local d="${1:-${ROOT}/weights}"
  local f
  f="$(ls -t "${d}"/mixed_da_bpw*.safetensors 2>/dev/null | grep -v _anchor | head -1 || true)"
  [[ -n "${f}" ]] || fail "no mixed_da_bpw*.safetensors under ${d}"
  echo "${f}"
}
