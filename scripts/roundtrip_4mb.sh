#!/usr/bin/env bash
# Incremental AC encode → blind decode → byte-identical prefix.
# Default N=4 MiB (4,194,304 symbols). Same DECODE.env as production.
# GPU required. Not CI. Full 1 GB: ./scripts/test_enwik9.sh (days).
set -euo pipefail
# shellcheck source=common.sh
. "$(cd "$(dirname "$0")" && pwd)/common.sh"

BYTES=4194304
PRODUCT="${ROOT}/data/enwik9.blsmc_full.m3v2.payload_sim"
WEIGHTS_DIR="${ROOT}/weights"
OUT="${ROOT}/work/roundtrip_4mb"
BITSTREAM=""
SKIP_ENCODE=0

usage() {
  cat <<EOF
Usage: $0 [--bytes 4194304] [--product data/enwik9.blsmc_full.m3v2.payload_sim]
          [--out work/roundtrip_4mb] [--weights-dir weights]
          [--bitstream payload_final_fullsha.bin]

Default is a self-contained 4 MiB encode + blind decode of the payload_sim
prefix (the check reviewers should run). Pass --bitstream to skip encode and
decode the first --bytes symbols from an existing stream (Release or a prior
prefix encode).

This is not CI. Expect an H100-class GPU. 4 MiB decode is hours, not minutes.
Full 576,278,322-symbol reconstruct: ./scripts/test_enwik9.sh (days).
EOF
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bytes) BYTES="$2"; shift 2 ;;
    --product) PRODUCT="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --weights-dir) WEIGHTS_DIR="$2"; shift 2 ;;
    --bitstream) BITSTREAM="$2"; SKIP_ENCODE=1; shift 2 ;;
    -h|--help) usage ;;
    *) fail "unknown arg: $1" ;;
  esac
done

[[ "${BYTES}" =~ ^[1-9][0-9]*$ ]] || fail "--bytes must be a positive integer"
load_decode_env
export COMPRESSION_DETERMINISTIC="${COMPRESSION_DETERMINISTIC:-strict}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

[[ -f "${PRODUCT}" ]] || fail "missing ${PRODUCT}; run ./scripts/peel.sh (or WITH_PEEL=1 ./scripts/setup.sh)"
prod_len=$(wc -c < "${PRODUCT}" | tr -d ' ')
# payload_sim on disk includes the M3/BLSMETA1 trailer; AC scores the stripped stream.
# 4 MiB is well under the 576,278,322-byte AC prefix, so head -c is the product prefix.
[[ "${prod_len}" -ge "${BYTES}" ]] || fail "${PRODUCT} is ${prod_len} B, need at least ${BYTES}"

CODEC="$(find_codec "${WEIGHTS_DIR}")"
DENSE="${WEIGHTS_DIR}/student_dense.safetensors"
mkdir -p "${OUT}" "${WEIGHTS_DIR}"

echo "=== incremental AC roundtrip: N=${BYTES} B (${OUT}) ==="
echo "[codec] ${CODEC} → ${DENSE}"
"${PY}" -m hyperflow_distillation.mixed_bit_delta decode \
  --codec "${CODEC}" --out "${DENSE}"

if [[ "${SKIP_ENCODE}" -eq 0 ]]; then
  BITSTREAM="${OUT}/payload_final_fullsha.bin"
  echo "[1/3] incremental encode (${BYTES} B prefix)"
  "${PY}" -m xsa_ttt.train --eval-only "${DENSE}" \
    --data "${PRODUCT}" \
    --ac --ac-bytes "${BYTES}" --skip-tf \
    --out "${OUT}/enc" --profile "${PROFILE}" --seed "${SEED}"
  cp -f "${OUT}/enc/payload_final_fullsha.bin" "${BITSTREAM}"
else
  [[ -f "${BITSTREAM}" ]] || fail "missing --bitstream ${BITSTREAM}"
  echo "[1/3] skip encode; using ${BITSTREAM}"
fi

echo "[2/3] blind incremental decode (no source, ${BYTES} symbols)"
"${PY}" -m xsa_ttt.train --eval-only "${DENSE}" \
  --decode-payload "${BITSTREAM}" \
  --ac-bytes "${BYTES}" \
  --decoded-path "${OUT}/decoded_stream.bin" \
  --out "${OUT}/dec" --profile "${PROFILE}" --seed "${SEED}"

echo "[3/3] cmp first ${BYTES} B of payload_sim"
head -c "${BYTES}" "${PRODUCT}" > "${OUT}/expected_prefix.bin"
got=$(wc -c < "${OUT}/decoded_stream.bin" | tr -d ' ')
[[ "${got}" == "${BYTES}" ]] || fail "decoded ${got} B, expected ${BYTES}"
cmp "${OUT}/decoded_stream.bin" "${OUT}/expected_prefix.bin"

echo "OK incremental AC roundtrip"
echo "  bytes     ${BYTES}"
echo "  bitstream ${BITSTREAM} ($(wc -c < "${BITSTREAM}" | tr -d ' ') B)"
echo "  decoded   ${OUT}/decoded_stream.bin"
