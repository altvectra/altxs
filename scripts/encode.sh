#!/usr/bin/env bash
# enwik9 → payload_sim → AC bitstream (same DECODE.env as decode).
set -euo pipefail
# shellcheck source=common.sh
. "$(cd "$(dirname "$0")" && pwd)/common.sh"

PEEL_ONLY=0
BITSTREAM_ONLY=0
ENWIK9="${ROOT}/data/enwik9"
PRODUCT="${ROOT}/data/enwik9.blsmc_full.m3v2.payload_sim"
OUT="${ROOT}/work/ac_encode"
WEIGHTS_DIR="${ROOT}/weights"
AC_BYTES=0

usage() {
  cat <<EOF
Usage: $0 [--peel-only | --bitstream-only]
          [--enwik9 data/enwik9] [--out work/ac_encode]
          [--weights-dir weights] [--bytes N]

--bytes N encodes the first N symbols of payload_sim (4 MiB = 4194304).
Default 0 = full stream (576,278,322). See VALIDATE.md.
EOF
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --peel-only) PEEL_ONLY=1; shift ;;
    --bitstream-only) BITSTREAM_ONLY=1; shift ;;
    --enwik9) ENWIK9="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --weights-dir) WEIGHTS_DIR="$2"; shift 2 ;;
    --bytes) AC_BYTES="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) fail "unknown arg: $1" ;;
  esac
done

load_decode_env
BLSMC="${ROOT}/blsmc/prepare/blsmc_prepare"
DIC="${ROOT}/dict/english.dic"
ORDER="${ROOT}/dict/new_article_order"

if [[ "${BITSTREAM_ONLY}" -ne 1 ]]; then
  [[ -f "${ENWIK9}" ]] || fail "missing ${ENWIK9}; run ./scripts/fetch_enwik9.sh"
  [[ -x "${BLSMC}" ]] || { echo "building blsmc_prepare"; make -C "${ROOT}/blsmc/prepare"; }
  [[ -f "${DIC}" ]] || fail "missing ${DIC}; run ./scripts/fetch_vendors.sh"
  [[ -f "${ORDER}" ]] || fail "missing article order; run ./scripts/fetch_vendors.sh"
  mkdir -p "${ROOT}/data/blsmc_prepare_work"
  echo "[peel] ${ENWIK9} → ${PRODUCT}"
  "${BLSMC}" encode "${ENWIK9}" "${PRODUCT}" \
    --dict "${DIC}" --order "${ORDER}" \
    --workdir "${ROOT}/data/blsmc_prepare_work"
fi

if [[ "${PEEL_ONLY}" -eq 1 ]]; then
  echo "OK peel-only ${PRODUCT}"
  exit 0
fi

[[ -f "${PRODUCT}" ]] || fail "missing ${PRODUCT}; run without --bitstream-only first"
CODEC="$(find_codec "${WEIGHTS_DIR}")"
DENSE="${WEIGHTS_DIR}/student_dense.safetensors"
mkdir -p "${OUT}" "${WEIGHTS_DIR}"

echo "[codec] ${CODEC} → ${DENSE}"
"${PY}" -m hyperflow_distillation.mixed_bit_delta decode \
  --codec "${CODEC}" --out "${DENSE}"

echo "[ac] encode ${PRODUCT}"
"${PY}" -m xsa_ttt.train --eval-only "${DENSE}" \
  --data "${PRODUCT}" \
  --ac --ac-bytes "${AC_BYTES}" --skip-tf \
  --out "${OUT}" --profile "${PROFILE}" --seed "${SEED}"

echo "bitstream: ${OUT}/payload_final_fullsha.bin"
ls -l "${OUT}"/payload_final_fullsha.bin
