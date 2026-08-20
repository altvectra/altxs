#!/usr/bin/env bash
# bitstream → payload_sim stream → peel inverse → enwik9
set -euo pipefail
# shellcheck source=common.sh
. "$(cd "$(dirname "$0")" && pwd)/common.sh"

BITSTREAM=""
DECODER_ZIP=""
OUT="${ROOT}/work/enwik9"
WEIGHTS_DIR="${ROOT}/weights"
SIDECAR="${ROOT}/sidecars/payload_sim.trailer.bin"
WORK="${ROOT}/work"
AC_BYTES=""

usage() {
  cat <<EOF
Usage: $0 --bitstream payload_final_fullsha.bin [--out work/enwik9]
          [--decoder-zip blsmc_ac_decoder.zip] [--weights-dir weights]
          [--sidecar sidecars/payload_sim.trailer.bin] [--bytes N]

--bytes N decodes the first N symbols (4 MiB = 4194304) and skips peel
inverse (the result is not a full product). Default is AC_N_SYMBOLS from
DECODE.env (full stream). See VALIDATE.md.
EOF
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bitstream) BITSTREAM="$2"; shift 2 ;;
    --decoder-zip) DECODER_ZIP="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --weights-dir) WEIGHTS_DIR="$2"; shift 2 ;;
    --sidecar) SIDECAR="$2"; shift 2 ;;
    --bytes) AC_BYTES="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) fail "unknown arg: $1" ;;
  esac
done

[[ -n "${BITSTREAM}" && -f "${BITSTREAM}" ]] || fail "--bitstream is required"
load_decode_env
mkdir -p "${WORK}" "${OUT%/*}"

if [[ -n "${DECODER_ZIP}" ]]; then
  [[ -f "${DECODER_ZIP}" ]] || fail "missing decoder zip: ${DECODER_ZIP}"
  UNPACK="${WORK}/decoder_unpack"
  rm -rf "${UNPACK}"
  mkdir -p "${UNPACK}"
  unzip -oq "${DECODER_ZIP}" -d "${UNPACK}"
  PKG="${UNPACK}/blsmc_ac_decoder"
  [[ -d "${PKG}" ]] || PKG="${UNPACK}"
  WEIGHTS_DIR="${PKG}/weights"
  SIDECAR="${PKG}/sidecars/payload_sim.trailer.bin"
  BLSMC="${PKG}/bin/blsmc_prepare"
  DIC="${PKG}/dict/english.dic"
  if [[ -f "${PKG}/DECODE.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    . "${PKG}/DECODE.env"
    set +a
  fi
  export PYTHONPATH="${PKG}/code${PYTHONPATH:+:${PYTHONPATH}}"
else
  BLSMC="${ROOT}/blsmc/prepare/blsmc_prepare"
  DIC="${ROOT}/dict/english.dic"
  [[ -x "${BLSMC}" ]] || fail "build peel first: make -C blsmc/prepare"
  [[ -f "${DIC}" ]] || fail "missing ${DIC}; run ./scripts/fetch_vendors.sh"
fi

# Prefix decode (--bytes N) does not re-seal or peel, so the trailer is unused.
if [[ -z "${AC_BYTES}" ]]; then
  [[ -f "${SIDECAR}" ]] || fail "missing sidecar trailer: ${SIDECAR}"
fi
CODEC="$(find_codec "${WEIGHTS_DIR}")"
DENSE="${WEIGHTS_DIR}/student_dense.safetensors"

echo "[1/4] codec → dense student"
"${PY}" -m hyperflow_distillation.mixed_bit_delta decode \
  --codec "${CODEC}" --out "${DENSE}"

if [[ -z "${AC_BYTES}" ]]; then
  AC_BYTES="${AC_N_SYMBOLS}"
fi

echo "[2/4] AC blind decode → payload_sim stream (${AC_BYTES} symbols)"
"${PY}" -m xsa_ttt.train --eval-only "${DENSE}" \
  --decode-payload "${BITSTREAM}" \
  --ac-bytes "${AC_BYTES}" \
  --decoded-path "${WORK}/payload_sim_stream.bin" \
  --out "${WORK}/ac_decode" --profile "${PROFILE}" --seed "${SEED}"

if [[ "${AC_BYTES}" != "${AC_N_SYMBOLS}" ]]; then
  echo "prefix decode (${AC_BYTES} B); skipping peel inverse (not a full product)"
  echo "wrote ${WORK}/payload_sim_stream.bin"
  exit 0
fi

echo "[3/4] re-seal product"
cat "${WORK}/payload_sim_stream.bin" "${SIDECAR}" > "${WORK}/payload_sim.product"

echo "[4/4] peel inverse"
mkdir -p "${WORK}/peel"
chmod +x "${BLSMC}" 2>/dev/null || true
"${BLSMC}" decode "${WORK}/payload_sim.product" "${OUT}" \
  --dict "${DIC}" --workdir "${WORK}/peel"

size=$(wc -c < "${OUT}" | tr -d ' ')
echo "wrote ${OUT} (${size} B)"
if [[ "${size}" != "1000000000" ]]; then
  echo "warning: size is ${size}, expected 1000000000" >&2
fi
if command -v shasum >/dev/null 2>&1; then
  echo "sha256  $(shasum -a 256 "${OUT}" | awk '{print $1}')"
  echo "sha1    $(shasum -a 1 "${OUT}" | awk '{print $1}')"
elif command -v sha256sum >/dev/null 2>&1; then
  echo "sha256  $(sha256sum "${OUT}" | awk '{print $1}')"
  echo "sha1    $(sha1sum "${OUT}" | awk '{print $1}')"
fi
if command -v md5 >/dev/null 2>&1; then
  echo "md5     $(md5 -q "${OUT}")"
elif command -v md5sum >/dev/null 2>&1; then
  echo "md5     $(md5sum "${OUT}" | awk '{print $1}')"
fi
