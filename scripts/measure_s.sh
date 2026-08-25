#!/usr/bin/env bash
# Print S = |bitstream| + |zip -9 decoder|.
set -euo pipefail
# shellcheck source=common.sh
. "$(cd "$(dirname "$0")" && pwd)/common.sh"

BITSTREAM=""
DECODER_ZIP=""
EXPECTED_S="${EXPECTED_S:-106924811}"
EXPECTED_BITSTREAM="${EXPECTED_BITSTREAM:-93434410}"
EXPECTED_ZIP="${EXPECTED_ZIP:-13490401}"

usage() {
  cat <<EOF
Usage: $0 --bitstream payload_final_fullsha.bin --decoder-zip blsmc_ac_decoder.zip
EOF
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bitstream) BITSTREAM="$2"; shift 2 ;;
    --decoder-zip) DECODER_ZIP="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) fail "unknown arg: $1" ;;
  esac
done

[[ -f "${BITSTREAM}" ]] || fail "missing bitstream: ${BITSTREAM}"
[[ -f "${DECODER_ZIP}" ]] || fail "missing decoder zip: ${DECODER_ZIP}"

B="$(stat -c%s "${BITSTREAM}" 2>/dev/null || stat -f%z "${BITSTREAM}")"
Z="$(stat -c%s "${DECODER_ZIP}" 2>/dev/null || stat -f%z "${DECODER_ZIP}")"
S=$(( B + Z ))

printf "bitstream  %s  %s\n" "${B}" "${BITSTREAM}"
printf "decoder    %s  %s\n" "${Z}" "${DECODER_ZIP}"
printf "S          %s  (= |bitstream| + |zip|)\n" "${S}"

if [[ "${B}" == "${EXPECTED_BITSTREAM}" && "${Z}" == "${EXPECTED_ZIP}" ]]; then
  echo "matches tagged ltcb-3.15bpw (S=${EXPECTED_S})"
elif [[ "${S}" == "${EXPECTED_S}" ]]; then
  echo "S matches tagged ltcb-3.15bpw"
else
  echo "note: tagged ltcb-3.15bpw is bitstream=${EXPECTED_BITSTREAM} zip=${EXPECTED_ZIP} S=${EXPECTED_S}"
fi
