#!/usr/bin/env bash
# Full 1 GB reconstruct + official checksums. Not CI.
# GPU, multi-day wall clock possible. See README hardware note.
set -euo pipefail
# shellcheck source=common.sh
. "$(cd "$(dirname "$0")" && pwd)/common.sh"

BITSTREAM="${BITSTREAM:-}"
DECODER_ZIP="${DECODER_ZIP:-}"
ENWIK9_OUT="${ENWIK9_OUT:-${ROOT}/work/enwik9}"
EXPECTED_SIZE=1000000000
EXPECTED_MD5="e206c3450ac99950df65bf70ef61a12d"
EXPECTED_SHA1="2996e86fb978f93cca8f566cc56998923e7fe581"
EXPECTED_SHA256="159b85351e5f76e60cbe32e04c677847a9ecba3adc79addab6f4c6c7aa3744bc"

usage() {
  cat <<EOF
Usage: $0 --bitstream payload_final_fullsha.bin [--decoder-zip blsmc_ac_decoder.zip]

This is the ranking check. It is not CI. Expect H100-class GPU and a long
decode (hours to ~18 days).
EOF
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bitstream) BITSTREAM="$2"; shift 2 ;;
    --decoder-zip) DECODER_ZIP="$2"; shift 2 ;;
    --out) ENWIK9_OUT="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) fail "unknown arg: $1" ;;
  esac
done

[[ -n "${BITSTREAM}" ]] || fail "--bitstream is required (Release asset or encode output)"

DEC_ARGS=(--bitstream "${BITSTREAM}" --out "${ENWIK9_OUT}")
[[ -n "${DECODER_ZIP}" ]] && DEC_ARGS+=(--decoder-zip "${DECODER_ZIP}")

echo "=== full enwik9 reconstruct (not CI) ==="
"${ROOT}/scripts/decode.sh" "${DEC_ARGS[@]}"

size=$(wc -c < "${ENWIK9_OUT}" | tr -d ' ')
[[ "${size}" == "${EXPECTED_SIZE}" ]] || fail "size ${size} != ${EXPECTED_SIZE}"

hash_md5() {
  if command -v md5 >/dev/null 2>&1; then md5 -q "$1"
  elif command -v md5sum >/dev/null 2>&1; then md5sum "$1" | awk '{print $1}'
  fi
}
hash_sha1() {
  if command -v shasum >/dev/null 2>&1; then shasum -a 1 "$1" | awk '{print $1}'
  elif command -v sha1sum >/dev/null 2>&1; then sha1sum "$1" | awk '{print $1}'
  fi
}
hash_sha256() {
  if command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | awk '{print $1}'
  elif command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  fi
}

got_md5=$(hash_md5 "${ENWIK9_OUT}")
got_sha1=$(hash_sha1 "${ENWIK9_OUT}")
got_sha256=$(hash_sha256 "${ENWIK9_OUT}")

[[ "${got_md5}" == "${EXPECTED_MD5}" ]] || fail "MD5 ${got_md5} != ${EXPECTED_MD5}"
[[ "${got_sha1}" == "${EXPECTED_SHA1}" ]] || fail "SHA-1 ${got_sha1} != ${EXPECTED_SHA1}"
[[ "${got_sha256}" == "${EXPECTED_SHA256}" ]] || fail "SHA-256 ${got_sha256} != ${EXPECTED_SHA256}"

echo "OK enwik9 reconstruct"
echo "  size    ${size}"
echo "  md5     ${got_md5}"
echo "  sha1    ${got_sha1}"
echo "  sha256  ${got_sha256}"

if [[ -n "${DECODER_ZIP}" ]]; then
  "${ROOT}/scripts/measure_s.sh" --bitstream "${BITSTREAM}" --decoder-zip "${DECODER_ZIP}"
fi
