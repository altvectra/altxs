#!/usr/bin/env bash
# Download official enwik9 (10^9 B) into data/ and verify LTCB checksums.
# Do not git this file.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${ROOT}/data"
OUT="${DATA}/enwik9"
ZIP="${DATA}/enwik9.zip"
EXPECTED_SIZE=1000000000
EXPECTED_MD5="e206c3450ac99950df65bf70ef61a12d"
EXPECTED_SHA1="2996e86fb978f93cca8f566cc56998923e7fe581"
EXPECTED_SHA256="159b85351e5f76e60cbe32e04c677847a9ecba3adc79addab6f4c6c7aa3744bc"

URLS=(
  "http://mattmahoney.net/dc/enwik9.zip"
  "http://cs.fit.edu/~mmahoney/compression/enwik9.zip"
)

mkdir -p "${DATA}"

if [[ -f "${OUT}" ]]; then
  size=$(wc -c < "${OUT}" | tr -d ' ')
  if [[ "${size}" == "${EXPECTED_SIZE}" ]]; then
    echo "enwik9 already present (${OUT}, ${size} B)"
  else
    echo "removing wrong-sized enwik9 (${size} B)"
    rm -f "${OUT}"
  fi
fi

if [[ ! -f "${OUT}" ]]; then
  if [[ ! -f "${ZIP}" ]]; then
    ok=0
    for url in "${URLS[@]}"; do
      echo "fetching ${url} (~310–322 MB zip; slow)"
      if curl -fL --retry 3 --retry-delay 2 -o "${ZIP}" "${url}"; then
        ok=1
        break
      fi
      rm -f "${ZIP}"
    done
    if [[ "${ok}" -ne 1 ]]; then
      echo "failed to download enwik9.zip" >&2
      exit 1
    fi
  fi
  echo "unzipping ${ZIP}"
  unzip -o -d "${DATA}" "${ZIP}"
  if [[ ! -f "${OUT}" ]]; then
    found="$(find "${DATA}" -type f -name enwik9 | head -1 || true)"
    if [[ -n "${found}" && "${found}" != "${OUT}" ]]; then
      mv "${found}" "${OUT}"
    fi
  fi
fi

size=$(wc -c < "${OUT}" | tr -d ' ')
if [[ "${size}" != "${EXPECTED_SIZE}" ]]; then
  echo "size mismatch: got ${size}, expected ${EXPECTED_SIZE}" >&2
  exit 1
fi

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

got_md5=$(hash_md5 "${OUT}")
got_sha1=$(hash_sha1 "${OUT}")
got_sha256=$(hash_sha256 "${OUT}")

[[ -z "${got_md5}" || "${got_md5}" == "${EXPECTED_MD5}" ]] || {
  echo "MD5 mismatch: got ${got_md5}, expected ${EXPECTED_MD5}" >&2; exit 1
}
[[ -z "${got_sha1}" || "${got_sha1}" == "${EXPECTED_SHA1}" ]] || {
  echo "SHA-1 mismatch: got ${got_sha1}, expected ${EXPECTED_SHA1}" >&2; exit 1
}
[[ -z "${got_sha256}" || "${got_sha256}" == "${EXPECTED_SHA256}" ]] || {
  echo "SHA-256 mismatch: got ${got_sha256}, expected ${EXPECTED_SHA256}" >&2; exit 1
}

echo "OK ${OUT}"
echo "  size    ${size}"
echo "  md5     ${got_md5}"
echo "  sha1    ${got_sha1}"
echo "  sha256  ${got_sha256}"
