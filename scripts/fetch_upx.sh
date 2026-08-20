#!/usr/bin/env bash
# Procure UPX. Required: package_s.sh always UPX-packs bin/blsmc_prepare.
# Prefers an existing `upx` on PATH, else downloads a pinned release into
# vendor/upx/upx (gitignored).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${ROOT}/vendor/upx"
# Pinned official release (https://github.com/upx/upx/releases).
UPX_VER="${UPX_VER:-5.0.2}"

if command -v upx >/dev/null 2>&1; then
  echo "OK upx on PATH: $(command -v upx)  ($("$(command -v upx)" --version 2>/dev/null | head -1))"
  exit 0
fi
if [[ -x "${DEST}/upx" ]]; then
  echo "OK ${DEST}/upx  ($("${DEST}/upx" --version 2>/dev/null | head -1))"
  exit 0
fi

os="$(uname -s | tr '[:upper:]' '[:lower:]')"
arch="$(uname -m)"
case "${os}-${arch}" in
  linux-x86_64|linux-amd64)   asset="upx-${UPX_VER}-amd64_linux.tar.xz" ;;
  linux-aarch64|linux-arm64)  asset="upx-${UPX_VER}-arm64_linux.tar.xz" ;;
  darwin-x86_64)              asset="upx-${UPX_VER}-amd64_macos.tar.xz" ;;
  darwin-arm64)               asset="upx-${UPX_VER}-arm64_macos.tar.xz" ;;
  *)
    echo "error: no pinned UPX build for ${os}-${arch}." >&2
    echo "  install upx (brew install upx / apt install upx-ucl) and re-run." >&2
    exit 1
    ;;
esac

url="https://github.com/upx/upx/releases/download/v${UPX_VER}/${asset}"
STAGE="$(mktemp -d)"
trap 'rm -rf "${STAGE}"' EXIT
echo "fetching ${url}"
curl -fL --retry 3 --retry-delay 2 -o "${STAGE}/${asset}" "${url}"
mkdir -p "${DEST}"
tar -xJf "${STAGE}/${asset}" -C "${STAGE}"
found="$(find "${STAGE}" -type f -name upx -perm -111 | head -1 || true)"
[[ -n "${found}" ]] || { echo "error: upx binary not in ${asset}" >&2; exit 1; }
cp -p "${found}" "${DEST}/upx"
chmod +x "${DEST}/upx"
echo "OK ${DEST}/upx  ($("${DEST}/upx" --version 2>/dev/null | head -1))"
