#!/usr/bin/env bash
# Compatibility entry: assemble the decoder zip and Total S.
# Prefer ./scripts/package_s.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec "${ROOT}/scripts/package_s.sh" "$@"
