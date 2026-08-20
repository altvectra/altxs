#!/usr/bin/env bash
# Official enwik9 → payload_sim (the byte stream AC encode consumes).
set -euo pipefail
# shellcheck source=common.sh
. "$(cd "$(dirname "$0")" && pwd)/common.sh"

exec "${ROOT}/scripts/encode.sh" --peel-only "$@"
