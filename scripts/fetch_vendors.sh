#!/usr/bin/env bash
# Verify (or refresh) the cmix-lex peel subset used to build payload_sim.
#
# The repo already vendors that subset under vendor/cmix-lex plus
# dict/english.dic and dict/new_article_order. This script:
#   - checks those files are present
#   - with REFRESH=1, re-copies them from the pinned upstream commit
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENDOR="${ROOT}/vendor/cmix-lex"
PINNED="${CMIX_LEX_COMMIT:-370e698f7ea62168cc64326ff97950c3dc212691}"
URL="https://github.com/blahem/cmix-lex.git"

need=(
  "${VENDOR}/LICENSE"
  "${VENDOR}/src/readalike_prepr/misc.h"
  "${VENDOR}/src/readalike_prepr/article_reorder.h"
  "${VENDOR}/src/readalike_prepr/phda9_preprocess.h"
  "${VENDOR}/src/preprocess/preprocessor.cpp"
  "${VENDOR}/src/preprocess/preprocessor.h"
  "${VENDOR}/src/preprocess/dictionary.cpp"
  "${VENDOR}/src/preprocess/dictionary.h"
  "${VENDOR}/src/r1_reorder_transform.cpp"
  "${VENDOR}/src/r1_reorder_transform.h"
  "${VENDOR}/src/ds/emhash_map.hpp"
  "${ROOT}/dict/english.dic"
  "${ROOT}/dict/new_article_order"
)

copy_from_clone() {
  local src="$1"
  mkdir -p "${VENDOR}/src/readalike_prepr/data" \
           "${VENDOR}/src/preprocess" \
           "${VENDOR}/src/ds" \
           "${ROOT}/dict"
  cp -p "${src}/LICENSE" "${VENDOR}/LICENSE"
  cp -p "${src}/src/readalike_prepr/misc.h" "${VENDOR}/src/readalike_prepr/"
  cp -p "${src}/src/readalike_prepr/article_reorder.h" "${VENDOR}/src/readalike_prepr/"
  cp -p "${src}/src/readalike_prepr/phda9_preprocess.h" "${VENDOR}/src/readalike_prepr/"
  cp -p "${src}/src/preprocess/"*.{cpp,h} "${VENDOR}/src/preprocess/"
  cp -p "${src}/src/r1_reorder_transform.cpp" "${VENDOR}/src/"
  cp -p "${src}/src/r1_reorder_transform.h" "${VENDOR}/src/"
  cp -p "${src}/src/ds/"*.h "${src}/src/ds/"*.hpp "${VENDOR}/src/ds/"
  cp -p "${src}/src/readalike_prepr/data/new_article_order" \
    "${VENDOR}/src/readalike_prepr/data/new_article_order"
  cp -p "${src}/dictionary/english.dic" "${ROOT}/dict/english.dic"
  cp -p "${src}/src/readalike_prepr/data/new_article_order" \
    "${ROOT}/dict/new_article_order"
  git -C "${src}" rev-parse HEAD > "${VENDOR}/SOURCE.commit"
  echo "refreshed peel subset from $(cat "${VENDOR}/SOURCE.commit")"
}

missing=0
for f in "${need[@]}"; do
  if [[ ! -f "${f}" ]]; then
    echo "missing ${f}"
    missing=1
  fi
done

if [[ "${REFRESH:-0}" == "1" || "${missing}" == "1" ]]; then
  STAGE="$(mktemp -d)"
  trap 'rm -rf "${STAGE}"' EXIT
  echo "cloning ${URL} @ ${PINNED}"
  git clone --depth 1 "${URL}" "${STAGE}/cmix-lex"
  git -C "${STAGE}/cmix-lex" fetch --depth 1 origin "${PINNED}"
  git -C "${STAGE}/cmix-lex" checkout "${PINNED}"
  copy_from_clone "${STAGE}/cmix-lex"
  missing=0
  for f in "${need[@]}"; do
    [[ -f "${f}" ]] || { echo "error: still missing ${f}" >&2; exit 1; }
  done
fi

echo "OK peel vendors"
echo "  dict/english.dic         $(wc -c < "${ROOT}/dict/english.dic" | tr -d ' ') B"
echo "  dict/new_article_order   $(wc -c < "${ROOT}/dict/new_article_order" | tr -d ' ') B"
echo "  vendor/cmix-lex          pinned ${PINNED}"
