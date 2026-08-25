#!/usr/bin/env bash
# Assemble the LTCB decoder zip and print Total S.
#
#   S = |compressed enwik9| + |zip -9 of everything needed to decompress|
#
# This zip is the decompresser. The AC bitstream is counted separately and
# is never packed inside. Mahoney uses InfoZIP `zip -9` if you do not
# supply a zip — we do the same.
#
# In the zip (enters S):
#   bin/blsmc_prepare                 UPX-packed (required)
#   dict/english.dic
#   sidecars/payload_sim.trailer.bin     M3 + BLSMETA1 (not in the AC stream)
#   weights/mixed_da_bpw*.safetensors    mixed-bit ΔW (+ .json)
#   code/                                decode import closure
#   DECODE.env  DECODE.md  MANIFEST.txt
#
# Next to the zip (not in S):
#   *.S.txt     bitstream + zip + Total S
#
# Usage:
#   ./scripts/package_s.sh
#   ./scripts/package_s.sh --bitstream work/ac_encode/payload_final_fullsha.bin
#   ./scripts/package_s.sh --product data/enwik9.blsmc_full.m3v2.payload_sim \
#       --weights-dir weights --out work/blsmc_ac_decoder.zip
set -euo pipefail
# shellcheck source=common.sh
. "$(cd "$(dirname "$0")" && pwd)/common.sh"

BITSTREAM="${BITSTREAM:-}"
PRODUCT="${PRODUCT:-${ROOT}/data/enwik9.blsmc_full.m3v2.payload_sim}"
SIDECAR="${SIDECAR:-${ROOT}/sidecars/payload_sim.trailer.bin}"
WEIGHTS_DIR="${WEIGHTS_DIR:-${ROOT}/weights}"
BLSMC_BIN="${BLSMC_BIN:-${ROOT}/blsmc/prepare/blsmc_prepare}"
ENGLISH_DIC="${ENGLISH_DIC:-${ROOT}/dict/english.dic}"
OUT_ZIP="${OUT_ZIP:-${ROOT}/work/blsmc_ac_decoder.zip}"
EXPECTED_S="${EXPECTED_S:-106924811}"
EXPECTED_BITSTREAM="${EXPECTED_BITSTREAM:-93434410}"
EXPECTED_ZIP="${EXPECTED_ZIP:-13490401}"

usage() {
  cat <<EOF
Usage: $0 [options]

  --bitstream PATH     AC payload (counted in S, not packed)
  --product PATH       payload_sim product (trailer is stripped from this)
  --sidecar PATH       pre-cut trailer (used if --product is missing)
  --weights-dir DIR    mixed-bit ΔW codec directory
  --blsmc PATH         peel binary
  --dict PATH          english.dic
  --out PATH           decoder zip (default work/blsmc_ac_decoder.zip)

bin/blsmc_prepare is always UPX-packed before zip -9 (required for S).
Missing upx: ./scripts/fetch_upx.sh

Env: BITSTREAM PRODUCT SIDECAR WEIGHTS_DIR BLSMC_BIN ENGLISH_DIC OUT_ZIP UPX_BIN
EOF
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bitstream) BITSTREAM="$2"; shift 2 ;;
    --product) PRODUCT="$2"; shift 2 ;;
    --sidecar) SIDECAR="$2"; shift 2 ;;
    --weights-dir) WEIGHTS_DIR="$2"; shift 2 ;;
    --blsmc) BLSMC_BIN="$2"; shift 2 ;;
    --dict) ENGLISH_DIC="$2"; shift 2 ;;
    --out) OUT_ZIP="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) fail "unknown arg: $1" ;;
  esac
done

file_size() { stat -c%s "$1" 2>/dev/null || stat -f%z "$1"; }

sha256_list() {
  if command -v sha256sum >/dev/null 2>&1; then
    xargs -0 sha256sum
  else
    xargs -0 shasum -a 256
  fi
}

[[ -f "${ENGLISH_DIC}" ]] || fail "missing ${ENGLISH_DIC}"
if [[ ! -x "${BLSMC_BIN}" ]]; then
  echo "building blsmc_prepare"
  make -C "${ROOT}/blsmc/prepare"
fi
[[ -x "${BLSMC_BIN}" ]] || fail "missing blsmc_prepare: ${BLSMC_BIN}"

CODEC="$(find_codec "${WEIGHTS_DIR}")"
CODEC_JSON="${CODEC%.safetensors}.json"
[[ -f "${CODEC_JSON}" ]] || fail "missing codec sidecar json: ${CODEC_JSON}"
ANCHOR_MASKS="${CODEC%.safetensors}_anchor.safetensors"

STAGE="$(mktemp -d)"
PKG="${STAGE}/blsmc_ac_decoder"
trap 'rm -rf "${STAGE}"' EXIT
mkdir -p "${PKG}"/{bin,dict,sidecars,weights,code}

echo "[1/6] peel binary + WRT dictionary"
UPX_CMD=""
if [[ -n "${UPX_BIN:-}" && -x "${UPX_BIN}" ]]; then
  UPX_CMD="${UPX_BIN}"
elif [[ -x "${ROOT}/vendor/upx/upx" ]]; then
  UPX_CMD="${ROOT}/vendor/upx/upx"
elif command -v upx >/dev/null 2>&1; then
  UPX_CMD="$(command -v upx)"
else
  fail "upx is required for Total S (bin/blsmc_prepare is always UPX-packed). Run ./scripts/fetch_upx.sh"
fi
cp -p "${BLSMC_BIN}" "${PKG}/bin/blsmc_prepare"
if "${UPX_CMD}" -t "${PKG}/bin/blsmc_prepare" >/dev/null 2>&1; then
  echo "  already UPX-packed  ${UPX_CMD}"
else
  "${UPX_CMD}" -9 "${PKG}/bin/blsmc_prepare" >/dev/null
  echo "  UPX -9  ${UPX_CMD}  → bin/blsmc_prepare"
fi
cp -p "${ENGLISH_DIC}" "${PKG}/dict/english.dic"

echo "[2/6] sidecar trailer (not in the AC stream)"
mkdir -p "${ROOT}/sidecars"
if [[ -f "${PRODUCT}" ]]; then
  STREAM_N="$("${PY}" - "${PRODUCT}" <<'EOF'
# Stdlib only — do not import xsa_ttt (that pulls numpy).
import sys
from pathlib import Path

path = Path(sys.argv[1])
end = path.stat().st_size

def strip_footer(end_n: int, foot: bytes) -> int:
    need = len(foot) + 8
    if end_n < need:
        return end_n
    with open(path, "rb") as f:
        f.seek(end_n - need)
        tail = f.read(need)
    if tail[: len(foot)] != foot:
        return end_n
    blob_len = int.from_bytes(tail[len(foot) :], "little")
    stream_n = end_n - need - blob_len
    if stream_n < 0 or blob_len > end_n - need:
        return end_n
    return stream_n

end = strip_footer(end, b"BLSMETA1")
end = strip_footer(end, b"M3SIDFTR")
print(end)
EOF
)"
  FILE_N="$(file_size "${PRODUCT}")"
  TRAILER_N=$(( FILE_N - STREAM_N ))
  (( TRAILER_N > 0 )) || fail "no trailer on ${PRODUCT} (stream=${STREAM_N} file=${FILE_N})"
  tail -c "${TRAILER_N}" "${PRODUCT}" > "${PKG}/sidecars/payload_sim.trailer.bin"
  cp -p "${PKG}/sidecars/payload_sim.trailer.bin" "${ROOT}/sidecars/payload_sim.trailer.bin"
  echo "  from product  stream=${STREAM_N}  trailer=${TRAILER_N}  file=${FILE_N}"
elif [[ -f "${SIDECAR}" ]]; then
  cp -p "${SIDECAR}" "${PKG}/sidecars/payload_sim.trailer.bin"
  echo "  from sidecar  ${SIDECAR}  $(file_size "${SIDECAR}") B"
else
  fail "need --product (payload_sim) or --sidecar (payload_sim.trailer.bin)"
fi

echo "[3/6] mixed-bit ΔW codec (the published student product)"
cp -p "${CODEC}" "${CODEC_JSON}" "${PKG}/weights/"
[[ -f "${ANCHOR_MASKS}" ]] && cp -p "${ANCHOR_MASKS}" "${PKG}/weights/"
echo "  $(basename "${CODEC}")"

echo "[4/6] decode import closure"
mkdir -p "${PKG}/code/xsa_ttt" "${PKG}/code/model" "${PKG}/code/hyperflow_distillation"
XSA_TTT_FILES=(
  __init__.py __main__.py
  train.py compress.py incremental.py
  model.py fused_step.py persistent_step.py
  mega_step.py mega_encode.py split_attn.py ac_gemv.py
  gpu_ac.py row_commit.py train_attn.py
  config.py data.py device.py
  checkpoint.py ttt_lora.py chart.py
)
for f in "${XSA_TTT_FILES[@]}"; do
  [[ -f "${ROOT}/src/xsa_ttt/${f}" ]] || fail "missing src/xsa_ttt/${f}"
  cp -p "${ROOT}/src/xsa_ttt/${f}" "${PKG}/code/xsa_ttt/"
done
for f in __init__.py mixed_bit_delta.py weight_space.py train_hyperflow.py; do
  cp -p "${ROOT}/src/hyperflow_distillation/${f}" "${PKG}/code/hyperflow_distillation/"
done
cp -p "${ROOT}/src/model/arithmetic_coder_lm.py" "${PKG}/code/model/"
cp -p "${ROOT}/src/model/deterministic_mode.py" "${PKG}/code/model/"
find "${PKG}/code" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true

echo "[5/6] DECODE.env + DECODE.md"
cp -p "${ROOT}/DECODE.env" "${PKG}/DECODE.env"
cp -p "${ROOT}/DECODE.md" "${PKG}/DECODE.md"

echo "[6/6] MANIFEST + zip -9"
( cd "${STAGE}" && find blsmc_ac_decoder -type f ! -name MANIFEST.txt -print0 \
    | sort -z | sha256_list ) > "${PKG}/MANIFEST.txt"
{
  echo
  echo "# sizes"
  ( cd "${STAGE}" && find blsmc_ac_decoder -type f ! -name MANIFEST.txt -print0 \
      | sort -z | xargs -0 wc -c )
} >> "${PKG}/MANIFEST.txt"
cp -p "${PKG}/MANIFEST.txt" "${ROOT}/MANIFEST.txt"

mkdir -p "$(dirname "${OUT_ZIP}")"
rm -f "${OUT_ZIP}"
command -v zip >/dev/null 2>&1 || fail "need InfoZIP zip (Mahoney uses zip -9)"
( cd "${STAGE}" && zip -9 -rq "${OUT_ZIP}" blsmc_ac_decoder )

S_TXT="${OUT_ZIP%.zip}.S.txt"
"${PY}" - "${OUT_ZIP}" "${BITSTREAM:-}" "${EXPECTED_BITSTREAM}" "${EXPECTED_ZIP}" "${EXPECTED_S}" "${S_TXT}" <<'EOF'
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

zpath = Path(sys.argv[1])
bpath = Path(sys.argv[2]) if sys.argv[2] else None
exp_b, exp_z, exp_s = (int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
s_txt = Path(sys.argv[6])
zip_n = zpath.stat().st_size
bit_n = bpath.stat().st_size if bpath and bpath.is_file() else None


def group_of(name: str) -> str:
    parts = name.split("/")
    if len(parts) >= 2 and parts[1] in ("bin", "dict", "sidecars", "weights", "code"):
        return parts[1]
    return "(root)"

by_g: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
with zipfile.ZipFile(zpath) as zf:
    for info in zf.infolist():
        if info.is_dir():
            continue
        rel = info.filename
        if rel.startswith("blsmc_ac_decoder/"):
            rel = rel[len("blsmc_ac_decoder/") :]
        by_g[group_of(info.filename)].append(
            (rel, info.file_size, info.compress_size)
        )

order = ["bin", "dict", "sidecars", "weights", "code", "(root)"]
lines: list[str] = []

def emit(s: str = "") -> None:
    lines.append(s)
    print(s)

emit("LTCB Total S  =  |compressed enwik9|  +  |zip -9 decompresser|")
emit("This zip is the decompresser. The AC bitstream is counted separately.")
emit()
emit(f"{'group':<10} {'file':<62} {'raw':>12} {'in zip':>12}")
emit("-" * 98)
tot_raw = tot_pack = 0
for g in order:
    items = by_g.get(g, [])
    if not items:
        continue
    g_raw = g_pack = 0
    for rel, raw, packed in sorted(items, key=lambda t: -t[2]):
        emit(f"{g:<10} {rel:<62} {raw:>12,} {packed:>12,}")
        g_raw += raw
        g_pack += packed
    emit(f"{'':10} {('— ' + g + ' subtotal'):<62} {g_raw:>12,} {g_pack:>12,}")
    emit()
    tot_raw += g_raw
    tot_pack += g_pack
overhead = zip_n - tot_pack
emit("-" * 98)
emit(f"{'':10} {'sum of members':<62} {tot_raw:>12,} {tot_pack:>12,}")
emit(f"{'':10} {'zip central-dir / local-hdr overhead':<62} {'':>12} {overhead:>12,}")
emit(f"{'S zip':<10} {str(zpath):<62} {'':>12} {zip_n:>12,}")
emit()
nfiles = sum(len(v) for v in by_g.values())
emit(
    f"files={nfiles}  raw={tot_raw:,}  packed members={tot_pack:,}  "
    f"zip={zip_n:,}  ({100.0 * tot_pack / tot_raw:.1f}% of raw members)"
)
emit()
if bit_n is not None:
    s = bit_n + zip_n
    emit(f"bitstream  {bit_n:>12,}   {bpath}")
    emit(f"decoder    {zip_n:>12,}   {zpath}")
    emit(f"S          {s:>12,}   = |bitstream| + |zip -9|")
    if bit_n == exp_b and zip_n == exp_z:
        emit(f"matches tagged ltcb-3.15bpw (S={exp_s:,})")
    elif s == exp_s:
        emit("S matches tagged ltcb-3.15bpw")
    else:
        emit(
            f"note: tagged ltcb-3.15bpw is bitstream={exp_b:,} "
            f"zip={exp_z:,} S={exp_s:,}"
        )
else:
    emit(f"S = |AC bitstream| + {zip_n:,}")
    emit("pass --bitstream payload_final_fullsha.bin to close the sum")

s_txt.write_text("\n".join(lines) + "\n", encoding="utf8")
print(f"\nwrote {s_txt}")
EOF

echo
echo "decoder zip  ${OUT_ZIP}  ($(file_size "${OUT_ZIP}") B)"
echo "accounting   ${S_TXT}  (not in S)"
if [[ -n "${BITSTREAM}" && -f "${BITSTREAM}" ]]; then
  echo "bitstream    ${BITSTREAM}  ($(file_size "${BITSTREAM}") B)  — not in the zip"
fi
