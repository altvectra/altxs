# Encode: enwik9 → bitstream + decoder zip

Encode is required so others can reproduce the bitstream. Mahoney only needs the decoder zip + bitstream to rank.

Use the **same** `DECODE.env` as decode.

## 0. Procure deps

```bash
./scripts/setup.sh                    # peel vendors + build blsmc_prepare
WITH_PYTHON=1 ./scripts/setup.sh      # also uv sync --extra dev (needed for AC)
WITH_ENWIK9=1 ./scripts/setup.sh      # also official enwik9
```

Peel itself is C++ and does not need PyTorch. AC encode does.

## 1. Official enwik9

Do not git `enwik9`.

```bash
./scripts/fetch_enwik9.sh
# data/enwik9  →  1,000,000,000 B
# MD5  e206c3450ac99950df65bf70ef61a12d
# SHA-1  2996e86fb978f93cca8f566cc56998923e7fe581
```

## 2. Peel / preprocess → payload_sim

Full stack: [PEEL.md](PEEL.md). One command:

```bash
./scripts/peel.sh
```

This runs `blsmc_prepare encode` with `dict/english.dic` and `dict/new_article_order` (cmix-lex article table, vendored). Output:

- `data/enwik9.blsmc_full.m3v2.payload_sim` — scored product
- AC stream length **576,278,322** B (trailers stripped; see `xsa_ttt.data._byte_stream_len`)
- Trailer (M3 side + `BLSMETA1`) is **not** in the AC stream; `package_decoder.sh` copies it into the decoder zip

## 3. Rebuild the dense student (Init(seed) + mixed-bit ΔW)

Put the Release codec in `weights/` (`mixed_da_bpw*.safetensors` + matching `.json`). That file **is** the student product: a quantized residual on a clean `Init(seed)`.

```bash
export PYTHONPATH=src
python -m hyperflow_distillation.mixed_bit_delta decode \
  --codec weights/mixed_da_bpw3.15_upb1.8.safetensors \
  --out weights/student_dense.safetensors
```

There is no pack/calibrate command here. Do not look for a teacher checkpoint.

## 4. AC encode

```bash
set -a && . ./DECODE.env && set +a
export PYTHONPATH=src
python -m xsa_ttt.train --eval-only weights/student_dense.safetensors \
  --data data/enwik9.blsmc_full.m3v2.payload_sim \
  --ac --ac-bytes 0 --skip-tf \
  --out work/ac_encode --profile "${PROFILE}" --seed "${SEED}"
```

`--ac-bytes 0` means the full stream. Writes `work/ac_encode/payload_final_fullsha.bin`.

Or: `./scripts/encode.sh --bitstream-only`.

For a 4 MiB incremental encode + blind decode (the check reviewers should run, not the multi-day full verify):

```bash
./scripts/roundtrip_4mb.sh          # N=4,194,304; same DECODE.env
# ./scripts/encode.sh --bitstream-only --bytes 4194304
```

See [VALIDATE.md](VALIDATE.md). A 4 MiB bitstream is not a prefix of the ranking file (AC flushes at `N`).

## 5. Zip the decoder and compute S

```bash
./scripts/package_s.sh \
  --bitstream work/ac_encode/payload_final_fullsha.bin \
  --product data/enwik9.blsmc_full.m3v2.payload_sim
```

`package_s.sh` uses InfoZIP `zip -9` (what Mahoney uses if you do not supply a zip). It always UPX-packs `bin/blsmc_prepare` first (`./scripts/fetch_upx.sh` if `upx` is missing), then packs decode code, the mixed-bit ΔW codec, `english.dic`, the sidecar trailer, `DECODE.md`, `DECODE.env`, and `MANIFEST.txt`. It does **not** pack the AC bitstream (counted separately), tests, or anything used to *create* the ΔW.

It writes `work/blsmc_ac_decoder.zip` (enters S) and `work/blsmc_ac_decoder.S.txt` (accounting only, not in S). Same zip without the bitstream:

```bash
./scripts/package_s.sh --product data/enwik9.blsmc_full.m3v2.payload_sim
./scripts/measure_s.sh \
  --bitstream work/ac_encode/payload_final_fullsha.bin \
  --decoder-zip work/blsmc_ac_decoder.zip
```

**S = |payload_final_fullsha.bin| + |blsmc_ac_decoder.zip|**

Tagged `ltcb-3.15bpw`: 93,154,708 + 13,437,796 = **106,592,504**.
