# Validate incremental AC (4 MiB, not the multi-day full verify)

Three checks, in increasing cost. Reviewers should stop at **4 MiB**.

| Check | What it proves | GPU | Wall clock |
|---|---|---|---|
| `./scripts/test_lockstep.sh` | integer freq tables + encode ≡ decode on a short synthetic prefix | optional (CUDA tests skip) | minutes |
| `./scripts/roundtrip_4mb.sh` | production incremental path on a real `payload_sim` prefix | H100 required | hours (decode is the long half) |
| `./scripts/test_enwik9.sh` | full 1 GB reconstruct + official MD5 / SHA-1 / SHA-256 | H100 required | days (up to ~18) — **not CI** |

Both AC sides must load the same `DECODE.env`. Incremental is on (`XSA_AC_INCREMENTAL=1`). Encode uses the W=64 mega (`XSA_AC_WINDOW=64`, `XSA_AC_MEGA_ENCODE=1`). Decode steps one newly known token on the W=1 mega (`XSA_AC_DECODE_ROW=1`). Do not set `XSA_AC_DECODE_UNROLL` or `XSA_AC_NO_GRAPH=1`. `XSA_AC_ENC_ATTN_GROUP` must stay `1`.

`--ac-bytes N` is the symbol count (bytes of the stripped `payload_sim` stream). `0` on encode means the full 576,278,322-symbol stream. Decode always needs `N > 0`.

## 0. Procure

```bash
./scripts/setup.sh                       # peel vendors + build blsmc_prepare
WITH_PYTHON=1 ./scripts/setup_python.sh  # uv sync --extra dev
WITH_ENWIK9=1 WITH_PEEL=1 ./scripts/setup.sh
```

Put the Release mixed-bit ΔW in `weights/` (`mixed_da_bpw*.safetensors` + matching `.json`).

You need `data/enwik9.blsmc_full.m3v2.payload_sim` (from peel). The 4 MiB check compares against the **prefix of that product**, not reconstructed `enwik9`. Peel inverse is only for the full 1 GB verify.

## 1. Fast lockstep (CI)

```bash
./scripts/test_lockstep.sh
```

Runs `tests/test_incremental_ac_window.py`. This is not the production window sizes; it only proves the integer AC + incremental KV path is self-consistent.

## 2. 4 MiB production roundtrip (the check to run)

4 MiB = **4,194,304** symbols. Same student rebuild, same `DECODE.env`, same `python -m xsa_ttt.train` entry as a full encode/decode.

```bash
./scripts/roundtrip_4mb.sh
# optional: --bytes 1048576   # ~1 MiB, shorter
#           --out work/roundtrip_4mb
```

That is:

1. `python -m hyperflow_distillation.mixed_bit_delta decode` — `student = Init(seed) + ΔW`
2. Incremental encode of the first `N` bytes of `payload_sim` → `payload_final_fullsha.bin`
3. Blind incremental decode of that bitstream (`--decode-payload`, no source) → `decoded_stream.bin`
4. `cmp` against `head -c N` of the product

Equivalent raw commands:

```bash
set -a && . ./DECODE.env && set +a
export PYTHONPATH=src
export COMPRESSION_DETERMINISTIC=strict
N=4194304
PRODUCT=data/enwik9.blsmc_full.m3v2.payload_sim

python -m hyperflow_distillation.mixed_bit_delta decode \
  --codec weights/mixed_da_bpw3.15_upb1.8.safetensors \
  --out weights/student_dense.safetensors

python -m xsa_ttt.train --eval-only weights/student_dense.safetensors \
  --data "${PRODUCT}" \
  --ac --ac-bytes "${N}" --skip-tf \
  --out work/roundtrip_4mb/enc --profile "${PROFILE}" --seed "${SEED}"

python -m xsa_ttt.train --eval-only weights/student_dense.safetensors \
  --decode-payload work/roundtrip_4mb/enc/payload_final_fullsha.bin \
  --ac-bytes "${N}" \
  --decoded-path work/roundtrip_4mb/decoded_stream.bin \
  --out work/roundtrip_4mb/dec --profile "${PROFILE}" --seed "${SEED}"

head -c "${N}" "${PRODUCT}" | cmp - work/roundtrip_4mb/decoded_stream.bin
```

Or via the wrappers: `./scripts/encode.sh --bitstream-only --bytes 4194304` then decode with `--bytes 4194304` (do **not** peel-inverse a 4 MiB stream — it is not a full product).

**Timing (H100, `DECODE.env`).** Encode of 4 MiB is the cheap half (W=64 mega). Decode pays one fused window step per accepted token, so 4 MiB is **hours**. A W=1 encode of the same prefix was ~9 h at ~124 sym/s; production decode is in that ballpark. `--bytes 1048576` is the shorter variant. This is still not a several-day job.

A 4 MiB-only bitstream is **not** a prefix of the ranking 93,154,708-byte stream (the AC coder flushes at the end of the chosen `N`). It only proves the same model + env + incremental path.

To decode the first `N` symbols from the **published** full bitstream instead:

```bash
./scripts/roundtrip_4mb.sh \
  --bitstream /path/to/payload_final_fullsha.bin \
  --bytes 4194304
```

That skips encode and still `cmp`s against the product prefix. The model trajectory on the first `N` symbols matches a full encode; the bitstream is the ranking one.

## 3. Full 1 GB (skip unless ranking)

```bash
./scripts/test_enwik9.sh \
  --bitstream /path/to/payload_final_fullsha.bin \
  --decoder-zip /path/to/blsmc_ac_decoder.zip
```

This is `decode.sh` (full `AC_N_SYMBOLS=576278322`) + official checksums. Expect multi-day wall clock. See [DECODE.md](DECODE.md).
