# altxs

Public compressor for the [Matt Mahoney Large Text Compression Benchmark](https://mattmahoney.net/dc/text.html#notes).

LTCB ranks **Total S = |compressed enwik9| + |zip of the decompresser|**. This git tree is how the result is published and verified. It is not the same object as Total S.

| Tag | Algorithm | Kind | Bitstream | Decoder zip | **S** |
|---|---|---|---:|---:|---:|
| `ltcb-3.15bpw` | Transformer + AC + dict/peel | `xd` (source/binary + separate bitstream) | 93,154,708 | 13,437,796 | **106,592,504** |

Ranking stays **pending** until a stranger reconstructs `enwik9` byte-identical.

## Official enwik9 checksums

| File | Size | MD5 | SHA-1 |
|---|---:|---|---|
| enwik9 | 1,000,000,000 | `e206c3450ac99950df65bf70ef61a12d` | `2996e86fb978f93cca8f566cc56998923e7fe581` |

Published SHA-256 of reconstructed `enwik9`:

```
159b85351e5f76e60cbe32e04c677847a9ecba3adc79addab6f4c6c7aa3744bc
```

## Hardware (decode)

| | |
|---|---|
| GPU | NVIDIA H100 (80 GB). CUDA required. CPU-only is **not** supported. |
| VRAM | ~80 GB class (profile `large`, ~32M params + KV + fused window) |
| RAM / disk | A few GiB host RAM; ~2–3 GiB scratch for peel inverse |
| Encode wall clock | ~14 h for the full 576,278,322-symbol AC on one H100 |
| Decode wall clock | Encode pays one window step per W=64 symbols; decode pays one step per accepted token. Multi-day (up to ~18 days) is still a valid LTCB result if the output is byte-identical. **This run is not CI.** |

Determinism: `COMPRESSION_DETERMINISTIC=strict` (see `DECODE.env`).

## Student weights (what is public)

The dense student is **not** a dumped checkpoint. The Release ships a mixed-bit **ΔW** codec (`mixed_da_bpw*.safetensors` + `.json`). Anyone rebuilds the same student with:

```text
student = Init(seed from codec metadata) + dequantized ΔW
```

That codec is the product. How it was fit is not part of this repo.

**Not published:** a teacher model, calibration / distillation recipes, or any other path that produces the ΔW.

## Not supported

- Legacy non-incremental / chunked-TF bitstreams
- CPU-only decode
- Reproducing the student from anything except the shipped mixed-bit ΔW + `Init(seed)`
- Hutter Prize `archive9` limits (≤10 GiB RAM, CPU, Geekbench time). This is an LTCB GPU entry, same class as nncp / jax-compress.

## One-command decode

Release assets for tag `ltcb-3.15bpw`: `payload_final_fullsha.bin` + `blsmc_ac_decoder.zip`.

```bash
./scripts/setup.sh                  # peel vendors + build blsmc_prepare
./scripts/decode.sh \
  --bitstream /path/to/payload_final_fullsha.bin \
  --decoder-zip /path/to/blsmc_ac_decoder.zip \
  --out work/enwik9
```

Or from a checkout that already has weights + trailer (after `package_decoder.sh` or a unpacked zip):

```bash
./scripts/decode.sh --bitstream /path/to/payload_final_fullsha.bin --out work/enwik9
```

Details: [DECODE.md](DECODE.md). Payload (`payload_sim`): [PEEL.md](PEEL.md). Encode / S: [ENCODE.md](ENCODE.md). Incremental AC check (4 MiB, not the multi-day full verify): [VALIDATE.md](VALIDATE.md).

```bash
./scripts/setup.sh                         # vendors + peel binary
WITH_ENWIK9=1 WITH_PEEL=1 ./scripts/setup.sh   # also fetch enwik9 and emit payload_sim
WITH_PYTHON=1 ./scripts/setup_python.sh    # uv sync --extra dev (.venv)
```

## Tests

```bash
./scripts/test_lockstep.sh    # fast, CI-able: encode ≡ decode on a short prefix
./scripts/roundtrip_4mb.sh    # production incremental path, 4 MiB prefix (GPU, hours)
./scripts/test_enwik9.sh      # full 1 GB reconstruct + official checksums (GPU, days — skip)
./scripts/package_s.sh --bitstream payload_final_fullsha.bin   # zip -9 + Total S
./scripts/measure_s.sh --bitstream payload_final_fullsha.bin --decoder-zip work/blsmc_ac_decoder.zip
```

See [VALIDATE.md](VALIDATE.md). Reviewers should run the 4 MiB roundtrip, not the full reconstruct.

## Report a result to LTCB

Send Matt Mahoney the bitstream, the `zip -9` decoder zip, this repo URL + tag, and the machine (GPU model, wall clock). Notes: https://mattmahoney.net/dc/text.html#notes

## Layout

```
README.md  LICENSE  NOTICE  DECODE.md  ENCODE.md  PEEL.md  VALIDATE.md  DECODE.env
pyproject.toml uv.lock requirements.txt   pinned AC runtime (uv sync)
src/                      encode + decode Python import closure
blsmc/prepare/            peel (enwik9 ↔ payload_sim)
vendor/cmix-lex/          PHDA9 / WRT / reorder sources (GPL-3)
dict/english.dic          WRT dictionary (counts in S)
dict/new_article_order    M2 article table (needed to *make* payload_sim)
scripts/setup.sh          procure vendors, UPX, compiler build, optional enwik9/venv
scripts/peel.sh           enwik9 → payload_sim
scripts/package_s.sh      decoder zip + Total S = |bitstream| + |zip -9|
scripts/                 decode, encode, measure S, tests
tests/                   lockstep unit tests (not packed into S)
```
