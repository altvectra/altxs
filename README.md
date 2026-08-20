# altxs

Lossless compressor for the [Large Text Compression Benchmark](https://mattmahoney.net/dc/text.html) (enwik9: the first 10^9 bytes of the 3 Mar 2006 English Wikipedia XML dump).

This repository is the public encode/decode implementation. Ranking uses **Total S**:

```text
S = |compressed enwik9| + |zip of the decompresser|
```

The bitstream and the decoder zip are **Release assets**, not git objects. `enwik9` itself is not in this repo.

## What it does

1. **Peel** Wikipedia markup, headers, and dictionary/word transforms into a denser byte stream (`payload_sim`).
2. **Predict** the next byte with a small Transformer (exclusive self-attention, 256-byte vocabulary).
3. **Arithmetic-code** those probabilities into a segment-framed bitstream.
4. **Adapt online** while encoding and decoding: at every 16 KiB boundary the model retrains on the already-agreed prefix (replenish + explorative modeling). Encode and decode follow the same lockstep so probabilities stay identical.
5. **Invert the peels** to reconstruct `enwik9` byte-for-byte.

The shipped model is a quantized student (~3.15 bits/weight). Dense weights are rebuilt at decode time from a deterministic seed plus a mixed-bit ΔW codec. No teacher checkpoint is required to decompress.

This is an `xd` entry: a **separate** decompresser plus compressed data, not a self-extracting archive.

## Reported operating point

Tag `ltcb-3.15bpw` (update these numbers when you cut a release):


| Piece                  | Bytes           |
| ---------------------- | --------------- |
| AC bitstream           | 93,154,708      |
| Decoder zip (`zip -9`) | 13,437,796      |
| **Total S**            | **106,592,504** |


Hardware: NVIDIA H100, bf16. Decode is GPU-bound and currently takes on the order of days for the full corpus. LTCB ranking is pending until a full reconstruct is verified.

## Official enwik9 checksums


|       |                                            |
| ----- | ------------------------------------------ |
| Size  | 1,000,000,000 bytes                        |
| MD5   | `e206c3450ac99950df65bf70ef61a12d`         |
| SHA-1 | `2996e86fb978f93cca8f566cc56998923e7fe581` |


## What this repo contains

- Encode and decode source (import closure only)
- Mixed-bit codec rebuild
- Peel inverter (`blsmc_prepare`) and `english.dic`
- Lockstep env (`DECODE.env`)
- Fast lockstep tests and a full-enwik9 verification script
- Decoder packaging (`zip -9`) and an S measurement script

Not included: teacher training, distillation lab, peel-discovery research, or the 1 GB dump.

## Quick start

**Lockstep (CI):**

```bash
./scripts/test_lockstep.sh
```

**Decode Release bitstream → enwik9** (GPU, long):

```bash
# 1. Download payload_final_fullsha.bin from the matching GitHub Release
# 2. Rebuild dense student from the shipped codec (see DECODE.md)
./scripts/decode.sh /path/to/payload_final_fullsha.bin /path/to/enwik9_out
```

**Full ranking check** (not CI):

```bash
./scripts/test_enwik9.sh
```

Details: [DECODE.md](DECODE.md), [ENCODE.md](ENCODE.md). Report a result with the Release zip, bitstream, machine, and measured S.

## Requirements

- Python 3, PyTorch with CUDA, NumPy, safetensors
- NVIDIA GPU (H100 class for the reported run)
- `COMPRESSION_DETERMINISTIC=strict` for encode/decode lockstep

CPU-only decode and legacy (non-incremental) bitstreams are not supported.

## License

See [LICENSE](LICENSE) and [NOTICE](NOTICE) (dictionary / UPX / third-party).