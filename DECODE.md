# Decode: bitstream → enwik9

Reconstructs official `enwik9` from the AC bitstream of the `payload_sim` byte stream. Vocab is V=256 raw bytes — no BPE step.

The bitstream is **not** in this git tree. Download it from the GitHub Release for the matching tag (`payload_final_fullsha.bin`).

## Dependencies

- Python 3.11–3.13, CUDA PyTorch, `safetensors`, `numpy`, `tqdm` (`uv sync --extra dev`; Triton on Linux)
- NVIDIA H100-class GPU (see README)
- `blsmc_prepare` (build with `./scripts/fetch_vendors.sh && make -C blsmc/prepare`)
- `dict/english.dic`
- Sidecar trailer (`sidecars/payload_sim.trailer.bin` inside the decoder zip)

`COMPRESSION_DETERMINISTIC=strict` and the rest of `DECODE.env` must match encode.

## 1. Rebuild the dense student from the shipped ΔW

`student = Init(seed from codec metadata) + dequantized ΔW`. The Release `mixed_da_bpw*.safetensors` is that residual. Seed and arch are in the sidecar `.json`.

```bash
export PYTHONPATH=src
python -m hyperflow_distillation.mixed_bit_delta decode \
  --codec weights/mixed_da_bpw3.15_upb1.8.safetensors \
  --out weights/student_dense.safetensors
```

## 2. AC decode → payload_sim stream (576,278,322 B)

The bitstream is segment-framed (35,174 × 16k segments, 8 B header each) and encoded with the shared incremental prob path (`XSA_AC_INCREMENTAL=1`). Encode conditions on the true tokens with the canonical W=64 mega (split-8 attn + 3-stage FFN). Decode (`XSA_AC_DECODE_ROW=1`) steps one newly known token on the W=1 mega. Leave `XSA_AC_DECODE_UNROLL` unset. Decode is blind — no source product needed. Replenish/XM retrain replays on the decoded prefix.

```bash
set -a && . ./DECODE.env && set +a
export PYTHONPATH=src
python -m xsa_ttt.train --eval-only weights/student_dense.safetensors \
  --decode-payload /path/to/payload_final_fullsha.bin \
  --ac-bytes "${AC_N_SYMBOLS}" \
  --decoded-path work/payload_sim_stream.bin \
  --out work/ac_decode --profile "${PROFILE}" --seed "${SEED}"
```

Legacy non-incremental bitstreams will not blind-decode on this package.

## 3. Re-seal the product

```bash
cat work/payload_sim_stream.bin sidecars/payload_sim.trailer.bin \
  > work/payload_sim.product
```

The trailer is M3 densify side + `BLSMETA1` (not in the AC stream). Product must be byte-identical to encode-side `enwik9.blsmc_full.m3v2.payload_sim`.

## 4. Peel inverse → enwik9

```bash
mkdir -p work/peel
./blsmc/prepare/blsmc_prepare decode \
  work/payload_sim.product work/enwik9 \
  --dict dict/english.dic --workdir work/peel
```

Prints `DECODE OK`. Verify:

```bash
# size
wc -c < work/enwik9    # 1000000000

# official LTCB
md5 work/enwik9        # e206c3450ac99950df65bf70ef61a12d
shasum -a 1 work/enwik9
# 2996e86fb978f93cca8f566cc56998923e7fe581

# published SHA-256
shasum -a 256 work/enwik9
# 159b85351e5f76e60cbe32e04c677847a9ecba3adc79addab6f4c6c7aa3744bc
```

Run the peel binary from local disk (a UPX-packed binary can SIGSEGV on NFS). Scratch is ~2–3 GiB under `work/`.

The same steps are `./scripts/decode.sh`.

To check the incremental path without a multi-day full reconstruct, encode and blind-decode a 4 MiB prefix (`./scripts/roundtrip_4mb.sh`). See [VALIDATE.md](VALIDATE.md).
