# Unique contributions to Total S

LTCB ranks **S = |bitstream| + |zip -9 of the decoder|**. Tagged
`ltcb-3.15bpw`: 93,154,708 + 13,437,796 = **106,592,504**.

This note is only the pieces that are original to this stack **and**
move S. Two levers: fewer (or cheaper) AC symbols, and a smaller decoder
zip. Speed, lockstep engineering, and inherited cmix / NNCP / LTCB
machinery are out of scope.

The S win is a coupling: mixed-bit ΔW makes the zip small; peel + motif
+ replenish keep the bitstream from paying that quantization back.

---

## 1. Mixed-bit `Init(seed) + ΔW` — decoder zip

A dense ~32 M student as fp16 would dominate S. The shipped product is
not a checkpoint:

```
student = Init(seed) + dequantized ΔW
```

`mixed_da_bpw3.15_upb1.8.safetensors` (+ `.json`) is that residual. It
sits in the decoder zip.

- **Per-channel 2/3/4-bit allocation.** Bits go to channels that cut
  AC rate; dead / cheap channels stay 2-bit.
- **Functional banks**, not per-layer dumps (`qo_bank`, `kv_bank`,
  `mlp_up_bank`, `mlp_down_bank`, `tok_emb.weight`). One layout for the
  codec and the model.
- **Small 1-D params** (gates, lambdas, skip weights, …) as raw fp16 on
  top of `Init(seed)` — not mixed-bit, not a second dense dump.
- **`MBZ1` zlib packing** of the packed streams, so the codec file
  itself zips smaller.

Decoder zip is 13.4 MiB *including* peel, `english.dic`, and Python.
Without this codec the student would not fit that budget.

---

## 2. Replenish + Forward-XM — bitstream, given (1)

Mixed-bit ΔW would raise AC rate if the student stayed frozen. Online
adaptation is NNCP-class; these two changes are the S-relevant ones:

- **Replenish.** Full-parameter CE on the just-coded chunk (previous
  block as context). Heals quantization damage during the stream so
  the cheap ΔW in the zip does not become a bitstream tax.
- **Forward-XM.** At each 16 k boundary, from the same snapshot, try
  K=3 update scales (0.5× / 1× / 2×, plus 4× while prefix < 100 MiB).
  Score on a held-out prefix probe; keep the winner. Encode and decode
  pick the same update from `(cfg, end)` + decoded prefix.

Together with (1): zip pays ~3.15 bits/weight; replenish/XM buy back
the bpw that a static quantized student would lose.

---

## 3. Densify + `payload_sim` — fewer, easier symbols

AC codes a peeled byte stream, not raw `enwik9`. Stock cmix-lex already
does split / reorder / PHDA9 / WRT. Two original stages cut S:

**M3 densify.** PHDA9 headers and language packs become typed ops
(`M3H2` / `M3L1`). WRT then sees a denser stream. Incompressible
densify side (~3 KiB) is sealed onto the product and shipped in the
zip, not arithmetic-coded.

**M5 `payload_sim`.** Replaces cmix-lex `payload_lex` (lexical sort on
fixed ~586 MB offsets). Blocks are ordered by structural key + 64-bit
n-gram SimHash, so similar Wikipedia structure sits in the same
transformer window — lower bits/symbol.

AC stream length is **576,278,322** bytes vs `payload_lex` **587,138,826**.
That is ~11 M fewer symbols at the ranking rate, plus the SimHash
locality gain on the symbols that remain.

---

## 4. Motif at fixed student size — bitstream per parameter

The `large` student is 11L / 512d (~32 M, tied byte embeddings). Extra
parameters would inflate ΔW and the zip. The motif buys rate without
that:

| Motif | S role |
|---|---|
| **XSA-all** | Residual attention: subtract KV-group value projection |
| **SmearGate** | Cheap previous-token mixing in embedding space |
| **SparseAttnGate** | Per-head gate from 12 dims (tiny in ΔW) |
| **U-Net skips + gates** | Encoder→decoder reuse, no extra layers |
| **Depth recurrence** | Layers 3–5 looped twice: extra depth, same bank rows / same ΔW |
| **Parallel residuals** | Attn and MLP lanes from layer 8; mean mix |
| **AsymLogit** | Separate +/− tanh softcaps (two scalars, not a bigger head) |
| **`LeakyReLU(0.5)²` MLP** | Same hidden width, better byte-LM fit than SwiGLU here |

Depth recurrence is the explicit zip-side trick: effective depth
without more `qo_bank` / `kv_bank` / MLP rows to encode.

---

S moves when the zip stays a mixed-bit residual and the bitstream stays
a short, local, adapted byte stream. That coupling is the advantage.
