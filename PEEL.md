# Peel: enwik9 → payload_sim

This is the byte product AC encode consumes. Vocab is V=256. Trailers are
**not** in the AC stream.

```
enwik9 (1,000,000,000 B)
  M1  split4Comp          → .intro / .main / .coda
  M2  article reorder     → dict/new_article_order
  M3  PHDA9 + densify     → ready4cmix + M3 side
  M4  WRT                 → dict/english.dic
  M5  payload_sim         → struct + SimHash reorder
  seal                    → M3 side + BLSMETA1 trailer
data/enwik9.blsmc_full.m3v2.payload_sim
```

| Piece | Size | In AC stream? |
|---|---:|---|
| payload_sim byte stream | **576,278,322** | yes (`AC_N_SYMBOLS`) |
| M3 densify side + `BLSMETA1` trailer | ~3,208 | no (decoder-zip sidecar) |
| scored product file | ~576,281,530 | stream + trailer |

## Procure deps and build

```bash
./scripts/setup.sh                         # vendors + blsmc_prepare
WITH_ENWIK9=1 ./scripts/setup.sh           # also download official enwik9
WITH_ENWIK9=1 WITH_PEEL=1 ./scripts/setup.sh
```

`setup.sh` calls:

| Script | What it procures / builds |
|---|---|
| `scripts/fetch_vendors.sh` | cmix-lex peel subset (already in `vendor/cmix-lex`); copies `dict/english.dic` + `dict/new_article_order`. `REFRESH=1` re-pulls pinned commit `370e698f7ea62168cc64326ff97950c3dc212691`. |
| `make -C blsmc/prepare` | `blsmc_prepare` (needs clang++/c++, C++17) |
| `scripts/fetch_enwik9.sh` | official 1 GB corpus (not git) |
| `scripts/setup_python.sh` | `.venv` + `requirements.txt` (needed for AC, not for peel) |

## Run peel

```bash
./scripts/peel.sh
# or: ./scripts/encode.sh --peel-only
# or: make -C blsmc/prepare encode9
```

Needs `data/enwik9` (from `fetch_enwik9.sh`), `dict/english.dic`, `dict/new_article_order`, and a built `blsmc_prepare`. Scratch is ~2–3 GiB under `data/blsmc_prepare_work`. PHDA9 + WRT is slow (CPU, tens of minutes to hours).

Then AC encode that product with the same `DECODE.env` — see [ENCODE.md](ENCODE.md).
