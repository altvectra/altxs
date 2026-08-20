# Vendors

`vendor/cmix-lex` is the **peel compile subset** of [cmix-lex](https://github.com/blahem/cmix-lex) (GPL-3), pinned in `cmix-lex/SOURCE.txt`. It is enough to build `blsmc_prepare` and emit `payload_sim`. The cmix neural compressor is not included.

Tables used at encode time also live in `dict/`:

| File | Role |
|---|---|
| `dict/english.dic` | M4 WRT (also packed into the decoder zip / S) |
| `dict/new_article_order` | M2 article reorder (encode-side; decode uses `BLSMETA1` + page IDs) |

```bash
./scripts/fetch_vendors.sh          # verify peel subset
REFRESH=1 ./scripts/fetch_vendors.sh
./scripts/fetch_upx.sh              # required: package_s.sh always UPX-packs the peel binary
```
