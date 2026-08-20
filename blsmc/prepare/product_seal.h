/* Outer product seal: M3 densify side + peel meta after payload_sim.
 *
 * Layout (scored product):
 *   [stream][m3_side][M3SIDFTR][u64le side_len][meta][BLSMETA1][u64le meta_len]
 *
 * Meta is not model-compressed; BPE / train strip both trailers.
 * Legacy products with only M3SIDFTR still parse.
 */
#ifndef BLSMC_PRODUCT_SEAL_H
#define BLSMC_PRODUCT_SEAL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BLSMC_M3_SIDE_FOOTER "M3SIDFTR"
#define BLSMC_META_FOOTER "BLSMETA1"
#define BLSMC_META_VERSION 1ull

typedef struct BlsmcProductMeta {
  uint64_t version;     /* BLSMC_META_VERSION */
  uint64_t intro_bytes; /* M1 .intro size */
  uint64_t coda_bytes;  /* M1 .coda size */
  uint64_t main_bytes;  /* M1 .main size (informational) */
} BlsmcProductMeta;

/* Append side to product: [product][side][M3SIDFTR][u64le side_len].
   No-op (returns 0) if side_path missing or empty. */
int product_seal_append_m3_side(const char *product_path, const char *side_path);

/* Append peel meta after M3 seal (or bare stream). */
int product_seal_append_meta(const char *product_path,
                             const BlsmcProductMeta *meta);

/* Convenience: M3 side (if present) + meta in one call. */
int product_seal_append(const char *product_path, const char *side_path,
                        const BlsmcProductMeta *meta);

/* Strip outer BLSMETA1 if present. Optionally fill *meta.
   Returns 1 if stripped, 0 if none, -1 malformed. */
int product_seal_strip_meta(const uint8_t *buf, size_t *n,
                            BlsmcProductMeta *meta);

/* If buf ends with an M3 side trailer, set *n to stream length (strip trailer).
   Also strips outer BLSMETA1 first so callers see the bare stream.
   Returns 1 if an M3 trailer was stripped, 0 if none, -1 on malformed. */
int product_seal_strip_m3_side(const uint8_t *buf, size_t *n);

/* Split scored product → stream + M3 side (+ optional meta out).
   Returns 1 if M3 trailer present, 0 if none, -1 on error. */
int product_seal_split_file(const char *product_path, const char *stream_path,
                            const char *side_path, BlsmcProductMeta *meta_out);

#ifdef __cplusplus
}
#endif

#endif
