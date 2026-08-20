/* M3 PHDA9 densify — header (M3H2) + lang pack (M3L1). Pure C. */
#ifndef BLSMC_M3_DENSIFY_H
#define BLSMC_M3_DENSIFY_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Densify one PHDA9 file: writes PATH.dense and PATH.dense.side.
   Round-trips header+lang before writing. Returns 0 on success. */
int m3_densify_file(const char *path);

/* In-memory densify. Caller frees *out_data and *side via free(). */
int m3_densify_phda9(const uint8_t *in, size_t in_n,
                     uint8_t **out_data, size_t *out_n,
                     uint8_t **side, size_t *side_n);

/* Inverse of densify: dense PHDA9 + M3H2 side → raw PHDA9.
   Caller frees *out_data via free(). */
int m3_undensify_phda9(const uint8_t *dense, size_t dense_n,
                       const uint8_t *side, size_t side_n,
                       uint8_t **out_data, size_t *out_n);

/* File helper: write PATH.raw (or out_path if non-NULL). */
int m3_undensify_file(const char *dense_path, const char *side_path,
                      const char *out_path);

#ifdef __cplusplus
}
#endif

#endif
