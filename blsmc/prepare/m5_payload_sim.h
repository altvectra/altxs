/* M5 payload_sim — similarity-cluster reorder of post-WRT blocks (C).
 *
 * Unlike cmix-lex payload_lex (lexical sort on fixed 586MB offsets), this:
 *   - auto-detects block markers (densify WRT `    L/\\xDF\\x99N` or stock
 *     exact `\\xDF\\x99N` lines)
 *   - orders by structural key + 64-bit n-gram SimHash (embedding-like)
 *   - works on any stream size (our ~582MB densified product included)
 *
 * Side magic: PLSIM1  (EOF blob + footer, same packaging shape as R1ORD3)
 */
#ifndef BLSMC_M5_PAYLOAD_SIM_H
#define BLSMC_M5_PAYLOAD_SIM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Reorder PATH in place; write side to PATH.payload_sim_side.
   Verifies round-trip into a temp buffer. Returns 0 on success. */
int m5_payload_sim_file(const char *path);

/* In-memory: *out_n may exceed in_n by side+footer. Caller frees *out. */
int m5_payload_sim_encode(const uint8_t *in, size_t in_n,
                          uint8_t **out, size_t *out_n, size_t *side_n);

/* Strip EOF side, restore original block order. */
int m5_payload_sim_restore(const uint8_t *in, size_t in_n,
                           uint8_t **out, size_t *out_n);

#ifdef __cplusplus
}
#endif

#endif
