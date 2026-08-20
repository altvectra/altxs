/* Byte-level BPE tokenizer (C) for blsmc prepare → custom vocab.
 *
 * Vocab: [0..255] bytes, [256..V) learned merges — same layout as xsa_ttt.bpe.
 * Native dict magic BBPE1 (binary, no lzma). Token file = packed u16le.
 */
#ifndef BLSMC_BPE_TOKENIZER_H
#define BLSMC_BPE_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BPE_DEFAULT_VOCAB 16384
#define BPE_DEFAULT_TRAIN_BYTES (32u << 20)
#define BPE_DEFAULT_CHUNK_BYTES (1u << 20)

typedef struct {
  uint32_t left, right;
} BpeMerge;

typedef struct {
  uint32_t vocab_size; /* model size (requested); active = 256 + n_merges */
  uint32_t n_merges;
  BpeMerge *merges; /* owned */
} BpeVocab;

void bpe_vocab_free(BpeVocab *v);

/* Train merges on data[0..train_n) (or all if train_n==0 / > n). */
int bpe_train(const uint8_t *data, size_t n, uint32_t vocab_size,
              size_t max_train_bytes, BpeVocab *out);

int bpe_save_dict(const BpeVocab *v, const char *path);
int bpe_load_dict(const char *path, BpeVocab *out);

/* Encode one contiguous byte span → newly malloc'd u16 ids (*n_out tokens). */
int bpe_encode(const BpeVocab *v, const uint8_t *data, size_t n,
               uint16_t **ids_out, size_t *n_out);

/* Decode token ids → newly malloc'd bytes. */
int bpe_decode(const BpeVocab *v, const uint16_t *ids, size_t n_ids,
               uint8_t **bytes_out, size_t *n_out);

/* Chunked encode (exact round-trip with chunk lengths). */
int bpe_encode_chunked(const BpeVocab *v, const uint8_t *data, size_t n,
                       size_t chunk_bytes, uint16_t **ids_out, size_t *n_ids,
                       uint32_t **chunk_n_out, size_t *n_chunks);

int bpe_decode_chunked(const BpeVocab *v, const uint16_t *ids, size_t n_ids,
                       const uint32_t *chunk_n, size_t n_chunks,
                       uint8_t **bytes_out, size_t *n_out);

/* Write/read packed u16le token file. */
int bpe_write_tokens(const char *path, const uint16_t *ids, size_t n);
int bpe_read_tokens(const char *path, uint16_t **ids_out, size_t *n_out);

/* Write/read chunk length sidecar (.chunks). */
int bpe_write_chunks(const char *path, size_t chunk_bytes,
                     const uint32_t *chunk_n, size_t n_chunks);
int bpe_read_chunks(const char *path, size_t *chunk_bytes,
                    uint32_t **chunk_n_out, size_t *n_chunks);

/*
 * End-to-end: train + encode + write
 *   PREFIX.bpe{V} / PREFIX.bpe{V}.dict / PREFIX.bpe{V}.chunks / PREFIX.bpe{V}.json
 * For V==16384 filename uses .bpe16384 (xsa_ttt convention).
 * mode: 0=train+encode, 1=dict-only (train+save dict), 2=encode-only (load dict).
 */
int bpe_prepare_file(const char *src_path, const char *out_prefix,
                     uint32_t vocab_size, size_t max_train_bytes,
                     size_t chunk_bytes, int verify, int mode);

#define BPE_MODE_FULL 0
#define BPE_MODE_DICT_ONLY 1
#define BPE_MODE_ENCODE_ONLY 2

#ifdef __cplusplus
}
#endif

#endif
