#include "product_seal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static size_t file_size(const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f)
    return 0;
  if (fseek(f, 0, SEEK_END) != 0) {
    fclose(f);
    return 0;
  }
  long n = ftell(f);
  fclose(f);
  return n > 0 ? (size_t)n : 0;
}

static int read_all(const char *path, uint8_t **out, size_t *out_n) {
  FILE *f = fopen(path, "rb");
  if (!f)
    return -1;
  if (fseek(f, 0, SEEK_END) != 0) {
    fclose(f);
    return -1;
  }
  long n = ftell(f);
  if (n < 0) {
    fclose(f);
    return -1;
  }
  if (fseek(f, 0, SEEK_SET) != 0) {
    fclose(f);
    return -1;
  }
  uint8_t *buf = (uint8_t *)malloc((size_t)n);
  if (!buf && n) {
    fclose(f);
    return -1;
  }
  if (n && fread(buf, 1, (size_t)n, f) != (size_t)n) {
    free(buf);
    fclose(f);
    return -1;
  }
  fclose(f);
  *out = buf;
  *out_n = (size_t)n;
  return 0;
}

static void write_u64le(uint8_t *dst, uint64_t v) {
  for (int i = 0; i < 8; i++)
    dst[i] = (uint8_t)((v >> (8 * i)) & 0xff);
}

static uint64_t read_u64le(const uint8_t *src) {
  uint64_t v = 0;
  for (int i = 0; i < 8; i++)
    v |= (uint64_t)src[i] << (8 * i);
  return v;
}

static int has_m3_trailer(const uint8_t *buf, size_t n) {
  const size_t foot = strlen(BLSMC_M3_SIDE_FOOTER);
  if (n < foot + 8)
    return 0;
  return memcmp(buf + n - foot - 8, BLSMC_M3_SIDE_FOOTER, foot) == 0;
}

static int has_meta_trailer(const uint8_t *buf, size_t n) {
  const size_t foot = strlen(BLSMC_META_FOOTER);
  if (n < foot + 8)
    return 0;
  return memcmp(buf + n - foot - 8, BLSMC_META_FOOTER, foot) == 0;
}

int product_seal_strip_meta(const uint8_t *buf, size_t *n,
                            BlsmcProductMeta *meta) {
  const size_t foot = strlen(BLSMC_META_FOOTER);
  if (!buf || !n || *n < foot + 8)
    return 0;
  size_t end = *n;
  if (memcmp(buf + end - foot - 8, BLSMC_META_FOOTER, foot) != 0)
    return 0;
  uint64_t mlen = read_u64le(buf + end - 8);
  if (mlen > end - foot - 8)
    return -1;
  if (mlen < 32)
    return -1;
  size_t meta_off = end - foot - 8 - (size_t)mlen;
  if (meta) {
    memset(meta, 0, sizeof(*meta));
    meta->version = read_u64le(buf + meta_off + 0);
    meta->intro_bytes = read_u64le(buf + meta_off + 8);
    meta->coda_bytes = read_u64le(buf + meta_off + 16);
    meta->main_bytes = read_u64le(buf + meta_off + 24);
  }
  *n = meta_off;
  return 1;
}

int product_seal_strip_m3_side(const uint8_t *buf, size_t *n) {
  if (!buf || !n)
    return 0;
  /* Peel outer meta first so BPE/train always see bare stream. */
  if (product_seal_strip_meta(buf, n, NULL) < 0)
    return -1;

  const size_t foot = strlen(BLSMC_M3_SIDE_FOOTER);
  if (*n < foot + 8)
    return 0;
  size_t end = *n;
  if (memcmp(buf + end - foot - 8, BLSMC_M3_SIDE_FOOTER, foot) != 0)
    return 0;
  uint64_t slen = read_u64le(buf + end - 8);
  if (slen > end - foot - 8)
    return -1;
  size_t stream_n = end - foot - 8 - (size_t)slen;
  *n = stream_n;
  return 1;
}

int product_seal_append_m3_side(const char *product_path, const char *side_path) {
  size_t sn = file_size(side_path);
  if (!sn)
    return 0;

  {
    uint8_t *cur = NULL;
    size_t cn = 0;
    if (read_all(product_path, &cur, &cn) != 0)
      return -1;
    size_t probe = cn;
    if (product_seal_strip_meta(cur, &probe, NULL) < 0) {
      free(cur);
      return -1;
    }
    int already = has_m3_trailer(cur, probe);
    free(cur);
    if (already) {
      fprintf(stderr, "[seal] product already has M3 side trailer (%zu B side)\n",
              sn);
      return 0;
    }
  }

  uint8_t *side = NULL;
  size_t side_n = 0;
  if (read_all(side_path, &side, &side_n) != 0) {
    perror(side_path);
    return -1;
  }

  FILE *f = fopen(product_path, "ab");
  if (!f) {
    perror(product_path);
    free(side);
    return -1;
  }
  if (fwrite(side, 1, side_n, f) != side_n) {
    fclose(f);
    free(side);
    return -1;
  }
  if (fwrite(BLSMC_M3_SIDE_FOOTER, 1, strlen(BLSMC_M3_SIDE_FOOTER), f) !=
      strlen(BLSMC_M3_SIDE_FOOTER)) {
    fclose(f);
    free(side);
    return -1;
  }
  uint8_t lenbuf[8];
  write_u64le(lenbuf, side_n);
  if (fwrite(lenbuf, 1, 8, f) != 8) {
    fclose(f);
    free(side);
    return -1;
  }
  fclose(f);
  free(side);
  fprintf(stderr,
          "[seal] appended M3 densify side %zu B → scored product (+%zu)\n",
          side_n, side_n + strlen(BLSMC_M3_SIDE_FOOTER) + 8);
  return 0;
}

int product_seal_append_meta(const char *product_path,
                             const BlsmcProductMeta *meta) {
  if (!product_path || !meta)
    return -1;

  {
    uint8_t *cur = NULL;
    size_t cn = 0;
    if (read_all(product_path, &cur, &cn) != 0)
      return -1;
    int has = has_meta_trailer(cur, cn);
    free(cur);
    if (has) {
      fprintf(stderr, "[seal] product already has BLSMETA1 trailer\n");
      return 0;
    }
  }

  uint8_t blob[32];
  write_u64le(blob + 0, meta->version ? meta->version : BLSMC_META_VERSION);
  write_u64le(blob + 8, meta->intro_bytes);
  write_u64le(blob + 16, meta->coda_bytes);
  write_u64le(blob + 24, meta->main_bytes);

  FILE *f = fopen(product_path, "ab");
  if (!f) {
    perror(product_path);
    return -1;
  }
  if (fwrite(blob, 1, sizeof(blob), f) != sizeof(blob)) {
    fclose(f);
    return -1;
  }
  if (fwrite(BLSMC_META_FOOTER, 1, strlen(BLSMC_META_FOOTER), f) !=
      strlen(BLSMC_META_FOOTER)) {
    fclose(f);
    return -1;
  }
  uint8_t lenbuf[8];
  write_u64le(lenbuf, sizeof(blob));
  if (fwrite(lenbuf, 1, 8, f) != 8) {
    fclose(f);
    return -1;
  }
  fclose(f);
  fprintf(stderr,
          "[seal] appended peel meta intro=%llu coda=%llu main=%llu (+%zu B)\n",
          (unsigned long long)meta->intro_bytes,
          (unsigned long long)meta->coda_bytes,
          (unsigned long long)meta->main_bytes,
          sizeof(blob) + strlen(BLSMC_META_FOOTER) + 8);
  return 0;
}

int product_seal_append(const char *product_path, const char *side_path,
                        const BlsmcProductMeta *meta) {
  if (side_path && side_path[0] &&
      product_seal_append_m3_side(product_path, side_path) != 0)
    return -1;
  if (meta && product_seal_append_meta(product_path, meta) != 0)
    return -1;
  return 0;
}

int product_seal_split_file(const char *product_path, const char *stream_path,
                            const char *side_path, BlsmcProductMeta *meta_out) {
  uint8_t *buf = NULL;
  size_t n = 0;
  if (read_all(product_path, &buf, &n) != 0) {
    perror(product_path);
    return -1;
  }

  BlsmcProductMeta meta;
  memset(&meta, 0, sizeof(meta));
  size_t after_meta = n;
  int meta_st = product_seal_strip_meta(buf, &after_meta, &meta);
  if (meta_st < 0) {
    free(buf);
    return -1;
  }
  if (meta_out) {
    if (meta_st == 1)
      *meta_out = meta;
    else
      memset(meta_out, 0, sizeof(*meta_out));
  }

  size_t stream_n = after_meta;
  int st = 0;
  {
    /* Strip M3 without recursively stripping meta again. */
    const size_t foot = strlen(BLSMC_M3_SIDE_FOOTER);
    if (stream_n >= foot + 8 &&
        memcmp(buf + stream_n - foot - 8, BLSMC_M3_SIDE_FOOTER, foot) == 0) {
      uint64_t slen = read_u64le(buf + stream_n - 8);
      if (slen > stream_n - foot - 8) {
        free(buf);
        return -1;
      }
      size_t side_n = (size_t)slen;
      size_t m3_end = stream_n;
      stream_n = m3_end - foot - 8 - side_n;
      st = 1;

      if (side_path) {
        FILE *fside = fopen(side_path, "wb");
        if (!fside) {
          perror(side_path);
          free(buf);
          return -1;
        }
        if (side_n && fwrite(buf + stream_n, 1, side_n, fside) != side_n) {
          fclose(fside);
          free(buf);
          return -1;
        }
        fclose(fside);
      }
      fprintf(stderr, "[seal] split: stream=%zu B side=%zu B meta=%s\n", stream_n,
              side_n, meta_st == 1 ? "yes" : "no");
    } else {
      fprintf(stderr, "[seal] split: no M3 trailer; stream=%zu B meta=%s\n",
              stream_n, meta_st == 1 ? "yes" : "no");
    }
  }

  FILE *fs = fopen(stream_path, "wb");
  if (!fs) {
    perror(stream_path);
    free(buf);
    return -1;
  }
  if (stream_n && fwrite(buf, 1, stream_n, fs) != stream_n) {
    fclose(fs);
    free(buf);
    return -1;
  }
  fclose(fs);
  free(buf);
  return st;
}
