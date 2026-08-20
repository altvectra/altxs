/* M5 payload_sim — structural-key + n-gram SimHash block reorder.
 * memmem: -D_GNU_SOURCE / -D_DARWIN_C_SOURCE from Makefile. */
#include "m5_payload_sim.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAGIC "PLSIM1\n"
#define FOOTER "PLSIMFTR"
#define MARKER_DENSE "    L/\xDF\x99N\n"
#define MARKER_STOCK "\xDF\x99N\n"

typedef struct {
  size_t start, len;
  uint64_t struct_key;
  uint64_t simhash;
  uint32_t orig;
} Block;

typedef struct {
  uint8_t *p;
  size_t n, cap;
} Buf;

static int buf_reserve(Buf *b, size_t need) {
  if (need <= b->cap)
    return 0;
  size_t cap = b->cap ? b->cap : 1 << 20;
  while (cap < need)
    cap *= 2;
  uint8_t *p = (uint8_t *)realloc(b->p, cap);
  if (!p)
    return -1;
  b->p = p;
  b->cap = cap;
  return 0;
}

static int buf_put(Buf *b, const void *src, size_t n) {
  if (buf_reserve(b, b->n + n) != 0)
    return -1;
  memcpy(b->p + b->n, src, n);
  b->n += n;
  return 0;
}

static void buf_free(Buf *b) {
  free(b->p);
  memset(b, 0, sizeof(*b));
}

static int buf_uvarint(Buf *b, uint64_t v) {
  uint8_t tmp[10];
  int k = 0;
  while (v >= 0x80) {
    tmp[k++] = (uint8_t)((v & 0x7f) | 0x80);
    v >>= 7;
  }
  tmp[k++] = (uint8_t)v;
  return buf_put(b, tmp, (size_t)k);
}

static int read_uvarint(const uint8_t *p, size_t n, size_t *off, uint64_t *out) {
  uint64_t v = 0;
  unsigned shift = 0;
  while (*off < n) {
    uint8_t c = p[(*off)++];
    v |= (uint64_t)(c & 0x7f) << shift;
    if (!(c & 0x80)) {
      *out = v;
      return 0;
    }
    shift += 7;
    if (shift > 63)
      return -1;
  }
  return -1;
}

/* ---- Fenwick / Lehmer (same idea as cmix-lex R1ORD3) ---- */

typedef struct {
  int *tree;
  size_t n;
} Fenwick;

static void fenwick_add(Fenwick *f, size_t index, int delta) {
  for (++index; index <= f->n; index += index & (0u - index))
    f->tree[index] += delta;
}

static int fenwick_init(Fenwick *f, size_t n) {
  f->n = n;
  f->tree = (int *)calloc(n + 1, sizeof(int));
  if (!f->tree)
    return -1;
  for (size_t i = 0; i < n; i++)
    fenwick_add(f, i, 1);
  return 0;
}

static void fenwick_free(Fenwick *f) {
  free(f->tree);
  memset(f, 0, sizeof(*f));
}

static size_t fenwick_sum_lt(const Fenwick *f, size_t index) {
  size_t r = 0;
  while (index > 0) {
    r += (size_t)f->tree[index];
    index -= index & (0u - index);
  }
  return r;
}

static size_t fenwick_find(const Fenwick *f, size_t rank) {
  size_t index = 0;
  size_t bit = 1;
  while ((bit << 1) <= f->n)
    bit <<= 1;
  while (bit) {
    size_t next = index + bit;
    if (next <= f->n && (size_t)f->tree[next] <= rank) {
      index = next;
      rank -= (size_t)f->tree[next];
    }
    bit >>= 1;
  }
  return index;
}

static int append_lehmer(Buf *out, const size_t *perm, size_t n) {
  Fenwick fw;
  if (fenwick_init(&fw, n) != 0)
    return -1;
  uint8_t *seen = (uint8_t *)calloc(n, 1);
  if (!seen) {
    fenwick_free(&fw);
    return -1;
  }
  for (size_t i = 0; i < n; i++) {
    size_t v = perm[i];
    if (v >= n || seen[v]) {
      free(seen);
      fenwick_free(&fw);
      return -1;
    }
    seen[v] = 1;
    if (buf_uvarint(out, fenwick_sum_lt(&fw, v)) != 0) {
      free(seen);
      fenwick_free(&fw);
      return -1;
    }
    fenwick_add(&fw, v, -1);
  }
  free(seen);
  fenwick_free(&fw);
  return 0;
}

static int read_lehmer(const uint8_t *p, size_t n, size_t *off, size_t count,
                       size_t *perm) {
  Fenwick fw;
  if (fenwick_init(&fw, count) != 0)
    return -1;
  for (size_t i = 0; i < count; i++) {
    uint64_t rank;
    if (read_uvarint(p, n, off, &rank) != 0 || rank >= count - i) {
      fenwick_free(&fw);
      return -1;
    }
    size_t v = fenwick_find(&fw, (size_t)rank);
    perm[i] = v;
    fenwick_add(&fw, v, -1);
  }
  fenwick_free(&fw);
  return 0;
}

/* ---- Similarity features ---- */

static uint64_t splitmix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

/* Structural key: length class + first-16-byte mix + DF-marker bitset. */
static uint64_t structural_key(const uint8_t *blk, size_t n) {
  uint64_t h = splitmix64(n);
  size_t take = n < 16 ? n : 16;
  for (size_t i = 0; i < take; i++)
    h = splitmix64(h ^ ((uint64_t)blk[i] << (8 * (i & 7))));
  /* Bitset of high DF second-bytes in first 256 bytes (structural tokens). */
  uint64_t bits = 0;
  size_t lim = n < 256 ? n : 256;
  for (size_t i = 0; i + 1 < lim; i++) {
    if (blk[i] == 0xDF)
      bits |= 1ull << (blk[i + 1] & 63);
  }
  return splitmix64(h ^ bits);
}

/*
 * 64-bit SimHash over byte 3-grams — a cheap random-hyperplane embedding so
 * blocks with shared n-grams land nearby in the sorted order (Hamming /
 * MI-style locality without a neural embedding table).
 */
static uint64_t simhash64(const uint8_t *blk, size_t n) {
  /* Skip short structural prefix (marker line ~12 bytes). */
  size_t off = n > 16 ? 12 : 0;
  if (off >= n)
    return 0;
  const uint8_t *p = blk + off;
  size_t m = n - off;
  int acc[64];
  memset(acc, 0, sizeof(acc));
  if (m < 3) {
    for (size_t i = 0; i < m; i++) {
      uint64_t h = splitmix64(p[i]);
      for (int b = 0; b < 64; b++)
        acc[b] += (h >> b) & 1 ? 1 : -1;
    }
  } else {
    for (size_t i = 0; i + 2 < m; i++) {
      uint64_t g = ((uint64_t)p[i] << 16) | ((uint64_t)p[i + 1] << 8) | p[i + 2];
      uint64_t h = splitmix64(g * 0x9e3779b97f4a7c15ull);
      for (int b = 0; b < 64; b++)
        acc[b] += (h >> b) & 1 ? 1 : -1;
    }
  }
  uint64_t out = 0;
  for (int b = 0; b < 64; b++)
    if (acc[b] > 0)
      out |= 1ull << b;
  return out;
}

static int cmp_block(const void *a, const void *b) {
  const Block *x = (const Block *)a;
  const Block *y = (const Block *)b;
  if (x->struct_key != y->struct_key)
    return x->struct_key < y->struct_key ? -1 : 1;
  if (x->simhash != y->simhash)
    return x->simhash < y->simhash ? -1 : 1;
  if (x->orig != y->orig)
    return x->orig < y->orig ? -1 : 1;
  return 0;
}

/* Collect marker offsets. Prefer densify marker; else stock exact line. */
static size_t *find_markers(const uint8_t *data, size_t n, size_t *count,
                            int *kind) {
  const char *pat = MARKER_DENSE;
  size_t plen = strlen(MARKER_DENSE);
  *kind = 1;
  size_t cap = 256000, nmarks = 0;
  size_t *offs = (size_t *)malloc(cap * sizeof(size_t));
  if (!offs)
    return NULL;

  /* Densify marker scan (memmem). */
  for (size_t i = 0; i + plen <= n;) {
    const void *hit = memmem(data + i, n - i, pat, plen);
    if (!hit)
      break;
    size_t at = (size_t)((const uint8_t *)hit - data);
    if (nmarks >= cap) {
      cap *= 2;
      size_t *nouts = (size_t *)realloc(offs, cap * sizeof(size_t));
      if (!nouts) {
        free(offs);
        return NULL;
      }
      offs = nouts;
    }
    offs[nmarks++] = at;
    i = at + plen;
  }

  if (nmarks >= 1000) {
    *count = nmarks;
    return offs;
  }

  /* Fallback: stock exact DF99N line */
  free(offs);
  pat = MARKER_STOCK;
  plen = 4; /* DF 99 N \n */
  *kind = 0;
  cap = 256000;
  nmarks = 0;
  offs = (size_t *)malloc(cap * sizeof(size_t));
  if (!offs)
    return NULL;
  for (size_t i = 0; i + plen <= n;) {
    const void *hit = memmem(data + i, n - i, pat, plen);
    if (!hit)
      break;
    size_t at = (size_t)((const uint8_t *)hit - data);
    if (at == 0 || data[at - 1] == '\n') {
      if (nmarks >= cap) {
        cap *= 2;
        size_t *nouts = (size_t *)realloc(offs, cap * sizeof(size_t));
        if (!nouts) {
          free(offs);
          return NULL;
        }
        offs = nouts;
      }
      offs[nmarks++] = at;
      i = at + plen;
    } else {
      i = at + 1;
    }
  }
  *count = nmarks;
  return offs;
}

static size_t median_gap(const size_t *offs, size_t n) {
  if (n < 2)
    return 1024;
  size_t sample = n - 1 < 4096 ? n - 1 : 4096;
  size_t *gaps = (size_t *)malloc(sample * sizeof(size_t));
  if (!gaps)
    return offs[1] - offs[0];
  size_t step = (n - 1) / sample;
  if (step == 0)
    step = 1;
  size_t k = 0;
  for (size_t i = 0; i + 1 < n && k < sample; i += step)
    gaps[k++] = offs[i + 1] - offs[i];
  /* insertion sort small */
  for (size_t i = 1; i < k; i++) {
    size_t v = gaps[i];
    size_t j = i;
    while (j > 0 && gaps[j - 1] > v) {
      gaps[j] = gaps[j - 1];
      j--;
    }
    gaps[j] = v;
  }
  size_t med = gaps[k / 2];
  free(gaps);
  return med ? med : 1024;
}

static uint8_t *read_file(const char *path, size_t *n_out) {
  FILE *f = fopen(path, "rb");
  if (!f)
    return NULL;
  if (fseek(f, 0, SEEK_END) != 0) {
    fclose(f);
    return NULL;
  }
  long sz = ftell(f);
  if (sz < 0) {
    fclose(f);
    return NULL;
  }
  rewind(f);
  uint8_t *buf = (uint8_t *)malloc((size_t)sz);
  if (!buf) {
    fclose(f);
    return NULL;
  }
  if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
    free(buf);
    fclose(f);
    return NULL;
  }
  fclose(f);
  *n_out = (size_t)sz;
  return buf;
}

static int write_file(const char *path, const uint8_t *p, size_t n) {
  FILE *f = fopen(path, "wb");
  if (!f)
    return -1;
  if (fwrite(p, 1, n, f) != n) {
    fclose(f);
    return -1;
  }
  fclose(f);
  return 0;
}

int m5_payload_sim_encode(const uint8_t *in, size_t in_n, uint8_t **out,
                          size_t *out_n, size_t *side_n_out) {
  size_t nmarks = 0;
  int kind = 0;
  size_t *offs = find_markers(in, in_n, &nmarks, &kind);
  if (!offs || nmarks < 2) {
    free(offs);
    fprintf(stderr, "m5_payload_sim: need >=2 block markers (found %zu)\n",
            nmarks);
    return -1;
  }

  size_t med = median_gap(offs, nmarks);
  size_t last_len = med;
  if (offs[nmarks - 1] + last_len > in_n)
    last_len = in_n - offs[nmarks - 1];
  /* Prefer previous gap when plausible */
  if (nmarks >= 2) {
    size_t prev = offs[nmarks - 1] - offs[nmarks - 2];
    if (prev > 32 && offs[nmarks - 1] + prev <= in_n)
      last_len = prev;
  }

  size_t region_end = offs[nmarks - 1] + last_len;
  size_t prelude = offs[0];
  size_t suffix = in_n - region_end;

  Block *blocks = (Block *)calloc(nmarks, sizeof(Block));
  if (!blocks) {
    free(offs);
    return -1;
  }
  for (size_t i = 0; i < nmarks; i++) {
    size_t end = (i + 1 < nmarks) ? offs[i + 1] : region_end;
    blocks[i].start = offs[i];
    blocks[i].len = end - offs[i];
    blocks[i].orig = (uint32_t)i;
    blocks[i].struct_key = structural_key(in + blocks[i].start, blocks[i].len);
    blocks[i].simhash = simhash64(in + blocks[i].start, blocks[i].len);
  }

  Block *sorted = (Block *)malloc(nmarks * sizeof(Block));
  if (!sorted) {
    free(blocks);
    free(offs);
    return -1;
  }
  memcpy(sorted, blocks, nmarks * sizeof(Block));
  qsort(sorted, nmarks, sizeof(Block), cmp_block);

  /* permutation: sorted position → original index */
  size_t *perm = (size_t *)malloc(nmarks * sizeof(size_t));
  if (!perm) {
    free(sorted);
    free(blocks);
    free(offs);
    return -1;
  }
  for (size_t i = 0; i < nmarks; i++)
    perm[i] = sorted[i].orig;

  Buf side = {0};
  if (buf_put(&side, MAGIC, strlen(MAGIC)) != 0 ||
      buf_uvarint(&side, in_n) != 0 || buf_uvarint(&side, prelude) != 0 ||
      buf_uvarint(&side, region_end - prelude) != 0 ||
      buf_uvarint(&side, suffix) != 0 || buf_uvarint(&side, (uint64_t)kind) != 0 ||
      buf_uvarint(&side, nmarks) != 0 || buf_uvarint(&side, last_len) != 0 ||
      append_lehmer(&side, perm, nmarks) != 0) {
    buf_free(&side);
    free(perm);
    free(sorted);
    free(blocks);
    free(offs);
    return -1;
  }

  Buf outb = {0};
  if (buf_put(&outb, in, prelude) != 0)
    goto fail;
  for (size_t i = 0; i < nmarks; i++) {
    if (buf_put(&outb, in + sorted[i].start, sorted[i].len) != 0)
      goto fail;
  }
  if (suffix && buf_put(&outb, in + region_end, suffix) != 0)
    goto fail;
  if (outb.n != in_n) {
    fprintf(stderr, "m5_payload_sim: size drift %zu vs %zu\n", outb.n, in_n);
    goto fail;
  }
  if (buf_put(&outb, side.p, side.n) != 0 ||
      buf_put(&outb, FOOTER, strlen(FOOTER)) != 0)
    goto fail;
  /* side length u64le */
  uint8_t lenbuf[8];
  for (int i = 0; i < 8; i++)
    lenbuf[i] = (uint8_t)((side.n >> (8 * i)) & 0xff);
  if (buf_put(&outb, lenbuf, 8) != 0)
    goto fail;

  fprintf(stderr,
          "m5_payload_sim: blocks=%zu kind=%s prelude=%zu region=%zu "
          "suffix=%zu side=%zu order=struct+simhash64\n",
          nmarks, kind ? "densify_L/DF99N" : "stock_DF99N", prelude,
          region_end - prelude, suffix, side.n);

  *out = outb.p;
  *out_n = outb.n;
  if (side_n_out)
    *side_n_out = side.n;
  outb.p = NULL;
  buf_free(&side);
  free(perm);
  free(sorted);
  free(blocks);
  free(offs);
  return 0;

fail:
  buf_free(&side);
  buf_free(&outb);
  free(perm);
  free(sorted);
  free(blocks);
  free(offs);
  return -1;
}

int m5_payload_sim_restore(const uint8_t *in, size_t in_n, uint8_t **out,
                           size_t *out_n) {
  const size_t footer_len = strlen(FOOTER);
  if (in_n < footer_len + 8)
    return -1;
  size_t footer_pos = in_n - footer_len - 8;
  if (memcmp(in + footer_pos, FOOTER, footer_len) != 0)
    return -1;
  uint64_t side_len = 0;
  for (int i = 0; i < 8; i++)
    side_len |= (uint64_t)in[footer_pos + footer_len + i] << (8 * i);
  if (side_len > footer_pos)
    return -1;
  size_t side_pos = footer_pos - (size_t)side_len;
  const uint8_t *side = in + side_pos;
  size_t stream_n = side_pos;
  const uint8_t *stream = in;

  if (side_len < strlen(MAGIC) || memcmp(side, MAGIC, strlen(MAGIC)) != 0)
    return -1;
  size_t off = strlen(MAGIC);
  uint64_t orig_n, prelude, region_len, suffix, kind64, nmarks64, last_len64;
  if (read_uvarint(side, (size_t)side_len, &off, &orig_n) != 0 ||
      read_uvarint(side, (size_t)side_len, &off, &prelude) != 0 ||
      read_uvarint(side, (size_t)side_len, &off, &region_len) != 0 ||
      read_uvarint(side, (size_t)side_len, &off, &suffix) != 0 ||
      read_uvarint(side, (size_t)side_len, &off, &kind64) != 0 ||
      read_uvarint(side, (size_t)side_len, &off, &nmarks64) != 0 ||
      read_uvarint(side, (size_t)side_len, &off, &last_len64) != 0)
    return -1;
  size_t nmarks = (size_t)nmarks64;
  if (stream_n != orig_n || prelude + region_len + suffix != stream_n)
    return -1;

  size_t *perm = (size_t *)malloc(nmarks * sizeof(size_t));
  if (!perm)
    return -1;
  if (read_lehmer(side, (size_t)side_len, &off, nmarks, perm) != 0 ||
      off != side_len) {
    free(perm);
    return -1;
  }

  /* Re-parse sorted blocks from transformed stream (same marker scan). */
  size_t nmarks2 = 0;
  int kind2 = 0;
  size_t *offs = find_markers(stream, stream_n, &nmarks2, &kind2);
  if (!offs || nmarks2 != nmarks) {
    free(offs);
    free(perm);
    return -1;
  }
  size_t region_end = prelude + (size_t)region_len;
  Block *sorted_blocks = (Block *)calloc(nmarks, sizeof(Block));
  if (!sorted_blocks) {
    free(offs);
    free(perm);
    return -1;
  }
  for (size_t i = 0; i < nmarks; i++) {
    size_t end = (i + 1 < nmarks) ? offs[i + 1] : region_end;
    sorted_blocks[i].start = offs[i];
    sorted_blocks[i].len = end - offs[i];
  }

  /* perm[d86_pos]-style: perm[i] = original index of sorted block i
     Wait — we stored perm[sorted_pos] = original_index.
     So original_blocks[perm[i]] = sorted_blocks[i] */
  Block **by_orig = (Block **)calloc(nmarks, sizeof(Block *));
  if (!by_orig) {
    free(sorted_blocks);
    free(offs);
    free(perm);
    return -1;
  }
  for (size_t i = 0; i < nmarks; i++) {
    size_t oi = perm[i];
    if (oi >= nmarks || by_orig[oi]) {
      free(by_orig);
      free(sorted_blocks);
      free(offs);
      free(perm);
      return -1;
    }
    by_orig[oi] = &sorted_blocks[i];
  }

  Buf outb = {0};
  if (buf_put(&outb, stream, (size_t)prelude) != 0)
    goto fail;
  for (size_t i = 0; i < nmarks; i++) {
    Block *b = by_orig[i];
    if (!b || buf_put(&outb, stream + b->start, b->len) != 0)
      goto fail;
  }
  if (suffix &&
      buf_put(&outb, stream + region_end, (size_t)suffix) != 0)
    goto fail;
  if (outb.n != stream_n)
    goto fail;

  *out = outb.p;
  *out_n = outb.n;
  outb.p = NULL;
  free(by_orig);
  free(sorted_blocks);
  free(offs);
  free(perm);
  buf_free(&outb);
  return 0;

fail:
  free(by_orig);
  free(sorted_blocks);
  free(offs);
  free(perm);
  buf_free(&outb);
  return -1;
}

int m5_payload_sim_file(const char *path) {
  size_t n = 0;
  uint8_t *in = read_file(path, &n);
  if (!in) {
    perror(path);
    return -1;
  }
  uint8_t *encoded = NULL;
  size_t enc_n = 0, side_n = 0;
  if (m5_payload_sim_encode(in, n, &encoded, &enc_n, &side_n) != 0) {
    free(in);
    return -1;
  }

  /* Round-trip check */
  uint8_t *restored = NULL;
  size_t rest_n = 0;
  if (m5_payload_sim_restore(encoded, enc_n, &restored, &rest_n) != 0 ||
      rest_n != n || memcmp(restored, in, n) != 0) {
    fprintf(stderr, "m5_payload_sim: round-trip FAILED\n");
    free(in);
    free(encoded);
    free(restored);
    return -1;
  }
  free(restored);
  free(in);

  char side_path[4096];
  snprintf(side_path, sizeof(side_path), "%s.payload_sim_side", path);
  /* Extract side for sidecar convenience (also embedded at EOF). */
  const size_t footer_len = strlen(FOOTER);
  size_t footer_pos = enc_n - footer_len - 8;
  uint64_t slen = 0;
  for (int i = 0; i < 8; i++)
    slen |= (uint64_t)encoded[footer_pos + footer_len + i] << (8 * i);
  size_t side_pos = footer_pos - (size_t)slen;
  if (write_file(side_path, encoded + side_pos, (size_t)slen) != 0 ||
      write_file(path, encoded, enc_n) != 0) {
    free(encoded);
    return -1;
  }
  fprintf(stderr, "[M5] payload_sim OK stream=%zu (+side %zu embedded) wrote %s\n",
          enc_n, (size_t)slen, path);
  free(encoded);
  (void)side_n;
  return 0;
}
