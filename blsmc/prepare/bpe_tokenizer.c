/* Byte-level BPE — train / encode / decode for custom vocab (C). */
#include "bpe_tokenizer.h"
#include "product_seal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define DICT_MAGIC "BBPE1\n"
#define CHUNK_MAGIC "BCH1\n"

static double bpe_now(void) {
  struct timeval tv;
  if (gettimeofday(&tv, NULL) != 0)
    return 0;
  return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
}

static void bpe_fmt_secs(double s, char *buf, size_t n) {
  if (s < 0 || s > 1e12) {
    snprintf(buf, n, "?");
    return;
  }
  unsigned long sec = (unsigned long)(s + 0.5);
  unsigned long h = sec / 3600;
  unsigned long m = (sec % 3600) / 60;
  unsigned long r = sec % 60;
  if (h)
    snprintf(buf, n, "%luh%02lum%02lus", h, m, r);
  else if (m)
    snprintf(buf, n, "%lum%02lus", m, r);
  else
    snprintf(buf, n, "%lus", r);
}

void bpe_vocab_free(BpeVocab *v) {
  if (!v)
    return;
  free(v->merges);
  memset(v, 0, sizeof(*v));
}

static uint32_t rd_u32le(const uint8_t *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
         ((uint32_t)p[3] << 24);
}

static void wr_u32le(uint8_t *p, uint32_t v) {
  p[0] = (uint8_t)(v & 0xff);
  p[1] = (uint8_t)((v >> 8) & 0xff);
  p[2] = (uint8_t)((v >> 16) & 0xff);
  p[3] = (uint8_t)((v >> 24) & 0xff);
}

static uint8_t *read_all(const char *path, size_t *n_out) {
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
  uint8_t *buf = (uint8_t *)malloc((size_t)sz ? (size_t)sz : 1);
  if (!buf) {
    fclose(f);
    return NULL;
  }
  if (sz && fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
    free(buf);
    fclose(f);
    return NULL;
  }
  fclose(f);
  *n_out = (size_t)sz;
  return buf;
}

static int write_all(const char *path, const void *p, size_t n) {
  FILE *f = fopen(path, "wb");
  if (!f)
    return -1;
  if (n && fwrite(p, 1, n, f) != n) {
    fclose(f);
    return -1;
  }
  fclose(f);
  return 0;
}

/* ---- pair hash for training ---- */

typedef struct {
  uint32_t a, b;
  uint32_t count;
  int used;
} PairSlot;

typedef struct {
  PairSlot *slots;
  size_t cap, fill;
} PairMap;

static uint32_t pair_hash(uint32_t a, uint32_t b) {
  uint64_t x = ((uint64_t)a << 32) | b;
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdull;
  x ^= x >> 33;
  return (uint32_t)x;
}

static int pairmap_init(PairMap *m, size_t cap) {
  /* power of two */
  size_t c = 1024;
  while (c < cap)
    c *= 2;
  m->cap = c;
  m->fill = 0;
  m->slots = (PairSlot *)calloc(c, sizeof(PairSlot));
  return m->slots ? 0 : -1;
}

static void pairmap_free(PairMap *m) {
  free(m->slots);
  memset(m, 0, sizeof(*m));
}

static PairSlot *pairmap_find(PairMap *m, uint32_t a, uint32_t b, int create) {
  uint32_t h = pair_hash(a, b);
  size_t i = h & (m->cap - 1);
  for (;;) {
    PairSlot *s = &m->slots[i];
    if (!s->used) {
      if (!create)
        return NULL;
      if (m->fill * 10 >= m->cap * 7)
        return NULL;
      s->used = 1;
      s->a = a;
      s->b = b;
      s->count = 0;
      m->fill++;
      return s;
    }
    if (s->a == a && s->b == b)
      return s;
    i = (i + 1) & (m->cap - 1);
  }
}

int bpe_train(const uint8_t *data, size_t n, uint32_t vocab_size,
              size_t max_train_bytes, BpeVocab *out) {
  memset(out, 0, sizeof(*out));
  if (vocab_size < 256)
    return -1;
  size_t train_n = n;
  if (max_train_bytes > 0 && max_train_bytes < train_n)
    train_n = max_train_bytes;
  if (train_n == 0) {
    out->vocab_size = vocab_size;
    out->n_merges = 0;
    out->merges = NULL;
    return 0;
  }

  double t0 = bpe_now();
  fprintf(stderr, "  [bpe] init ids from %zu train bytes…\n", train_n);
  fflush(stderr);

  uint32_t *ids = (uint32_t *)malloc(train_n * sizeof(uint32_t));
  if (!ids)
    return -1;
  for (size_t i = 0; i < train_n; i++) {
    ids[i] = data[i];
    if ((i & ((1u << 26) - 1)) == 0 && i) {
      double pct = 100.0 * (double)i / (double)train_n;
      fprintf(stderr, "  [bpe] init %5.1f%%  (%zu / %zu)\n", pct, i, train_n);
      fflush(stderr);
    }
  }
  size_t n_ids = train_n;
  fprintf(stderr, "  [bpe] init done (%.1fs) — learning %u merges\n",
          bpe_now() - t0, vocab_size - 256);
  fflush(stderr);

  uint32_t max_merges = vocab_size - 256;
  BpeMerge *merges = (BpeMerge *)malloc((size_t)max_merges * sizeof(BpeMerge));
  if (!merges) {
    free(ids);
    return -1;
  }
  uint32_t n_merges = 0;
  double t_train = bpe_now();
  double t_last_report = t_train;
  size_t ids0 = n_ids;

  while (n_merges < max_merges && n_ids >= 2) {
    PairMap map;
    /* ~2x pairs capacity */
    if (pairmap_init(&map, n_ids * 2 + 1024) != 0)
      break;

    /* Pair count — dominant cost early; report inside the scan. */
    double t_count = bpe_now();
    double t_scan_report = t_count;
    for (size_t i = 0; i + 1 < n_ids; i++) {
      PairSlot *s = pairmap_find(&map, ids[i], ids[i + 1], 1);
      if (!s) {
        /* grow: rebuild larger — rare */
        pairmap_free(&map);
        goto train_done;
      }
      s->count++;
      if ((i & ((1u << 25) - 1)) == 0) { /* ~32M steps */
        double now = bpe_now();
        if (now - t_scan_report >= 2.0 || i == 0) {
          double pct = 100.0 * (double)i / (double)(n_ids - 1);
          double rate = (now > t_count) ? (double)i / (now - t_count) : 0;
          fprintf(stderr,
                  "  [bpe] count pairs merge %u/%u  %5.1f%%  ids=%zu  "
                  "%.0f Mpos/s\r",
                  n_merges + 1, max_merges, pct, n_ids, rate / 1e6);
          fflush(stderr);
          t_scan_report = now;
        }
      }
    }
    fprintf(stderr, "\n");

    uint32_t best_a = 0, best_b = 0, best_c = 0;
    int found = 0;
    for (size_t i = 0; i < map.cap; i++) {
      PairSlot *s = &map.slots[i];
      if (!s->used || s->count < 2)
        continue;
      if (!found || s->count > best_c ||
          (s->count == best_c &&
           (s->a < best_a || (s->a == best_a && s->b < best_b)))) {
        /* Python: max by (count, -a, -b) → higher count, then lower a, lower b
           Wait Python: key=lambda kv: (kv[1], -kv[0][0], -kv[0][1])
           so max count, then max -a i.e. min a, then min b. Yes. */
        found = 1;
        best_c = s->count;
        best_a = s->a;
        best_b = s->b;
      }
    }
    pairmap_free(&map);
    if (!found)
      break;

    uint32_t new_id = 256 + n_merges;
    merges[n_merges].left = best_a;
    merges[n_merges].right = best_b;
    n_merges++;

    uint32_t *nxt = (uint32_t *)malloc(n_ids * sizeof(uint32_t));
    if (!nxt)
      break;
    size_t j = 0;
    size_t i = 0;
    while (i < n_ids) {
      if (i + 1 < n_ids && ids[i] == best_a && ids[i + 1] == best_b) {
        nxt[j++] = new_id;
        i += 2;
      } else {
        nxt[j++] = ids[i++];
      }
    }
    free(ids);
    ids = nxt;
    n_ids = j;

    double now = bpe_now();
    int report = 0;
    if (n_merges <= 5 || n_merges == max_merges)
      report = 1;
    else if (n_merges < 64 && (n_merges % 8) == 0)
      report = 1;
    else if ((n_merges % 64) == 0)
      report = 1;
    else if (now - t_last_report >= 5.0)
      report = 1;
    if (report) {
      double elapsed = now - t_train;
      double pct = 100.0 * (double)n_merges / (double)max_merges;
      double mps = (elapsed > 0) ? (double)n_merges / elapsed : 0;
      double eta = (mps > 0) ? (double)(max_merges - n_merges) / mps : -1;
      char eta_s[32], el_s[32];
      bpe_fmt_secs(eta, eta_s, sizeof(eta_s));
      bpe_fmt_secs(elapsed, el_s, sizeof(el_s));
      double shrink = (ids0 > 0) ? 100.0 * (1.0 - (double)n_ids / (double)ids0) : 0;
      fprintf(stderr,
              "  [bpe] merge %u / %u  (%5.1f%%)  ids=%zu (%.1f%% shrunk)  "
              "best=%u  %.2f merges/s  elapsed %s  eta %s\n",
              n_merges, max_merges, pct, n_ids, shrink, best_c, mps, el_s,
              eta_s);
      fflush(stderr);
      t_last_report = now;
    }
  }

train_done:
  free(ids);
  out->vocab_size = vocab_size;
  out->n_merges = n_merges;
  out->merges = merges;
  {
    char el_s[32];
    bpe_fmt_secs(bpe_now() - t0, el_s, sizeof(el_s));
    fprintf(stderr, "[bpe] trained active_vocab=%u n_merges=%u  total %s\n",
            256 + n_merges, n_merges, el_s);
    fflush(stderr);
  }
  return 0;
}

int bpe_save_dict(const BpeVocab *v, const char *path) {
  size_t hdr = strlen(DICT_MAGIC) + 8;
  size_t body = (size_t)v->n_merges * 8;
  uint8_t *buf = (uint8_t *)malloc(hdr + body);
  if (!buf)
    return -1;
  memcpy(buf, DICT_MAGIC, strlen(DICT_MAGIC));
  size_t o = strlen(DICT_MAGIC);
  wr_u32le(buf + o, v->vocab_size);
  o += 4;
  wr_u32le(buf + o, v->n_merges);
  o += 4;
  for (uint32_t i = 0; i < v->n_merges; i++) {
    wr_u32le(buf + o, v->merges[i].left);
    o += 4;
    wr_u32le(buf + o, v->merges[i].right);
    o += 4;
  }
  int rc = write_all(path, buf, o);
  free(buf);
  return rc;
}

int bpe_load_dict(const char *path, BpeVocab *out) {
  memset(out, 0, sizeof(*out));
  size_t n = 0;
  uint8_t *buf = read_all(path, &n);
  if (!buf)
    return -1;
  size_t mlen = strlen(DICT_MAGIC);
  if (n < mlen + 8 || memcmp(buf, DICT_MAGIC, mlen) != 0) {
    free(buf);
    return -1;
  }
  size_t o = mlen;
  uint32_t vocab = rd_u32le(buf + o);
  o += 4;
  uint32_t nm = rd_u32le(buf + o);
  o += 4;
  if (o + (size_t)nm * 8 != n) {
    free(buf);
    return -1;
  }
  BpeMerge *merges = (BpeMerge *)malloc((size_t)nm * sizeof(BpeMerge));
  if (!merges && nm) {
    free(buf);
    return -1;
  }
  for (uint32_t i = 0; i < nm; i++) {
    merges[i].left = rd_u32le(buf + o);
    o += 4;
    merges[i].right = rd_u32le(buf + o);
    o += 4;
  }
  free(buf);
  out->vocab_size = vocab;
  out->n_merges = nm;
  out->merges = merges;
  return 0;
}

/* rank table: pair → merge index; -1 if absent. Sparse via open addressing. */
typedef struct {
  uint32_t a, b;
  int32_t rank; /* -1 empty */
} RankSlot;

typedef struct {
  RankSlot *slots;
  size_t cap;
} RankMap;

static int rankmap_build(RankMap *m, const BpeVocab *v) {
  size_t c = 1024;
  while (c < (size_t)v->n_merges * 2 + 1024)
    c *= 2;
  m->cap = c;
  m->slots = (RankSlot *)malloc(c * sizeof(RankSlot));
  if (!m->slots)
    return -1;
  for (size_t i = 0; i < c; i++)
    m->slots[i].rank = -1;
  for (uint32_t i = 0; i < v->n_merges; i++) {
    uint32_t a = v->merges[i].left, b = v->merges[i].right;
    size_t j = pair_hash(a, b) & (c - 1);
    for (;;) {
      if (m->slots[j].rank < 0) {
        m->slots[j].a = a;
        m->slots[j].b = b;
        m->slots[j].rank = (int32_t)i;
        break;
      }
      j = (j + 1) & (c - 1);
    }
  }
  return 0;
}

static void rankmap_free(RankMap *m) {
  free(m->slots);
  memset(m, 0, sizeof(*m));
}

static int rankmap_get(const RankMap *m, uint32_t a, uint32_t b) {
  size_t j = pair_hash(a, b) & (m->cap - 1);
  for (;;) {
    if (m->slots[j].rank < 0)
      return -1;
    if (m->slots[j].a == a && m->slots[j].b == b)
      return m->slots[j].rank;
    j = (j + 1) & (m->cap - 1);
  }
}

static int apply_merges(const BpeVocab *v, const RankMap *rm, uint32_t *ids,
                        size_t *n_ids) {
  if (!v->n_merges || *n_ids < 2)
    return 0;
  for (;;) {
    int best_rank = -1;
    size_t best_i = 0;
    size_t n = *n_ids;
    for (size_t i = 0; i + 1 < n; i++) {
      int r = rankmap_get(rm, ids[i], ids[i + 1]);
      if (r < 0)
        continue;
      if (best_rank < 0 || r < best_rank) {
        best_rank = r;
        best_i = i;
      }
    }
    if (best_rank < 0)
      break;
    ids[best_i] = 256u + (uint32_t)best_rank;
    memmove(ids + best_i + 1, ids + best_i + 2,
            (n - best_i - 2) * sizeof(uint32_t));
    (*n_ids)--;
  }
  return 0;
}

int bpe_encode(const BpeVocab *v, const uint8_t *data, size_t n,
               uint16_t **ids_out, size_t *n_out) {
  if (n == 0) {
    *ids_out = NULL;
    *n_out = 0;
    return 0;
  }
  uint32_t *ids = (uint32_t *)malloc(n * sizeof(uint32_t));
  if (!ids)
    return -1;
  for (size_t i = 0; i < n; i++)
    ids[i] = data[i];
  size_t n_ids = n;

  RankMap rm;
  if (rankmap_build(&rm, v) != 0) {
    free(ids);
    return -1;
  }
  apply_merges(v, &rm, ids, &n_ids);
  rankmap_free(&rm);

  uint16_t *out = (uint16_t *)malloc(n_ids * sizeof(uint16_t));
  if (!out) {
    free(ids);
    return -1;
  }
  for (size_t i = 0; i < n_ids; i++) {
    if (ids[i] > 0xffff) {
      free(out);
      free(ids);
      return -1;
    }
    out[i] = (uint16_t)ids[i];
  }
  free(ids);
  *ids_out = out;
  *n_out = n_ids;
  return 0;
}

static int expand_rec(const BpeVocab *v, uint32_t tid, uint8_t **buf,
                      size_t *n, size_t *cap) {
  if (tid < 256) {
    if (*n + 1 > *cap) {
      size_t nc = *cap ? *cap * 2 : 256;
      while (nc < *n + 1)
        nc *= 2;
      uint8_t *p = (uint8_t *)realloc(*buf, nc);
      if (!p)
        return -1;
      *buf = p;
      *cap = nc;
    }
    (*buf)[(*n)++] = (uint8_t)tid;
    return 0;
  }
  uint32_t mi = tid - 256;
  if (mi >= v->n_merges)
    return -1;
  if (expand_rec(v, v->merges[mi].left, buf, n, cap) != 0)
    return -1;
  return expand_rec(v, v->merges[mi].right, buf, n, cap);
}

int bpe_decode(const BpeVocab *v, const uint16_t *ids, size_t n_ids,
               uint8_t **bytes_out, size_t *n_out) {
  uint8_t *buf = NULL;
  size_t n = 0, cap = 0;
  for (size_t i = 0; i < n_ids; i++) {
    if (expand_rec(v, ids[i], &buf, &n, &cap) != 0) {
      free(buf);
      return -1;
    }
  }
  *bytes_out = buf ? buf : (uint8_t *)malloc(1);
  *n_out = n;
  return 0;
}

int bpe_encode_chunked(const BpeVocab *v, const uint8_t *data, size_t n,
                       size_t chunk_bytes, uint16_t **ids_out, size_t *n_ids,
                       uint32_t **chunk_n_out, size_t *n_chunks) {
  if (chunk_bytes == 0)
    return -1;
  size_t nc = n ? (n + chunk_bytes - 1) / chunk_bytes : 0;
  uint32_t *counts = (uint32_t *)calloc(nc ? nc : 1, sizeof(uint32_t));
  if (!counts)
    return -1;

  size_t cap = n ? n : 1; /* upper bound: no merge shrinks below 1 token/byte worst */
  uint16_t *flat = (uint16_t *)malloc(cap * sizeof(uint16_t));
  if (!flat) {
    free(counts);
    return -1;
  }
  size_t total = 0;
  double t0 = bpe_now();
  double t_last = t0;
  for (size_t c = 0; c < nc; c++) {
    size_t off = c * chunk_bytes;
    size_t len = chunk_bytes;
    if (off + len > n)
      len = n - off;
    uint16_t *piece = NULL;
    size_t pn = 0;
    if (bpe_encode(v, data + off, len, &piece, &pn) != 0) {
      free(flat);
      free(counts);
      return -1;
    }
    if (total + pn > cap) {
      cap = (total + pn) * 2;
      uint16_t *nf = (uint16_t *)realloc(flat, cap * sizeof(uint16_t));
      if (!nf) {
        free(piece);
        free(flat);
        free(counts);
        return -1;
      }
      flat = nf;
    }
    memcpy(flat + total, piece, pn * sizeof(uint16_t));
    total += pn;
    counts[c] = (uint32_t)pn;
    free(piece);

    double now = bpe_now();
    if (c == 0 || c + 1 == nc || now - t_last >= 2.0 || ((c + 1) % 64) == 0) {
      double pct = 100.0 * (double)(c + 1) / (double)nc;
      double elapsed = now - t0;
      double bps = elapsed > 0 ? (double)(off + len) / elapsed : 0;
      double eta =
          (bps > 0 && off + len < n) ? (double)(n - off - len) / bps : 0;
      char eta_s[32];
      bpe_fmt_secs(eta, eta_s, sizeof(eta_s));
      fprintf(stderr,
              "  [bpe] encode %zu / %zu chunks  (%5.1f%%)  tokens=%zu  "
              "%.1f MB/s  eta %s\n",
              c + 1, nc, pct, total, bps / (1024.0 * 1024.0), eta_s);
      fflush(stderr);
      t_last = now;
    }
  }
  *ids_out = flat;
  *n_ids = total;
  *chunk_n_out = counts;
  *n_chunks = nc;
  return 0;
}

int bpe_decode_chunked(const BpeVocab *v, const uint16_t *ids, size_t n_ids,
                       const uint32_t *chunk_n, size_t n_chunks,
                       uint8_t **bytes_out, size_t *n_out) {
  uint8_t *buf = NULL;
  size_t n = 0, cap = 0;
  size_t pos = 0;
  for (size_t c = 0; c < n_chunks; c++) {
    uint32_t tn = chunk_n[c];
    if (pos + tn > n_ids) {
      free(buf);
      return -1;
    }
    uint8_t *piece = NULL;
    size_t pn = 0;
    if (bpe_decode(v, ids + pos, tn, &piece, &pn) != 0) {
      free(buf);
      return -1;
    }
    if (n + pn > cap) {
      size_t nc = cap ? cap * 2 : 4096;
      while (nc < n + pn)
        nc *= 2;
      uint8_t *nb = (uint8_t *)realloc(buf, nc);
      if (!nb) {
        free(piece);
        free(buf);
        return -1;
      }
      buf = nb;
      cap = nc;
    }
    memcpy(buf + n, piece, pn);
    n += pn;
    free(piece);
    pos += tn;
  }
  if (pos != n_ids) {
    free(buf);
    return -1;
  }
  *bytes_out = buf ? buf : (uint8_t *)malloc(1);
  *n_out = n;
  return 0;
}

int bpe_write_tokens(const char *path, const uint16_t *ids, size_t n) {
  return write_all(path, ids, n * sizeof(uint16_t));
}

int bpe_read_tokens(const char *path, uint16_t **ids_out, size_t *n_out) {
  size_t nbytes = 0;
  uint8_t *raw = read_all(path, &nbytes);
  if (!raw)
    return -1;
  if (nbytes % 2) {
    free(raw);
    return -1;
  }
  *n_out = nbytes / 2;
  *ids_out = (uint16_t *)raw;
  return 0;
}

int bpe_write_chunks(const char *path, size_t chunk_bytes,
                     const uint32_t *chunk_n, size_t n_chunks) {
  size_t mlen = strlen(CHUNK_MAGIC);
  size_t total = mlen + 4 + 4 + n_chunks * 4;
  uint8_t *buf = (uint8_t *)malloc(total);
  if (!buf)
    return -1;
  memcpy(buf, CHUNK_MAGIC, mlen);
  size_t o = mlen;
  wr_u32le(buf + o, (uint32_t)n_chunks);
  o += 4;
  wr_u32le(buf + o, (uint32_t)chunk_bytes);
  o += 4;
  for (size_t i = 0; i < n_chunks; i++) {
    wr_u32le(buf + o, chunk_n[i]);
    o += 4;
  }
  int rc = write_all(path, buf, o);
  free(buf);
  return rc;
}

int bpe_read_chunks(const char *path, size_t *chunk_bytes,
                    uint32_t **chunk_n_out, size_t *n_chunks) {
  size_t n = 0;
  uint8_t *buf = read_all(path, &n);
  if (!buf)
    return -1;
  size_t mlen = strlen(CHUNK_MAGIC);
  if (n < mlen + 8 || memcmp(buf, CHUNK_MAGIC, mlen) != 0) {
    free(buf);
    return -1;
  }
  size_t o = mlen;
  uint32_t nc = rd_u32le(buf + o);
  o += 4;
  uint32_t cb = rd_u32le(buf + o);
  o += 4;
  if (o + (size_t)nc * 4 != n) {
    free(buf);
    return -1;
  }
  uint32_t *counts = (uint32_t *)malloc((size_t)nc * sizeof(uint32_t));
  if (!counts && nc) {
    free(buf);
    return -1;
  }
  for (uint32_t i = 0; i < nc; i++) {
    counts[i] = rd_u32le(buf + o);
    o += 4;
  }
  free(buf);
  *chunk_bytes = cb;
  *chunk_n_out = counts;
  *n_chunks = nc;
  return 0;
}

static void bpe_paths(const char *out_prefix, uint32_t vocab_size, char *tok,
                      size_t tok_sz, char *dict, size_t dict_sz, char *chunks,
                      size_t chunks_sz, char *meta, size_t meta_sz) {
  if (vocab_size == BPE_DEFAULT_VOCAB)
    snprintf(tok, tok_sz, "%s.bpe16384", out_prefix);
  else
    snprintf(tok, tok_sz, "%s.bpe%u", out_prefix, vocab_size);
  snprintf(dict, dict_sz, "%s.dict", tok);
  snprintf(chunks, chunks_sz, "%s.chunks", tok);
  snprintf(meta, meta_sz, "%s.json", tok);
}

int bpe_prepare_file(const char *src_path, const char *out_prefix,
                     uint32_t vocab_size, size_t max_train_bytes,
                     size_t chunk_bytes, int verify, int mode) {
  char tok_path[4096], dict_path[4096], chunks_path[4096], meta_path[4096];
  bpe_paths(out_prefix, vocab_size, tok_path, sizeof(tok_path), dict_path,
            sizeof(dict_path), chunks_path, sizeof(chunks_path), meta_path,
            sizeof(meta_path));

  size_t n = 0;
  uint8_t *data = NULL;
  BpeVocab v;
  memset(&v, 0, sizeof(v));

  if (mode == BPE_MODE_ENCODE_ONLY) {
    if (bpe_load_dict(dict_path, &v) != 0) {
      fprintf(stderr, "[bpe] encode-only: missing dict %s\n", dict_path);
      return -1;
    }
    data = read_all(src_path, &n);
    if (!data) {
      perror(src_path);
      bpe_vocab_free(&v);
      return -1;
    }
    {
      size_t file_n = n;
      int st = product_seal_strip_m3_side(data, &n);
      if (st < 0) {
        fprintf(stderr, "[bpe] malformed M3 side trailer on %s\n", src_path);
        free(data);
        bpe_vocab_free(&v);
        return -1;
      }
      if (st == 1)
        fprintf(stderr, "[bpe] stripped M3 side trailer (%zu → %zu B)\n", file_n,
                n);
    }
    fprintf(stderr, "[bpe] encode-only vocab=%u src=%s (%zu B) dict=%s\n",
            v.vocab_size, src_path, n, dict_path);
  } else {
    data = read_all(src_path, &n);
    if (!data) {
      perror(src_path);
      return -1;
    }
    {
      size_t file_n = n;
      int st = product_seal_strip_m3_side(data, &n);
      if (st < 0) {
        fprintf(stderr, "[bpe] malformed M3 side trailer on %s\n", src_path);
        free(data);
        return -1;
      }
      if (st == 1)
        fprintf(stderr, "[bpe] stripped M3 side trailer (%zu → %zu B)\n", file_n,
                n);
    }
    fprintf(stderr, "[bpe] train vocab=%u on %s (%zu B, train_cap=%zu)\n",
            vocab_size, src_path, n, max_train_bytes);
    if (bpe_train(data, n, vocab_size, max_train_bytes, &v) != 0) {
      free(data);
      return -1;
    }
    /* Always persist dict immediately after train (H100 can encode later). */
    if (bpe_save_dict(&v, dict_path) != 0) {
      bpe_vocab_free(&v);
      free(data);
      return -1;
    }
    fprintf(stderr, "[bpe] wrote dict %s (n_merges=%u)\n", dict_path, v.n_merges);
    if (mode == BPE_MODE_DICT_ONLY) {
      FILE *mf = fopen(meta_path, "w");
      if (mf) {
        fprintf(mf,
                "{\n"
                "  \"format\": \"blsmc-bpe-v1\",\n"
                "  \"dict_magic\": \"BBPE1\",\n"
                "  \"vocab_size\": %u,\n"
                "  \"active_vocab\": %u,\n"
                "  \"n_merges\": %u,\n"
                "  \"source_bytes\": %zu,\n"
                "  \"max_train_bytes\": %zu,\n"
                "  \"dict_path\": \"%s\",\n"
                "  \"source_path\": \"%s\",\n"
                "  \"encode_pending\": true\n"
                "}\n",
                v.vocab_size, 256 + v.n_merges, v.n_merges, n, max_train_bytes,
                dict_path, src_path);
        fclose(mf);
      }
      free(data);
      bpe_vocab_free(&v);
      fprintf(stderr, "[bpe] dict-only done — encode on H100 with --encode-only\n");
      return 0;
    }
  }

  uint16_t *ids = NULL;
  size_t n_ids = 0;
  uint32_t *chunk_n = NULL;
  size_t n_chunks = 0;
  fprintf(stderr, "[bpe] encode chunk_bytes=%zu\n", chunk_bytes);
  if (bpe_encode_chunked(&v, data, n, chunk_bytes, &ids, &n_ids, &chunk_n,
                         &n_chunks) != 0) {
    bpe_vocab_free(&v);
    free(data);
    return -1;
  }

  if (verify) {
    uint8_t *back = NULL;
    size_t bn = 0;
    if (bpe_decode_chunked(&v, ids, n_ids, chunk_n, n_chunks, &back, &bn) != 0 ||
        bn != n || memcmp(back, data, n) != 0) {
      fprintf(stderr, "[bpe] round-trip FAILED\n");
      free(back);
      free(ids);
      free(chunk_n);
      bpe_vocab_free(&v);
      free(data);
      return -1;
    }
    free(back);
    fprintf(stderr, "[bpe] round-trip OK\n");
  }
  free(data);

  if (bpe_write_tokens(tok_path, ids, n_ids) != 0 ||
      bpe_write_chunks(chunks_path, chunk_bytes, chunk_n, n_chunks) != 0) {
    free(ids);
    free(chunk_n);
    bpe_vocab_free(&v);
    return -1;
  }

  FILE *mf = fopen(meta_path, "w");
  if (mf) {
    fprintf(mf,
            "{\n"
            "  \"format\": \"blsmc-bpe-v1\",\n"
            "  \"dict_magic\": \"BBPE1\",\n"
            "  \"vocab_size\": %u,\n"
            "  \"active_vocab\": %u,\n"
            "  \"n_merges\": %u,\n"
            "  \"n_tokens\": %zu,\n"
            "  \"token_file_bytes\": %zu,\n"
            "  \"source_bytes\": %zu,\n"
            "  \"chunk_size_bytes\": %zu,\n"
            "  \"n_chunks\": %zu,\n"
            "  \"max_train_bytes\": %zu,\n"
            "  \"token_path\": \"%s\",\n"
            "  \"dict_path\": \"%s\",\n"
            "  \"chunks_path\": \"%s\",\n"
            "  \"source_path\": \"%s\",\n"
            "  \"encode_pending\": false\n"
            "}\n",
            v.vocab_size, 256 + v.n_merges, v.n_merges, n_ids, n_ids * 2, n,
            chunk_bytes, n_chunks, max_train_bytes, tok_path, dict_path,
            chunks_path, src_path);
    fclose(mf);
  }

  fprintf(stderr, "[bpe] wrote %s (%zu tokens, %zu B) dict=%s chunks=%s\n",
          tok_path, n_ids, n_ids * 2, dict_path, chunks_path);

  free(ids);
  free(chunk_n);
  bpe_vocab_free(&v);
  return 0;
}
