/* M3 PHDA9 densify — port of m3_header_densify.py (M3H2 + M3L1). */
#include "m3_densify.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAGIC_H "M3H2"
#define MAGIC_L "M3L1"
#define OP_REV 0x01
#define OP_CONT 0x02
#define OP_ECONT 0x03
#define OP_PDELTA 0x04
#define OP_ID 0x05
#define OP_TS 0x06
#define OP_USER 0x07
#define OP_USER_LIT 0x08
#define OP_IP 0x09
#define OP_RAW 0x0A
#define OP_ID_DELTA 0x0B
#define OP_UID 0x0C

typedef struct {
  uint8_t *p;
  size_t n, cap;
} Buf;

typedef struct {
  const uint8_t *p;
  size_t n;
} Slice;

typedef struct {
  const uint8_t *p;
  uint32_t len;
  uint32_t count;
  uint32_t seq;
  int used;
} StrSlot;

typedef struct {
  uint64_t uid;
  uint32_t count;
  uint32_t seq;
  int used;
} UidSlot;

typedef struct {
  StrSlot *slots;
  size_t cap, fill;
  uint32_t next_seq;
} StrMap;

typedef struct {
  UidSlot *slots;
  size_t cap, fill;
  uint32_t next_seq;
} UidMap;

static void die(const char *msg) {
  fprintf(stderr, "m3_densify: %s\n", msg);
}

static int buf_reserve(Buf *b, size_t need) {
  if (need <= b->cap)
    return 0;
  size_t cap = b->cap ? b->cap : 4096;
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

static int buf_putc(Buf *b, uint8_t c) {
  return buf_put(b, &c, 1);
}

static uint32_t fnv1a(const uint8_t *p, size_t n) {
  uint32_t h = 2166136261u;
  for (size_t i = 0; i < n; i++) {
    h ^= p[i];
    h *= 16777619u;
  }
  return h;
}

static int str_eq(const uint8_t *a, size_t an, const uint8_t *b, size_t bn) {
  return an == bn && (an == 0 || memcmp(a, b, an) == 0);
}

static int strmap_init(StrMap *m, size_t cap) {
  memset(m, 0, sizeof(*m));
  m->cap = cap;
  m->slots = (StrSlot *)calloc(cap, sizeof(StrSlot));
  return m->slots ? 0 : -1;
}

static void strmap_free(StrMap *m) {
  free(m->slots);
  memset(m, 0, sizeof(*m));
}

static StrSlot *strmap_find(StrMap *m, const uint8_t *p, uint32_t len,
                            int create) {
  uint32_t h = fnv1a(p, len);
  size_t i = h & (m->cap - 1);
  for (;;) {
    StrSlot *s = &m->slots[i];
    if (!s->used) {
      if (!create)
        return NULL;
      if (m->fill * 10 >= m->cap * 7)
        return NULL; /* caller should grow — we size generously */
      s->used = 1;
      s->p = p;
      s->len = len;
      s->count = 0;
      s->seq = m->next_seq++;
      m->fill++;
      return s;
    }
    if (str_eq(s->p, s->len, p, len))
      return s;
    i = (i + 1) & (m->cap - 1);
  }
}

static int uidmap_init(UidMap *m, size_t cap) {
  memset(m, 0, sizeof(*m));
  m->cap = cap;
  m->slots = (UidSlot *)calloc(cap, sizeof(UidSlot));
  return m->slots ? 0 : -1;
}

static void uidmap_free(UidMap *m) {
  free(m->slots);
  memset(m, 0, sizeof(*m));
}

static UidSlot *uidmap_find(UidMap *m, uint64_t uid, int create) {
  uint32_t h = (uint32_t)(uid ^ (uid >> 33)) * 0x9e3779b9u;
  size_t i = h & (m->cap - 1);
  for (;;) {
    UidSlot *s = &m->slots[i];
    if (!s->used) {
      if (!create)
        return NULL;
      s->used = 1;
      s->uid = uid;
      s->count = 0;
      s->seq = m->next_seq++;
      m->fill++;
      return s;
    }
    if (s->uid == uid)
      return s;
    i = (i + 1) & (m->cap - 1);
  }
}

static int buf_uvarint(Buf *b, uint64_t n) {
  uint8_t tmp[10];
  int k = 0;
  while (n > 0x7f) {
    tmp[k++] = (uint8_t)((n & 0x7f) | 0x80);
    n >>= 7;
  }
  tmp[k++] = (uint8_t)(n & 0x7f);
  return buf_put(b, tmp, (size_t)k);
}

static int read_uvarint(const uint8_t *buf, size_t n, size_t *off,
                        uint64_t *out) {
  uint64_t v = 0;
  unsigned shift = 0;
  while (*off < n) {
    uint8_t c = buf[(*off)++];
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

static uint64_t zz_encode(int64_t d) {
  if (d < 0)
    return ((uint64_t)(-d) << 1) | 1ull;
  return (uint64_t)d << 1;
}

static int64_t zz_decode(uint64_t zz) {
  if (zz & 1ull)
    return -(int64_t)(zz >> 1);
  return (int64_t)(zz >> 1);
}

static int is_digit(uint8_t c) { return c >= '0' && c <= '9'; }

static int line_is_digits(const uint8_t *p, size_t n) {
  if (n == 0)
    return 0;
  for (size_t i = 0; i < n; i++)
    if (!is_digit(p[i]))
      return 0;
  return 1;
}

static int64_t parse_i64(const uint8_t *p, size_t n) {
  size_t i = 0;
  int neg = 0;
  if (i < n && p[i] == '-') {
    neg = 1;
    i++;
  }
  int64_t v = 0;
  for (; i < n; i++)
    v = v * 10 + (p[i] - '0');
  return neg ? -v : v;
}

static uint64_t parse_u64(const uint8_t *p, size_t n) {
  uint64_t v = 0;
  for (size_t i = 0; i < n; i++)
    v = v * 10 + (uint64_t)(p[i] - '0');
  return v;
}

static int starts_with(const uint8_t *p, size_t n, const char *s) {
  size_t L = strlen(s);
  return n >= L && memcmp(p, s, L) == 0;
}

static int eq_str(const uint8_t *p, size_t n, const char *s) {
  size_t L = strlen(s);
  return n == L && memcmp(p, s, L) == 0;
}

/* Split on \\n; drop trailing empty from final split. */
static int split_lines(const uint8_t *data, size_t n, Slice **out, size_t *nout) {
  size_t count = 0;
  for (size_t i = 0; i < n; i++)
    if (data[i] == '\n')
      count++;
  if (n > 0 && data[n - 1] != '\n')
    count++;
  Slice *lines = (Slice *)calloc(count + 1, sizeof(Slice));
  if (!lines)
    return -1;
  size_t k = 0, start = 0;
  for (size_t i = 0; i < n; i++) {
    if (data[i] == '\n') {
      lines[k].p = data + start;
      lines[k].n = i - start;
      k++;
      start = i + 1;
    }
  }
  if (start < n) {
    lines[k].p = data + start;
    lines[k].n = n - start;
    k++;
  }
  if (k > 0 && lines[k - 1].n == 0)
    k--;
  *out = lines;
  *nout = k;
  return 0;
}

static int parse_ts(const uint8_t *ln, size_t n, int *y, int *d, unsigned *s) {
  static const char pref[] = "timestamp>";
  const size_t plen = sizeof(pref) - 1;
  if (n < plen || memcmp(ln, pref, plen) != 0)
    return 0;
  const uint8_t *rest = ln + plen;
  size_t rn = n - plen;
  size_t colon = (size_t)-1;
  for (size_t i = 0; i < rn; i++)
    if (rest[i] == ':') {
      colon = i;
      break;
    }
  if (colon == (size_t)-1 || colon < 2)
    return 0;
  *y = (rest[0] - '0') * 10 + (rest[1] - '0');
  *d = 0;
  for (size_t i = 2; i < colon; i++) {
    if (!is_digit(rest[i]))
      return 0;
    *d = *d * 10 + (rest[i] - '0');
  }
  unsigned sec = 0;
  for (size_t i = colon + 1; i < rn; i++) {
    if (!is_digit(rest[i]))
      return 0;
    sec = sec * 10u + (unsigned)(rest[i] - '0');
  }
  *s = sec;
  return 1;
}

static int cmp_str_count_desc(const void *a, const void *b) {
  const StrSlot *x = *(const StrSlot *const *)a;
  const StrSlot *y = *(const StrSlot *const *)b;
  if (x->count != y->count)
    return x->count > y->count ? -1 : 1;
  if (x->seq != y->seq)
    return x->seq < y->seq ? -1 : 1;
  return 0;
}

static int cmp_uid_count_desc(const void *a, const void *b) {
  const UidSlot *x = *(const UidSlot *const *)a;
  const UidSlot *y = *(const UidSlot *const *)b;
  if (x->count != y->count)
    return x->count > y->count ? -1 : 1;
  if (x->seq != y->seq)
    return x->seq < y->seq ? -1 : 1;
  return 0;
}

static int densify_header(const uint8_t *header, size_t hn, Buf *dense,
                          Buf *side) {
  Slice *lines = NULL;
  size_t nlines = 0;
  if (split_lines(header, hn, &lines, &nlines) != 0)
    return -1;

  StrMap users;
  UidMap uids;
  StrMap user_idx;
  UidMap uid_idx;
  StrSlot **urank = NULL;
  UidSlot **idrank = NULL;
  memset(&user_idx, 0, sizeof(user_idx));
  memset(&uid_idx, 0, sizeof(uid_idx));
  /* power-of-two caps */
  if (strmap_init(&users, 1u << 18) != 0 || uidmap_init(&uids, 1u << 16) != 0) {
    free(lines);
    return -1;
  }

  int expect = 0; /* 0 none, 1 rev, 2 user */
  for (size_t i = 0; i < nlines; i++) {
    Slice ln = lines[i];
    if (eq_str(ln.p, ln.n, "revision>")) {
      expect = 1;
    } else if (starts_with(ln.p, ln.n, "username>")) {
      const uint8_t *name = ln.p + 9;
      uint32_t nlen = (uint32_t)(ln.n - 9);
      StrSlot *s = strmap_find(&users, name, nlen, 1);
      if (!s) {
        die("username map full");
        goto fail;
      }
      s->count++;
      expect = 2;
    } else if (starts_with(ln.p, ln.n, "ip>")) {
      expect = 0;
    } else if (starts_with(ln.p, ln.n, "id>") &&
               line_is_digits(ln.p + 3, ln.n > 3 ? ln.n - 3 : 0)) {
      if (expect == 2) {
        uint64_t id = parse_u64(ln.p + 3, ln.n - 3);
        UidSlot *s = uidmap_find(&uids, id, 1);
        if (!s) {
          die("uid map full");
          goto fail;
        }
        s->count++;
      }
      expect = 0;
    } else if (eq_str(ln.p, ln.n, "contributor>") ||
               eq_str(ln.p, ln.n, "/contributor>") ||
               (ln.n > 0 && ln.p[0] == '>')) {
      expect = 0;
    }
  }

  urank = (StrSlot **)calloc(users.fill ? users.fill : 1, sizeof(StrSlot *));
  idrank = (UidSlot **)calloc(uids.fill ? uids.fill : 1, sizeof(UidSlot *));
  if (!urank || !idrank)
    goto fail;
  size_t nu = 0, ni = 0;
  for (size_t i = 0; i < users.cap; i++)
    if (users.slots[i].used && users.slots[i].count >= 2)
      urank[nu++] = &users.slots[i];
  for (size_t i = 0; i < uids.cap; i++)
    if (uids.slots[i].used && uids.slots[i].count >= 2)
      idrank[ni++] = &uids.slots[i];
  qsort(urank, nu, sizeof(StrSlot *), cmp_str_count_desc);
  qsort(idrank, ni, sizeof(UidSlot *), cmp_uid_count_desc);
  if (nu > 255)
    nu = 255;
  if (ni > 255)
    ni = 255;

  if (strmap_init(&user_idx, 512) != 0 || uidmap_init(&uid_idx, 512) != 0)
    goto fail;
  for (size_t i = 0; i < nu; i++) {
    StrSlot *s = strmap_find(&user_idx, urank[i]->p, urank[i]->len, 1);
    if (!s)
      goto fail;
    s->count = (uint32_t)i; /* reuse count as index */
  }
  for (size_t i = 0; i < ni; i++) {
    UidSlot *s = uidmap_find(&uid_idx, idrank[i]->uid, 1);
    if (!s)
      goto fail;
    s->count = (uint32_t)i;
  }

  expect = 0;
  int64_t last_rev = 0;
  int have_rev = 0;
  for (size_t i = 0; i < nlines; i++) {
    Slice ln = lines[i];
    if (eq_str(ln.p, ln.n, "revision>")) {
      if (buf_putc(dense, OP_REV) != 0)
        goto fail;
      expect = 1;
    } else if (eq_str(ln.p, ln.n, "contributor>")) {
      if (buf_putc(dense, OP_CONT) != 0)
        goto fail;
      expect = 0;
    } else if (eq_str(ln.p, ln.n, "/contributor>")) {
      if (buf_putc(dense, OP_ECONT) != 0)
        goto fail;
      expect = 0;
    } else if (ln.n > 1 && ln.p[0] == '>' &&
               (is_digit(ln.p[1]) ||
                (ln.p[1] == '-' && ln.n > 2 && is_digit(ln.p[2])))) {
      int64_t delta = parse_i64(ln.p + 1, ln.n - 1);
      if (buf_putc(dense, OP_PDELTA) != 0 ||
          buf_uvarint(dense, zz_encode(delta)) != 0)
        goto fail;
      expect = 0;
    } else if (starts_with(ln.p, ln.n, "id>") &&
               line_is_digits(ln.p + 3, ln.n > 3 ? ln.n - 3 : 0)) {
      uint64_t id = parse_u64(ln.p + 3, ln.n - 3);
      if (expect == 1) {
        if (!have_rev) {
          if (buf_putc(dense, OP_ID) != 0 || buf_uvarint(dense, id) != 0)
            goto fail;
        } else {
          int64_t d = (int64_t)id - last_rev;
          if (buf_putc(dense, OP_ID_DELTA) != 0 ||
              buf_uvarint(dense, zz_encode(d)) != 0)
            goto fail;
        }
        last_rev = (int64_t)id;
        have_rev = 1;
      } else if (expect == 2) {
        UidSlot *s = uidmap_find(&uid_idx, id, 0);
        if (s) {
          if (buf_putc(dense, OP_UID) != 0 ||
              buf_putc(dense, (uint8_t)s->count) != 0)
            goto fail;
        } else {
          if (buf_putc(dense, OP_ID) != 0 || buf_uvarint(dense, id) != 0)
            goto fail;
        }
      } else {
        if (buf_putc(dense, OP_ID) != 0 || buf_uvarint(dense, id) != 0)
          goto fail;
      }
      expect = 0;
    } else {
      int y, d;
      unsigned sec;
      if (parse_ts(ln.p, ln.n, &y, &d, &sec)) {
        if (y > 255 || d > 65535) {
          if (ln.n > 0xffff)
            goto fail;
          uint8_t hdr[3] = {OP_RAW, (uint8_t)(ln.n & 0xff),
                            (uint8_t)((ln.n >> 8) & 0xff)};
          if (buf_put(dense, hdr, 3) != 0 || buf_put(dense, ln.p, ln.n) != 0)
            goto fail;
        } else {
          uint8_t pack[7];
          pack[0] = OP_TS;
          pack[1] = (uint8_t)y;
          pack[2] = (uint8_t)(d & 0xff);
          pack[3] = (uint8_t)((d >> 8) & 0xff);
          pack[4] = (uint8_t)(sec & 0xff);
          pack[5] = (uint8_t)((sec >> 8) & 0xff);
          pack[6] = (uint8_t)((sec >> 16) & 0xff);
          uint8_t pack2 = (uint8_t)((sec >> 24) & 0xff);
          if (buf_put(dense, pack, 7) != 0 || buf_putc(dense, pack2) != 0)
            goto fail;
        }
        expect = 0;
      } else if (starts_with(ln.p, ln.n, "username>")) {
        const uint8_t *name = ln.p + 9;
        size_t nlen = ln.n - 9;
        StrSlot *s = strmap_find(&user_idx, name, (uint32_t)nlen, 0);
        if (s) {
          if (buf_putc(dense, OP_USER) != 0 ||
              buf_putc(dense, (uint8_t)s->count) != 0)
            goto fail;
        } else if (nlen <= 255) {
          if (buf_putc(dense, OP_USER_LIT) != 0 ||
              buf_putc(dense, (uint8_t)nlen) != 0 ||
              buf_put(dense, name, nlen) != 0)
            goto fail;
        } else {
          if (ln.n > 0xffff)
            goto fail;
          uint8_t hdr[3] = {OP_RAW, (uint8_t)(ln.n & 0xff),
                            (uint8_t)((ln.n >> 8) & 0xff)};
          if (buf_put(dense, hdr, 3) != 0 || buf_put(dense, ln.p, ln.n) != 0)
            goto fail;
        }
        expect = 2;
      } else if (starts_with(ln.p, ln.n, "ip>")) {
        const uint8_t *rest = ln.p + 3;
        size_t rn = ln.n - 3;
        int ok = 1;
        unsigned oct[4];
        size_t pos = 0;
        for (int k = 0; k < 4; k++) {
          if (pos >= rn || !is_digit(rest[pos])) {
            ok = 0;
            break;
          }
          unsigned v = 0;
          while (pos < rn && is_digit(rest[pos])) {
            v = v * 10 + (rest[pos] - '0');
            if (v > 255) {
              ok = 0;
              break;
            }
            pos++;
          }
          oct[k] = v;
          if (k < 3) {
            if (pos >= rn || rest[pos] != '.') {
              ok = 0;
              break;
            }
            pos++;
          }
        }
        if (ok && pos == rn) {
          uint8_t ipb[5] = {OP_IP, (uint8_t)oct[0], (uint8_t)oct[1],
                            (uint8_t)oct[2], (uint8_t)oct[3]};
          if (buf_put(dense, ipb, 5) != 0)
            goto fail;
        } else {
          if (ln.n > 0xffff)
            goto fail;
          uint8_t hdr[3] = {OP_RAW, (uint8_t)(ln.n & 0xff),
                            (uint8_t)((ln.n >> 8) & 0xff)};
          if (buf_put(dense, hdr, 3) != 0 || buf_put(dense, ln.p, ln.n) != 0)
            goto fail;
        }
        expect = 0;
      } else {
        if (ln.n > 0xffff)
          goto fail;
        uint8_t hdr[3] = {OP_RAW, (uint8_t)(ln.n & 0xff),
                          (uint8_t)((ln.n >> 8) & 0xff)};
        if (buf_put(dense, hdr, 3) != 0 || buf_put(dense, ln.p, ln.n) != 0)
          goto fail;
        expect = 0;
      }
    }
  }

  if (buf_put(side, MAGIC_H, 4) != 0 || buf_putc(side, (uint8_t)nu) != 0)
    goto fail;
  for (size_t i = 0; i < nu; i++) {
    if (urank[i]->len > 255)
      goto fail;
    if (buf_putc(side, (uint8_t)urank[i]->len) != 0 ||
        buf_put(side, urank[i]->p, urank[i]->len) != 0)
      goto fail;
  }
  if (buf_putc(side, (uint8_t)ni) != 0)
    goto fail;
  for (size_t i = 0; i < ni; i++) {
    if (buf_uvarint(side, idrank[i]->uid) != 0)
      goto fail;
  }

  strmap_free(&user_idx);
  uidmap_free(&uid_idx);
  free(urank);
  free(idrank);
  strmap_free(&users);
  uidmap_free(&uids);
  free(lines);
  return 0;

fail:
  strmap_free(&user_idx);
  uidmap_free(&uid_idx);
  free(urank);
  free(idrank);
  strmap_free(&users);
  uidmap_free(&uids);
  free(lines);
  return -1;
}

static int expand_header(const uint8_t *dense, size_t dn, const uint8_t *side,
                         size_t sn, Buf *out) {
  if (sn < 5 || memcmp(side, MAGIC_H, 4) != 0) {
    die("bad side magic");
    return -1;
  }
  size_t off = 4;
  unsigned nusers = side[off++];
  Slice *dict = (Slice *)calloc(nusers, sizeof(Slice));
  if (!dict)
    return -1;
  for (unsigned i = 0; i < nusers; i++) {
    if (off >= sn)
      goto fail;
    unsigned L = side[off++];
    if (off + L > sn)
      goto fail;
    dict[i].p = side + off;
    dict[i].n = L;
    off += L;
  }
  if (off >= sn)
    goto fail;
  unsigned nuids = side[off++];
  uint64_t *uids = (uint64_t *)calloc(nuids, sizeof(uint64_t));
  if (!uids)
    goto fail;
  for (unsigned i = 0; i < nuids; i++) {
    if (read_uvarint(side, sn, &off, &uids[i]) != 0)
      goto fail2;
  }

  size_t i = 0;
  int64_t last_rev = 0;
  int have_rev = 0;
  int prev_was_rev = 0;
  int first_line = 1;
  while (i < dn) {
    uint8_t op = dense[i++];
    char tmp[64];
    int tlen;
    if (!first_line) {
      if (buf_putc(out, '\n') != 0)
        goto fail2;
    }
    first_line = 0;
    switch (op) {
    case OP_REV:
      if (buf_put(out, "revision>", 9) != 0)
        goto fail2;
      prev_was_rev = 1;
      break;
    case OP_CONT:
      if (buf_put(out, "contributor>", 12) != 0)
        goto fail2;
      prev_was_rev = 0;
      break;
    case OP_ECONT:
      if (buf_put(out, "/contributor>", 13) != 0)
        goto fail2;
      prev_was_rev = 0;
      break;
    case OP_PDELTA: {
      uint64_t zz;
      if (read_uvarint(dense, dn, &i, &zz) != 0)
        goto fail2;
      tlen = snprintf(tmp, sizeof(tmp), ">%lld", (long long)zz_decode(zz));
      if (buf_put(out, tmp, (size_t)tlen) != 0)
        goto fail2;
      prev_was_rev = 0;
      break;
    }
    case OP_ID: {
      uint64_t id;
      if (read_uvarint(dense, dn, &i, &id) != 0)
        goto fail2;
      if (prev_was_rev) {
        last_rev = (int64_t)id;
        have_rev = 1;
      }
      tlen = snprintf(tmp, sizeof(tmp), "id>%llu", (unsigned long long)id);
      if (buf_put(out, tmp, (size_t)tlen) != 0)
        goto fail2;
      prev_was_rev = 0;
      break;
    }
    case OP_ID_DELTA: {
      uint64_t zz;
      if (read_uvarint(dense, dn, &i, &zz) != 0 || !have_rev)
        goto fail2;
      last_rev = last_rev + zz_decode(zz);
      tlen = snprintf(tmp, sizeof(tmp), "id>%lld", (long long)last_rev);
      if (buf_put(out, tmp, (size_t)tlen) != 0)
        goto fail2;
      prev_was_rev = 0;
      break;
    }
    case OP_UID: {
      if (i >= dn)
        goto fail2;
      unsigned idx = dense[i++];
      if (idx >= nuids)
        goto fail2;
      tlen = snprintf(tmp, sizeof(tmp), "id>%llu",
                      (unsigned long long)uids[idx]);
      if (buf_put(out, tmp, (size_t)tlen) != 0)
        goto fail2;
      prev_was_rev = 0;
      break;
    }
    case OP_TS: {
      if (i + 7 > dn)
        goto fail2;
      unsigned y = dense[i++];
      unsigned d = dense[i] | ((unsigned)dense[i + 1] << 8);
      i += 2;
      unsigned sec = dense[i] | ((unsigned)dense[i + 1] << 8) |
                     ((unsigned)dense[i + 2] << 16) |
                     ((unsigned)dense[i + 3] << 24);
      i += 4;
      tlen = snprintf(tmp, sizeof(tmp), "timestamp>%02u%u:%u", y, d, sec);
      if (buf_put(out, tmp, (size_t)tlen) != 0)
        goto fail2;
      prev_was_rev = 0;
      break;
    }
    case OP_USER: {
      if (i >= dn)
        goto fail2;
      unsigned idx = dense[i++];
      if (idx >= nusers)
        goto fail2;
      if (buf_put(out, "username>", 9) != 0 ||
          buf_put(out, dict[idx].p, dict[idx].n) != 0)
        goto fail2;
      prev_was_rev = 0;
      break;
    }
    case OP_USER_LIT: {
      if (i >= dn)
        goto fail2;
      unsigned L = dense[i++];
      if (i + L > dn)
        goto fail2;
      if (buf_put(out, "username>", 9) != 0 || buf_put(out, dense + i, L) != 0)
        goto fail2;
      i += L;
      prev_was_rev = 0;
      break;
    }
    case OP_IP: {
      if (i + 4 > dn)
        goto fail2;
      tlen = snprintf(tmp, sizeof(tmp), "ip>%u.%u.%u.%u", dense[i], dense[i + 1],
                      dense[i + 2], dense[i + 3]);
      i += 4;
      if (buf_put(out, tmp, (size_t)tlen) != 0)
        goto fail2;
      prev_was_rev = 0;
      break;
    }
    case OP_RAW: {
      if (i + 2 > dn)
        goto fail2;
      unsigned L = dense[i] | ((unsigned)dense[i + 1] << 8);
      i += 2;
      if (i + L > dn)
        goto fail2;
      if (buf_put(out, dense + i, L) != 0)
        goto fail2;
      i += L;
      prev_was_rev = 0;
      break;
    }
    default:
      die("bad header opcode");
      goto fail2;
    }
  }
  if (buf_putc(out, '\n') != 0)
    goto fail2;
  free(uids);
  free(dict);
  return 0;
fail2:
  free(uids);
fail:
  free(dict);
  return -1;
}

/* ---- lang pack ---- */

static int is_code_char(uint8_t c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
         (c >= '0' && c <= '9') || c == '-' || c == '_';
}

/* Parse [code:title]suffix — title ends at last ']'. */
static int parse_lang_line(const uint8_t *ln, size_t n, Slice *code, Slice *title,
                           Slice *suffix) {
  if (n < 4 || ln[0] != '[')
    return 0;
  size_t i = 1;
  while (i < n && is_code_char(ln[i]))
    i++;
  if (i == 1 || i >= n || ln[i] != ':')
    return 0;
  size_t code_end = i;
  size_t last_br = (size_t)-1;
  for (size_t j = i + 1; j < n; j++)
    if (ln[j] == ']')
      last_br = j;
  if (last_br == (size_t)-1)
    return 0;
  code->p = ln + 1;
  code->n = code_end - 1;
  title->p = ln + code_end + 1;
  title->n = last_br - (code_end + 1);
  suffix->p = ln + last_br + 1;
  suffix->n = n - (last_br + 1);
  return 1;
}

typedef struct {
  StrSlot *slot;
  uint32_t count;
} CodeRank;

static int cmp_code_rank(const void *a, const void *b) {
  const CodeRank *x = (const CodeRank *)a;
  const CodeRank *y = (const CodeRank *)b;
  if (x->count != y->count)
    return x->count > y->count ? -1 : 1;
  size_t n = x->slot->len < y->slot->len ? x->slot->len : y->slot->len;
  int c = memcmp(x->slot->p, y->slot->p, n);
  if (c != 0)
    return c;
  if (x->slot->len != y->slot->len)
    return x->slot->len < y->slot->len ? -1 : 1;
  return 0;
}

static int pack_lang(const uint8_t *lang, size_t ln, Buf *out) {
  Slice *lines = NULL;
  size_t nlines = 0;
  if (split_lines(lang, ln, &lines, &nlines) != 0)
    return -1;

  StrMap codes;
  if (strmap_init(&codes, 1024) != 0) {
    free(lines);
    return -1;
  }
  for (size_t i = 0; i < nlines; i++) {
    Slice code, title, suf;
    if (parse_lang_line(lines[i].p, lines[i].n, &code, &title, &suf)) {
      StrSlot *s = strmap_find(&codes, code.p, (uint32_t)code.n, 1);
      if (!s) {
        strmap_free(&codes);
        free(lines);
        return -1;
      }
      s->count++;
    }
  }

  CodeRank *rank = (CodeRank *)calloc(codes.fill, sizeof(CodeRank));
  if (!rank) {
    strmap_free(&codes);
    free(lines);
    return -1;
  }
  size_t nr = 0;
  for (size_t i = 0; i < codes.cap; i++)
    if (codes.slots[i].used) {
      rank[nr].slot = &codes.slots[i];
      rank[nr].count = codes.slots[i].count;
      nr++;
    }
  qsort(rank, nr, sizeof(CodeRank), cmp_code_rank);
  if (nr > 0xF0)
    nr = 0xF0;

  StrMap code_id;
  if (strmap_init(&code_id, 512) != 0) {
    free(rank);
    strmap_free(&codes);
    free(lines);
    return -1;
  }
  for (size_t i = 0; i < nr; i++) {
    StrSlot *s = strmap_find(&code_id, rank[i].slot->p, rank[i].slot->len, 1);
    if (!s)
      goto fail;
    s->count = (uint32_t)i;
  }

  if (buf_put(out, MAGIC_L, 4) != 0)
    goto fail;
  uint8_t nle[2] = {(uint8_t)(nr & 0xff), (uint8_t)((nr >> 8) & 0xff)};
  if (buf_put(out, nle, 2) != 0)
    goto fail;
  for (size_t i = 0; i < nr; i++) {
    if (buf_put(out, rank[i].slot->p, rank[i].slot->len) != 0 ||
        buf_putc(out, 0) != 0)
      goto fail;
  }

  size_t i = 0;
  while (i < nlines) {
    Slice code, title, suf;
    if (!parse_lang_line(lines[i].p, lines[i].n, &code, &title, &suf)) {
      if (lines[i].n > 0xffff)
        goto fail;
      uint8_t hdr[3] = {0xFF, (uint8_t)(lines[i].n & 0xff),
                        (uint8_t)((lines[i].n >> 8) & 0xff)};
      if (buf_put(out, hdr, 3) != 0 ||
          buf_put(out, lines[i].p, lines[i].n) != 0)
        goto fail;
      i++;
      continue;
    }
    StrSlot *cs = strmap_find(&code_id, code.p, (uint32_t)code.n, 0);
    if (!cs) {
      if (lines[i].n > 0xffff)
        goto fail;
      uint8_t hdr[3] = {0xFF, (uint8_t)(lines[i].n & 0xff),
                        (uint8_t)((lines[i].n >> 8) & 0xff)};
      if (buf_put(out, hdr, 3) != 0 ||
          buf_put(out, lines[i].p, lines[i].n) != 0)
        goto fail;
      i++;
      continue;
    }
    size_t j = i + 1;
    while (j < nlines) {
      Slice c2, t2, s2;
      if (!parse_lang_line(lines[j].p, lines[j].n, &c2, &t2, &s2))
        break;
      if (!strmap_find(&code_id, c2.p, (uint32_t)c2.n, 0))
        break;
      if (!str_eq(t2.p, t2.n, title.p, title.n) ||
          !str_eq(s2.p, s2.n, suf.p, suf.n))
        break;
      j++;
    }
    size_t L = j - i;
    if (L >= 2) {
      if (buf_putc(out, suf.n ? 0xFD : 0xFE) != 0 ||
          buf_uvarint(out, (uint64_t)L) != 0)
        goto fail;
      for (size_t k = i; k < j; k++) {
        Slice c2, t2, s2;
        parse_lang_line(lines[k].p, lines[k].n, &c2, &t2, &s2);
        StrSlot *id = strmap_find(&code_id, c2.p, (uint32_t)c2.n, 0);
        if (buf_putc(out, (uint8_t)id->count) != 0)
          goto fail;
      }
      if (buf_put(out, title.p, title.n) != 0)
        goto fail;
      if (suf.n) {
        if (buf_putc(out, 0) != 0 || buf_put(out, suf.p, suf.n) != 0)
          goto fail;
      }
      if (buf_putc(out, '\n') != 0)
        goto fail;
    } else {
      uint8_t cid = (uint8_t)cs->count;
      if (!suf.n) {
        if (buf_putc(out, cid) != 0 || buf_put(out, title.p, title.n) != 0 ||
            buf_putc(out, '\n') != 0)
          goto fail;
      } else {
        if (buf_putc(out, 0xF0) != 0 || buf_putc(out, cid) != 0 ||
            buf_put(out, title.p, title.n) != 0 || buf_putc(out, 0) != 0 ||
            buf_put(out, suf.p, suf.n) != 0 || buf_putc(out, '\n') != 0)
          goto fail;
      }
    }
    i = j;
  }

  strmap_free(&code_id);
  free(rank);
  strmap_free(&codes);
  free(lines);
  return 0;
fail:
  strmap_free(&code_id);
  free(rank);
  strmap_free(&codes);
  free(lines);
  return -1;
}

static int expand_lang(const uint8_t *packed, size_t pn, Buf *out) {
  if (pn < 6 || memcmp(packed, MAGIC_L, 4) != 0) {
    return buf_put(out, packed, pn);
  }
  size_t off = 4;
  unsigned ncodes = packed[off] | ((unsigned)packed[off + 1] << 8);
  off += 2;
  Slice *codes = (Slice *)calloc(ncodes, sizeof(Slice));
  if (!codes)
    return -1;
  for (unsigned i = 0; i < ncodes; i++) {
    size_t start = off;
    while (off < pn && packed[off] != 0)
      off++;
    if (off >= pn)
      goto fail;
    codes[i].p = packed + start;
    codes[i].n = off - start;
    off++;
  }
  int first = 1;
  while (off < pn) {
    uint8_t op = packed[off++];
    if (!first) {
      if (buf_putc(out, '\n') != 0)
        goto fail;
    }
    first = 0;
    if (op == 0xFF) {
      if (off + 2 > pn)
        goto fail;
      unsigned L = packed[off] | ((unsigned)packed[off + 1] << 8);
      off += 2;
      if (off + L > pn)
        goto fail;
      if (buf_put(out, packed + off, L) != 0)
        goto fail;
      off += L;
    } else if (op == 0xFE || op == 0xFD) {
      uint64_t n;
      if (read_uvarint(packed, pn, &off, &n) != 0 || off + n > pn)
        goto fail;
      const uint8_t *cids = packed + off;
      off += (size_t)n;
      Slice title, suf;
      suf.p = NULL;
      suf.n = 0;
      if (op == 0xFE) {
        size_t end = off;
        while (end < pn && packed[end] != '\n')
          end++;
        if (end >= pn)
          goto fail;
        title.p = packed + off;
        title.n = end - off;
        off = end + 1;
      } else {
        size_t z = off;
        while (z < pn && packed[z] != 0)
          z++;
        if (z >= pn)
          goto fail;
        title.p = packed + off;
        title.n = z - off;
        off = z + 1;
        size_t end = off;
        while (end < pn && packed[end] != '\n')
          end++;
        if (end >= pn)
          goto fail;
        suf.p = packed + off;
        suf.n = end - off;
        off = end + 1;
      }
      for (uint64_t k = 0; k < n; k++) {
        if (k && buf_putc(out, '\n') != 0)
          goto fail;
        unsigned cid = cids[k];
        if (cid >= ncodes)
          goto fail;
        if (buf_putc(out, '[') != 0 ||
            buf_put(out, codes[cid].p, codes[cid].n) != 0 ||
            buf_putc(out, ':') != 0 || buf_put(out, title.p, title.n) != 0 ||
            buf_putc(out, ']') != 0 || buf_put(out, suf.p, suf.n) != 0)
          goto fail;
      }
    } else if (op == 0xF0) {
      if (off >= pn)
        goto fail;
      unsigned cid = packed[off++];
      if (cid >= ncodes)
        goto fail;
      size_t z = off;
      while (z < pn && packed[z] != 0)
        z++;
      if (z >= pn)
        goto fail;
      Slice title = {packed + off, z - off};
      off = z + 1;
      size_t end = off;
      while (end < pn && packed[end] != '\n')
        end++;
      if (end >= pn)
        goto fail;
      Slice suf = {packed + off, end - off};
      off = end + 1;
      if (buf_putc(out, '[') != 0 || buf_put(out, codes[cid].p, codes[cid].n) != 0 ||
          buf_putc(out, ':') != 0 || buf_put(out, title.p, title.n) != 0 ||
          buf_putc(out, ']') != 0 || buf_put(out, suf.p, suf.n) != 0)
        goto fail;
    } else if (op < 0xF0) {
      size_t end = off;
      while (end < pn && packed[end] != '\n')
        end++;
      if (end >= pn)
        goto fail;
      Slice title = {packed + off, end - off};
      off = end + 1;
      if (op >= ncodes)
        goto fail;
      if (buf_putc(out, '[') != 0 || buf_put(out, codes[op].p, codes[op].n) != 0 ||
          buf_putc(out, ':') != 0 || buf_put(out, title.p, title.n) != 0 ||
          buf_putc(out, ']') != 0)
        goto fail;
    } else {
      goto fail;
    }
  }
  if (buf_putc(out, '\n') != 0)
    goto fail;
  free(codes);
  return 0;
fail:
  free(codes);
  return -1;
}

static int parse_phda9(const uint8_t *data, size_t n, size_t *prefix_n,
                       size_t *hs, size_t *ls, const uint8_t **header,
                       const uint8_t **lang, const uint8_t **body,
                       size_t *body_n) {
  size_t i = 0;
  while (i < n && i < 20 && is_digit(data[i]))
    i++;
  if (i == 0)
    return -1;
  size_t tail_len = 0;
  for (size_t k = 0; k < i; k++)
    tail_len = tail_len * 10 + (data[k] - '0');
  if (tail_len > n - i)
    return -1;
  *prefix_n = i;
  *body = data + i;
  *body_n = n - i - tail_len;
  const uint8_t *tail = data + n - tail_len;
  size_t nl1 = 0;
  while (nl1 < tail_len && tail[nl1] != '\n')
    nl1++;
  if (nl1 >= tail_len)
    return -1;
  size_t hsz = 0;
  for (size_t k = 0; k < nl1; k++)
    hsz = hsz * 10 + (tail[k] - '0');
  const uint8_t *rest = tail + nl1 + 1;
  size_t rest_n = tail_len - nl1 - 1;
  size_t nl2 = 0;
  while (nl2 < rest_n && rest[nl2] != '\n')
    nl2++;
  if (nl2 >= rest_n)
    return -1;
  size_t lsz = 0;
  for (size_t k = 0; k < nl2; k++)
    lsz = lsz * 10 + (rest[k] - '0');
  const uint8_t *after = rest + nl2 + 1;
  if (hsz + lsz > rest_n - nl2 - 1)
    return -1;
  *hs = hsz;
  *ls = lsz;
  *header = after;
  *lang = after + hsz;
  return 0;
}

static int rebuild_phda9(const uint8_t *body, size_t body_n,
                         const uint8_t *header, size_t hs, const uint8_t *lang,
                         size_t ls, Buf *out) {
  char size_lines[64];
  int sl = snprintf(size_lines, sizeof(size_lines), "%zu\n%zu\n", hs, ls);
  size_t tail_n = (size_t)sl + hs + ls;
  char digits[32];
  int dl = snprintf(digits, sizeof(digits), "%zu", tail_n);
  if (buf_put(out, digits, (size_t)dl) != 0 ||
      buf_put(out, body, body_n) != 0 || buf_put(out, size_lines, (size_t)sl) != 0 ||
      buf_put(out, header, hs) != 0 || buf_put(out, lang, ls) != 0)
    return -1;
  return 0;
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

int m3_densify_phda9(const uint8_t *in, size_t in_n, uint8_t **out_data,
                     size_t *out_n, uint8_t **side_out, size_t *side_n) {
  size_t prefix_n, hs, ls, body_n;
  const uint8_t *header, *lang, *body;
  if (parse_phda9(in, in_n, &prefix_n, &hs, &ls, &header, &lang, &body,
                  &body_n) != 0) {
    die("bad PHDA9");
    return -1;
  }

  Buf hbuf = {0}, lbuf = {0};
  /* Ensure trailing newlines like Python */
  if (buf_put(&hbuf, header, hs) != 0)
    return -1;
  if (hs == 0 || header[hs - 1] != '\n') {
    if (buf_putc(&hbuf, '\n') != 0)
      return -1;
  }
  if (buf_put(&lbuf, lang, ls) != 0)
    return -1;
  if (ls > 0 && lang[ls - 1] != '\n') {
    if (buf_putc(&lbuf, '\n') != 0)
      return -1;
  }

  Buf dense = {0}, side = {0}, packed = {0};
  Buf back_h = {0}, back_l = {0}, rebuilt = {0};

  if (densify_header(hbuf.p, hbuf.n, &dense, &side) != 0) {
    die("densify_header failed");
    goto fail;
  }
  if (expand_header(dense.p, dense.n, side.p, side.n, &back_h) != 0) {
    die("expand_header failed");
    goto fail;
  }
  if (back_h.n != hbuf.n || memcmp(back_h.p, hbuf.p, hbuf.n) != 0) {
    die("header round-trip mismatch");
    goto fail;
  }
  if (pack_lang(lbuf.p, lbuf.n, &packed) != 0) {
    die("pack_lang failed");
    goto fail;
  }
  if (expand_lang(packed.p, packed.n, &back_l) != 0) {
    die("expand_lang failed");
    goto fail;
  }
  if (back_l.n != lbuf.n || memcmp(back_l.p, lbuf.p, lbuf.n) != 0) {
    die("lang round-trip mismatch");
    goto fail;
  }
  if (rebuild_phda9(body, body_n, dense.p, dense.n, packed.p, packed.n,
                    &rebuilt) != 0)
    goto fail;

  *out_data = rebuilt.p;
  *out_n = rebuilt.n;
  *side_out = side.p;
  *side_n = side.n;
  rebuilt.p = NULL;
  side.p = NULL;

  free(hbuf.p);
  free(lbuf.p);
  free(dense.p);
  free(packed.p);
  free(back_h.p);
  free(back_l.p);
  free(rebuilt.p);
  free(side.p);
  return 0;
fail:
  free(hbuf.p);
  free(lbuf.p);
  free(dense.p);
  free(side.p);
  free(packed.p);
  free(back_h.p);
  free(back_l.p);
  free(rebuilt.p);
  return -1;
}

int m3_densify_file(const char *path) {
  size_t n = 0;
  uint8_t *in = read_file(path, &n);
  if (!in) {
    perror(path);
    return -1;
  }
  uint8_t *out = NULL, *side = NULL;
  size_t out_n = 0, side_n = 0;
  int rc = m3_densify_phda9(in, n, &out, &out_n, &side, &side_n);
  free(in);
  if (rc != 0)
    return rc;

  char out_path[4096], side_path[4096];
  snprintf(out_path, sizeof(out_path), "%s.dense", path);
  snprintf(side_path, sizeof(side_path), "%s.dense.side", path);
  if (write_file(out_path, out, out_n) != 0 ||
      write_file(side_path, side, side_n) != 0) {
    perror("write dense");
    free(out);
    free(side);
    return -1;
  }
  fprintf(stderr,
          "[M3] densify C: header+lang → %zu B (+ side %zu)  wrote %s\n", out_n,
          side_n, out_path);
  free(out);
  free(side);
  return 0;
}

int m3_undensify_phda9(const uint8_t *dense_in, size_t dense_n,
                       const uint8_t *side, size_t side_n,
                       uint8_t **out_data, size_t *out_n) {
  size_t prefix_n, hs, ls, body_n;
  const uint8_t *header, *lang, *body;
  if (!dense_in || !out_data || !out_n)
    return -1;
  if (parse_phda9(dense_in, dense_n, &prefix_n, &hs, &ls, &header, &lang, &body,
                  &body_n) != 0) {
    die("undensify: bad dense PHDA9");
    return -1;
  }
  Buf back_h = {0}, back_l = {0}, rebuilt = {0};
  if (expand_header(header, hs, side, side_n, &back_h) != 0) {
    die("undensify: expand_header failed");
    goto fail;
  }
  if (expand_lang(lang, ls, &back_l) != 0) {
    die("undensify: expand_lang failed");
    goto fail;
  }
  if (rebuild_phda9(body, body_n, back_h.p, back_h.n, back_l.p, back_l.n,
                    &rebuilt) != 0) {
    die("undensify: rebuild_phda9 failed");
    goto fail;
  }
  *out_data = rebuilt.p;
  *out_n = rebuilt.n;
  rebuilt.p = NULL;
  free(back_h.p);
  free(back_l.p);
  return 0;
fail:
  free(back_h.p);
  free(back_l.p);
  free(rebuilt.p);
  return -1;
}

int m3_undensify_file(const char *dense_path, const char *side_path,
                      const char *out_path) {
  size_t dn = 0, sn = 0;
  uint8_t *dense = read_file(dense_path, &dn);
  if (!dense) {
    perror(dense_path);
    return -1;
  }
  uint8_t *side = read_file(side_path, &sn);
  if (!side) {
    perror(side_path);
    free(dense);
    return -1;
  }
  uint8_t *out = NULL;
  size_t out_n = 0;
  int rc = m3_undensify_phda9(dense, dn, side, sn, &out, &out_n);
  free(dense);
  free(side);
  if (rc != 0)
    return rc;
  char default_out[4096];
  if (!out_path) {
    snprintf(default_out, sizeof(default_out), "%s.raw", dense_path);
    out_path = default_out;
  }
  if (write_file(out_path, out, out_n) != 0) {
    perror(out_path);
    free(out);
    return -1;
  }
  fprintf(stderr, "[M3] undensify: %zu B → %zu B  wrote %s\n", dn, out_n,
          out_path);
  free(out);
  return 0;
}
