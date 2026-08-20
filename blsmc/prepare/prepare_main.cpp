/*
 * blsmc_prepare — full blsmc preprocess (M1–M5).
 *
 * End-to-end from raw enwik9:
 *   M1 split4Comp → M2 reorder → M3 PHDA9 + densify → M4 WRT
 *   M5 payload_sim (struct + SimHash reorder).
 *
 * Compared against cmix-lex bars: ready 934,220,701 ; post-WRT 586,459,321 ;
 * post–payload_lex 587,138,826.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <unistd.h>  /* chdir, setenv */

#include "misc.h"
#include "phda9_preprocess.h"
#include "article_reorder.h"

#include "preprocess/preprocessor.h"
#include "r1_reorder_transform.h"

extern "C" {
#include "m3_densify.h"
#include "m5_payload_sim.h"
#include "product_seal.h"
#include "bpe_tokenizer.h"
}

namespace {

constexpr size_t kPostWrt = 586459321ull;
constexpr size_t kPostPayloadLex = 587138826ull;
constexpr size_t kReadyCert = 934220701ull; /* changes.md dictionary-decoded */
/* Official Hutter enwik9 M1 peel sizes (split4Comp line cuts → bytes). */
constexpr size_t kEnwik9IntroBytes = 1404ull;
constexpr size_t kEnwik9CodaBytes = 9745ull;

size_t FileSize(const std::string& path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in.is_open())
    return 0;
  return static_cast<size_t>(in.tellg());
}

bool CopyFile(const std::string& from, const std::string& to) {
  std::ifstream in(from, std::ios::binary);
  std::ofstream out(to, std::ios::binary | std::ios::trunc);
  if (!in.is_open() || !out.is_open())
    return false;
  std::vector<char> buf(1 << 20);
  while (in) {
    in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
    auto got = in.gcount();
    if (got > 0)
      out.write(buf.data(), got);
  }
  return static_cast<bool>(out);
}

void Checkpoint(const char* stage, size_t bytes, size_t expect) {
  long long d = (long long)bytes - (long long)expect;
  std::fprintf(stderr, "  %-22s %12zu B", stage, bytes);
  if (expect) {
    std::fprintf(stderr, "  (bar %zu  %+lld)%s", expect, d,
        bytes == expect ? " OK" : "");
  }
  std::fprintf(stderr, "\n");
}

void Usage(const char* argv0) {
  std::fprintf(stderr,
      "usage:\n"
      "  %s encode <enwik9> <out_587> [--workdir DIR] [--dict PATH] [--order PATH]\n"
      "  %s encode-from-ready <ready4cmix> <out_587> [--dict PATH]\n"
      "  %s decode <payload_sim> <out_enwik9> [--dict PATH] [--workdir DIR]\n"
      "           [--enwik9 REF | --intro-bytes N --coda-bytes N]\n"
      "           [--expect REF]   # optional byte-compare vs REF\n"
      "           # sizes: CLI > BLSMETA1 in product > enwik9 defaults\n"
      "  %s densify <phda9prepr>          # M3 header+lang densify (C)\n"
      "  %s undensify <dense_phda9> <m3_side> [out]\n"
      "  %s payload-sim <post_wrt_stream> # M5 similarity reorder (C)\n"
      "  %s seal-product <payload_sim> <m3_header_side>\n"
      "              [--intro-bytes N] [--coda-bytes N] [--main-bytes N]\n"
      "  %s bpe <src> [--out-prefix P] [--vocab N] [--max-train-bytes N]\n"
      "              [--chunk-bytes N] [--no-verify]\n"
      "              [--dict-only | --encode-only]\n",
      argv0, argv0, argv0, argv0, argv0, argv0, argv0, argv0);
}

bool WriteBytes(const std::string& path, const uint8_t* p, size_t n) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f)
    return false;
  if (n && std::fwrite(p, 1, n, f) != n) {
    std::fclose(f);
    return false;
  }
  std::fclose(f);
  return true;
}

bool ReadAll(const std::string& path, std::vector<uint8_t>* out) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f)
    return false;
  if (std::fseek(f, 0, SEEK_END) != 0) {
    std::fclose(f);
    return false;
  }
  long n = std::ftell(f);
  if (n < 0) {
    std::fclose(f);
    return false;
  }
  if (std::fseek(f, 0, SEEK_SET) != 0) {
    std::fclose(f);
    return false;
  }
  out->resize(static_cast<size_t>(n));
  if (n && std::fread(out->data(), 1, static_cast<size_t>(n), f) !=
               static_cast<size_t>(n)) {
    std::fclose(f);
    return false;
  }
  std::fclose(f);
  return true;
}

bool FilesEqual(const std::string& a, const std::string& b) {
  std::ifstream left(a, std::ios::binary);
  std::ifstream right(b, std::ios::binary);
  if (!left.is_open() || !right.is_open())
    return false;
  std::vector<char> lb(1 << 20), rb(1 << 20);
  while (left && right) {
    left.read(lb.data(), static_cast<std::streamsize>(lb.size()));
    right.read(rb.data(), static_cast<std::streamsize>(rb.size()));
    auto lg = left.gcount();
    auto rg = right.gcount();
    if (lg != rg)
      return false;
    if (lg > 0 &&
        std::memcmp(lb.data(), rb.data(), static_cast<size_t>(lg)) != 0)
      return false;
  }
  return left.eof() && right.eof();
}

/* Measure enwik9 intro/coda byte sizes via stock split4Comp (cwd = workdir). */
int MeasureIntroCoda(const std::string& enwik9, size_t* intro_n,
                     size_t* coda_n) {
  split4Comp(enwik9.c_str());
  *intro_n = FileSize(".intro");
  *coda_n = FileSize(".coda");
  if (!*intro_n || !*coda_n) {
    std::fprintf(stderr, "error: failed to measure intro/coda from %s\n",
        enwik9.c_str());
    return 1;
  }
  std::fprintf(stderr, "[M1] measured intro=%zu coda=%zu (from ref enwik9)\n",
      *intro_n, *coda_n);
  return 0;
}

int DecodeWrtToReady(const std::string& wrt_path, const std::string& ready_path,
                     const std::string& dict_path) {
  FILE* in = std::fopen(wrt_path.c_str(), "rb");
  if (!in) {
    std::perror(wrt_path.c_str());
    return 1;
  }
  FILE* dict = std::fopen(dict_path.c_str(), "rb");
  if (!dict) {
    std::perror(dict_path.c_str());
    std::fclose(in);
    return 1;
  }
  FILE* out = std::fopen(ready_path.c_str(), "wb");
  if (!out) {
    std::perror(ready_path.c_str());
    std::fclose(in);
    std::fclose(dict);
    return 1;
  }
  preprocessor::Decode(in, out, dict);
  std::fclose(in);
  std::fclose(dict);
  std::fclose(out);
  return 0;
}

/*
 * Reverse scored payload_sim → enwik9:
 *   unseal M3 → M5 restore → WRT decode → peel intro/coda by byte size →
 *   M3 undensify → phda9_resto → article sort → merge.
 *
 * Densified ready streams cannot use stock split4Decomp line counts; intro/coda
 * come from BLSMETA1 (preferred), CLI, or built-in enwik9 defaults.
 */
int DecodeProduct(const std::string& product_path, const std::string& out_enwik9,
                  const std::string& dict_path, const std::string& workdir,
                  size_t intro_n, size_t coda_n, int sizes_from_cli,
                  const std::string& expect_path) {
  if (chdir(workdir.c_str()) != 0) {
    std::perror(workdir.c_str());
    return 1;
  }

  std::fprintf(stderr,
      "\n======== blsmc prepare decode: payload_sim → enwik9 ========\n");
  Checkpoint("input product", FileSize(product_path), 0);

  std::fprintf(stderr, "[seal] split M3 + peel meta\n");
  BlsmcProductMeta meta{};
  int seal_st = product_seal_split_file(product_path.c_str(), "stream.m5",
      "m3.side", &meta);
  if (seal_st < 0)
    return 1;
  if (seal_st == 0) {
    std::fprintf(stderr,
        "error: product has no M3SIDFTR trailer (need scored payload_sim)\n");
    return 1;
  }
  size_t side_n = FileSize("m3.side");
  if (!side_n) {
    std::fprintf(stderr, "error: empty M3 side after split\n");
    return 1;
  }
  if (!sizes_from_cli && meta.intro_bytes && meta.coda_bytes) {
    intro_n = static_cast<size_t>(meta.intro_bytes);
    coda_n = static_cast<size_t>(meta.coda_bytes);
    std::fprintf(stderr,
        "[seal] BLSMETA1 intro=%zu coda=%zu main=%llu\n", intro_n, coda_n,
        (unsigned long long)meta.main_bytes);
  } else if (!sizes_from_cli) {
    std::fprintf(stderr,
        "[seal] no BLSMETA1; using provided/default intro=%zu coda=%zu\n",
        intro_n, coda_n);
  }

  std::fprintf(stderr, "[M5] payload_sim_restore\n");
  std::vector<uint8_t> m5;
  if (!ReadAll("stream.m5", &m5)) {
    std::fprintf(stderr, "error: read stream.m5\n");
    return 1;
  }
  uint8_t* wrt = nullptr;
  size_t wrt_n = 0;
  if (m5_payload_sim_restore(m5.data(), m5.size(), &wrt, &wrt_n) != 0) {
    std::fprintf(stderr, "error: M5 restore failed\n");
    return 1;
  }
  m5.clear();
  m5.shrink_to_fit();
  if (!WriteBytes("stream.wrt", wrt, wrt_n)) {
    std::free(wrt);
    return 1;
  }
  std::free(wrt);
  Checkpoint("M5 → post-WRT", FileSize("stream.wrt"), 0);

  std::fprintf(stderr, "[M4] WRT dictionary decode\n");
  if (DecodeWrtToReady("stream.wrt", ".ready_dense", dict_path) != 0)
    return 1;
  size_t ready_n = FileSize(".ready_dense");
  Checkpoint("M4 ready (dense)", ready_n, 0);
  if (ready_n < intro_n + coda_n) {
    std::fprintf(stderr,
        "error: ready %zu B smaller than intro+coda %zu+%zu\n", ready_n,
        intro_n, coda_n);
    return 1;
  }

  std::fprintf(stderr, "[split] peel dense_phda9 || intro || coda by bytes\n");
  std::vector<uint8_t> ready;
  if (!ReadAll(".ready_dense", &ready))
    return 1;
  size_t phda9_n = ready.size() - intro_n - coda_n;
  if (!WriteBytes(".main_decomp_dense", ready.data(), phda9_n) ||
      !WriteBytes(".intro_decomp", ready.data() + phda9_n, intro_n) ||
      !WriteBytes(".coda_decomp", ready.data() + phda9_n + intro_n, coda_n)) {
    std::fprintf(stderr, "error: writing peeled parts\n");
    return 1;
  }
  ready.clear();
  ready.shrink_to_fit();
  Checkpoint("dense PHDA9", phda9_n, 0);
  Checkpoint("intro", intro_n, 0);
  Checkpoint("coda", coda_n, 0);

  std::fprintf(stderr, "[M3] undensify header+lang\n");
  if (m3_undensify_file(".main_decomp_dense", "m3.side", ".main_decomp") != 0)
    return 1;
  Checkpoint("raw PHDA9", FileSize(".main_decomp"), 0);

  std::fprintf(stderr, "[M3] phda9_resto → wiki main\n");
  if (phda9_resto() != 0) {
    std::fprintf(stderr, "error: phda9_resto failed\n");
    return 1;
  }
  Checkpoint("main restored", FileSize(".main_decomp_restored"), 0);

  std::fprintf(stderr, "[M2] sort articles by id\n");
  ::sort();
  Checkpoint("main sorted", FileSize(".main_decomp_restored_sorted"), 0);

  std::fprintf(stderr, "[M1] merge intro + main + coda\n");
  if (!cat(".intro_decomp", ".main_decomp_restored_sorted", "un1_d") ||
      !cat("un1_d", ".coda_decomp", out_enwik9.c_str())) {
    std::fprintf(stderr, "error: final merge failed\n");
    return 1;
  }
  size_t out_n = FileSize(out_enwik9);
  Checkpoint("reconstructed enwik9", out_n, 1000000000ull);

  if (!expect_path.empty()) {
    std::fprintf(stderr, "[check] byte-compare vs %s\n", expect_path.c_str());
    if (!FilesEqual(expect_path, out_enwik9)) {
      std::fprintf(stderr, "FAIL: reconstructed != expect\n");
      return 1;
    }
    std::fprintf(stderr, "OK: exact byte match (%zu B)\n", out_n);
  }
  std::fprintf(stderr, "\nDECODE OK → %s (%zu B)\n", out_enwik9.c_str(), out_n);
  return 0;
}

struct RunSizes {
  size_t enwik9 = 0;
  size_t intro = 0, main = 0, coda = 0;
  size_t main_reordered = 0;
  size_t main_phda9_raw = 0;
  size_t main_phda9 = 0;
  size_t m3_densify_side = 0;
  size_t ready = 0;
  size_t post_wrt = 0;
  size_t post_m5_stream = 0; /* payload_sim with M5 side embedded */
  size_t post_m5 = 0;        /* scored product (+ M3 side trailer) */
  size_t side = 0;           /* M5 side bytes (embedded in stream) */
  int m5_ok = 0;
  int m3_densified = 0;
};

int EncodeWiki(const std::string& enwik9, const std::string& workdir,
    const std::string& order_src, RunSizes* sz) {
  if (chdir(workdir.c_str()) != 0) {
    std::perror(workdir.c_str());
    return 1;
  }
  if (!CopyFile(order_src, ".new_article_order")) {
    std::fprintf(stderr, "failed to copy order table from %s\n",
        order_src.c_str());
    return 1;
  }

  std::fprintf(stderr,
      "\n======== blsmc prepare: M1–M5 (densify + payload_sim) ========\n");
  sz->enwik9 = FileSize(enwik9);
  Checkpoint("input enwik9", sz->enwik9, 1000000000ull);

  std::fprintf(stderr, "[M1] split4Comp → .intro / .main / .coda\n");
  split4Comp(enwik9.c_str());
  sz->intro = FileSize(".intro");
  sz->main = FileSize(".main");
  sz->coda = FileSize(".coda");
  Checkpoint("M1 .intro", sz->intro, 0);
  Checkpoint("M1 .main", sz->main, 0);
  Checkpoint("M1 .coda", sz->coda, 0);
  Checkpoint("M1 sum", sz->intro + sz->main + sz->coda, sz->enwik9);

  std::fprintf(stderr, "[M2] reorder (.new_article_order)\n");
  reorder();
  sz->main_reordered = FileSize(".main_reordered");
  if (!sz->main_reordered)
    sz->main_reordered = FileSize(".main");
  Checkpoint("M2 .main_reordered", sz->main_reordered, sz->main);

  std::fprintf(stderr, "[M3] phda9_prepr (cmix-lex)\n");
  phda9_prepr();
  sz->main_phda9_raw = FileSize(".main_phda9prepr");
  Checkpoint("M3 .main_phda9prepr (raw)", sz->main_phda9_raw, 0);

  /* M3: densify header + lang pack (C, M3H2/M3L1). */
  {
    std::fprintf(stderr, "[M3] densify header+lang (C)\n");
    int drc = m3_densify_file(".main_phda9prepr");
    if (drc == 0 && FileSize(".main_phda9prepr.dense") > 0) {
      std::rename(".main_phda9prepr", ".main_phda9prepr.raw");
      std::rename(".main_phda9prepr.dense", ".main_phda9prepr");
      sz->m3_densify_side = FileSize(".main_phda9prepr.dense.side");
      if (!sz->m3_densify_side)
        sz->m3_densify_side = FileSize(".main_phda9prepr.raw.dense.side");
      sz->m3_densified = 1;
      std::fprintf(stderr, "[M3] densify OK side=%zu\n", sz->m3_densify_side);
    } else {
      std::fprintf(stderr, "[M3] densify failed rc=%d — using raw PHDA9\n", drc);
      sz->m3_densified = 0;
    }
  }

  sz->main_phda9 = FileSize(".main_phda9prepr");
  cat(".main_phda9prepr", ".intro", "un1");
  cat("un1", ".coda", ".ready4cmix");
  sz->ready = FileSize(".ready4cmix");
  Checkpoint("M3 .main_phda9prepr", sz->main_phda9, 0);
  if (sz->m3_densified)
    std::fprintf(stderr, "  %-22s %12zu B  (raw was %zu, Δ %+lld)\n",
        "M3 densify product",
        sz->main_phda9 + sz->m3_densify_side, sz->main_phda9_raw,
        (long long)(sz->main_phda9 + sz->m3_densify_side) -
            (long long)sz->main_phda9_raw);
  Checkpoint("M3 .ready4cmix", sz->ready, kReadyCert);

  return 0;
}

int EncodeWrtAndPayload(const std::string& ready_path,
    const std::string& out_587, const std::string& dict_path,
    const std::string& workdir, RunSizes* sz) {
  if (chdir(workdir.c_str()) != 0) {
    std::perror(workdir.c_str());
    return 1;
  }
  FILE* dict = std::fopen(dict_path.c_str(), "rb");
  if (!dict) {
    std::perror(dict_path.c_str());
    return 1;
  }
  FILE* in = std::fopen(ready_path.c_str(), "rb");
  if (!in) {
    std::perror(ready_path.c_str());
    std::fclose(dict);
    return 1;
  }
  std::fseek(in, 0, SEEK_END);
  unsigned long long n = static_cast<unsigned long long>(std::ftell(in));
  std::fseek(in, 0, SEEK_SET);

  const std::string wrt_path = "stream.wrt586";
  FILE* out = std::fopen(wrt_path.c_str(), "wb");
  if (!out) {
    std::perror(wrt_path.c_str());
    std::fclose(in);
    std::fclose(dict);
    return 1;
  }

  std::fprintf(stderr, "[M4] WRT Encode (%llu bytes in, english.dic)\n", n);
  preprocessor::Encode(in, out, n, wrt_path, dict);
  std::fclose(in);
  std::fclose(out);
  std::fclose(dict);

  sz->post_wrt = FileSize(wrt_path);
  Checkpoint("M4 post-WRT", sz->post_wrt, kPostWrt);

  /* M5 payload_sim: similarity-cluster reorder (struct key + n-gram SimHash). */
  std::fprintf(stderr, "[M5] payload_sim (C, struct+simhash64)\n");
  if (m5_payload_sim_file(wrt_path.c_str()) != 0) {
    std::fprintf(stderr, "[M5] payload_sim FAILED\n");
    return 1;
  }
  if (!CopyFile(wrt_path, out_587)) {
    std::fprintf(stderr, "failed to copy payload_sim product to %s\n",
        out_587.c_str());
    return 1;
  }
  /* Sidecar copy of embedded M5 side (also at EOF of stream). */
  {
    std::string side_src = wrt_path + ".payload_sim_side";
    if (FileSize(side_src))
      CopyFile(side_src, out_587 + ".payload_sim_side");
    sz->side = FileSize(side_src);
  }
  sz->post_m5_stream = FileSize(out_587);
  sz->m5_ok = 1;
  Checkpoint("M5 stream (+M5 side emb)", sz->post_m5_stream, kPostPayloadLex);

  /* Seal M3 densify side + peel meta (intro/coda) — not model-compacted. */
  {
    std::string m3_side = out_587 + ".m3_header_side";
    if (!FileSize(m3_side)) {
      std::string alt = workdir + "/.main_phda9prepr.dense.side";
      if (FileSize(alt))
        CopyFile(alt, m3_side);
    }
    BlsmcProductMeta meta{};
    meta.version = BLSMC_META_VERSION;
    meta.intro_bytes = sz->intro;
    meta.coda_bytes = sz->coda;
    meta.main_bytes = sz->main;
    const char* side_arg =
        (sz->m3_densify_side && FileSize(m3_side)) ? m3_side.c_str() : nullptr;
    if (product_seal_append(out_587.c_str(), side_arg, &meta) != 0) {
      std::fprintf(stderr, "[seal] failed to append M3/meta to %s\n",
          out_587.c_str());
      return 1;
    }
  }
  sz->post_m5 = FileSize(out_587);
  Checkpoint("M5 scored product (+M3+meta)", sz->post_m5, kPostPayloadLex);

  std::fprintf(stderr, "\n======== vs cmix-lex bars (scored = stream + sides) ========\n");
  size_t m3_product = sz->main_phda9 + sz->m3_densify_side;
  size_t ready_product = sz->ready + sz->m3_densify_side;
  size_t wrt_scored = sz->post_wrt + sz->m3_densify_side;
  Checkpoint("ready+side (pre-WRT)", ready_product, kReadyCert);
  Checkpoint("post-WRT stream", sz->post_wrt, kPostWrt);
  Checkpoint("post-WRT scored (+M3)", wrt_scored, kPostWrt);
  Checkpoint("post-M5 stream", sz->post_m5_stream, kPostPayloadLex);
  Checkpoint("post-M5 scored product", sz->post_m5, kPostPayloadLex);

  long long vs_ready = (long long)ready_product - (long long)kReadyCert;
  long long vs586 = (long long)wrt_scored - (long long)kPostWrt;
  long long vs587 = (long long)sz->post_m5 - (long long)kPostPayloadLex;
  std::fprintf(stderr,
      "\nVERDICT ready+side %zu vs 934: %+lld | post-WRT+M3 vs 586: %+lld | "
      "scored product vs 587: %+lld\n"
      "  (M5 side %zu embedded in stream; M3 side %zu + trailer in product)\n",
      ready_product, vs_ready, vs586, vs587, sz->side, sz->m3_densify_side);
  std::fprintf(stderr, "wrote scored product %s (%zu B)\n", out_587.c_str(),
      sz->post_m5);

  std::string meta = workdir + "/blsmc_prepare.json";
  FILE* mf = std::fopen(meta.c_str(), "w");
  if (mf) {
    std::fprintf(mf,
        "{\n"
        "  \"enwik9_bytes\": %zu,\n"
        "  \"m1_intro\": %zu,\n"
        "  \"m1_main\": %zu,\n"
        "  \"m1_coda\": %zu,\n"
        "  \"m2_main_reordered\": %zu,\n"
        "  \"m3_main_phda9_raw\": %zu,\n"
        "  \"m3_main_phda9\": %zu,\n"
        "  \"m3_densify_side\": %zu,\n"
        "  \"m3_densified\": %s,\n"
        "  \"m3_product\": %zu,\n"
        "  \"m3_ready4cmix\": %zu,\n"
        "  \"m4_post_wrt\": %zu,\n"
        "  \"m4_scored\": %zu,\n"
        "  \"m5_payload_sim_stream\": %zu,\n"
        "  \"m5_side\": %zu,\n"
        "  \"m5_scored_product\": %zu,\n"
        "  \"m5_ok\": %s,\n"
        "  \"bar_ready\": %zu,\n"
        "  \"bar_586\": %zu,\n"
        "  \"bar_587\": %zu,\n"
        "  \"vs_ready_product\": %lld,\n"
        "  \"vs_586\": %lld,\n"
        "  \"vs_587\": %lld,\n"
        "  \"under_ready_bar\": %s,\n"
        "  \"under_586\": %s,\n"
        "  \"under_587\": %s\n"
        "}\n",
        sz->enwik9, sz->intro, sz->main, sz->coda, sz->main_reordered,
        sz->main_phda9_raw, sz->main_phda9, sz->m3_densify_side,
        sz->m3_densified ? "true" : "false", m3_product, sz->ready,
        sz->post_wrt, wrt_scored, sz->post_m5_stream, sz->side, sz->post_m5,
        sz->m5_ok ? "true" : "false", kReadyCert, kPostWrt, kPostPayloadLex,
        vs_ready, vs586, vs587,
        ready_product < kReadyCert ? "true" : "false",
        wrt_scored < kPostWrt ? "true" : "false",
        sz->post_m5 < kPostPayloadLex ? "true" : "false");
    std::fclose(mf);
    std::fprintf(stderr, "wrote %s\n", meta.c_str());
  }

  return 0;
}

std::string AbsPathExisting(const std::string& path) {
  char* r = realpath(path.c_str(), nullptr);
  if (!r)
    return path;
  std::string out(r);
  free(r);
  return out;
}

std::string AbsOutPath(const std::string& out_path) {
  std::string parent = out_path;
  auto slash = parent.find_last_of('/');
  if (slash == std::string::npos)
    return out_path;
  parent.resize(slash);
  char* rp = realpath(parent.c_str(), nullptr);
  if (!rp)
    return out_path;
  std::string abs = std::string(rp) + out_path.substr(slash);
  free(rp);
  return abs;
}

}  // namespace

std::string DetectRepoRoot() {
  if (const char* r = std::getenv("BLSMC_ROOT"))
    return r;
  if (access("../../dict/english.dic", F_OK) == 0)
    return "../..";
  if (access("dict/english.dic", F_OK) == 0)
    return ".";
  return ".";
}

int main(int argc, char** argv) {
  if (argc < 2) {
    Usage(argv[0]);
    return 2;
  }
  const std::string cmd = argv[1];
  const std::string repo = DetectRepoRoot();
  std::string dict = repo + "/dict/english.dic";
  std::string order = repo + "/dict/new_article_order";
  std::string workdir = repo + "/data/blsmc_prepare_work";

  if (cmd == "encode") {
    if (argc < 4) {
      Usage(argv[0]);
      return 2;
    }
    std::string enwik9 = argv[2];
    std::string out_587 = argv[3];
    for (int i = 4; i < argc; i++) {
      if (std::strcmp(argv[i], "--workdir") == 0 && i + 1 < argc)
        workdir = argv[++i];
      else if (std::strcmp(argv[i], "--dict") == 0 && i + 1 < argc)
        dict = argv[++i];
      else if (std::strcmp(argv[i], "--order") == 0 && i + 1 < argc)
        order = argv[++i];
    }
    enwik9 = AbsPathExisting(enwik9);
    dict = AbsPathExisting(dict);
    order = AbsPathExisting(order);
    std::string abs_out = AbsOutPath(out_587);

    std::string cmd_mkdir = "mkdir -p '" + workdir + "'";
    if (std::system(cmd_mkdir.c_str()) != 0)
      return 1;
    workdir = AbsPathExisting(workdir);

    RunSizes sz;
    if (EncodeWiki(enwik9, workdir, order, &sz) != 0)
      return 1;
    if (sz.m3_densified && sz.m3_densify_side) {
      CopyFile(workdir + "/.main_phda9prepr.dense.side",
          abs_out + ".m3_header_side");
    }
    return EncodeWrtAndPayload(workdir + "/.ready4cmix", abs_out, dict,
        workdir, &sz);
  }

  if (cmd == "encode-from-ready") {
    if (argc < 4) {
      Usage(argv[0]);
      return 2;
    }
    std::string ready = argv[2];
    std::string out_587 = argv[3];
    for (int i = 4; i < argc; i++) {
      if (std::strcmp(argv[i], "--dict") == 0 && i + 1 < argc)
        dict = argv[++i];
      else if (std::strcmp(argv[i], "--workdir") == 0 && i + 1 < argc)
        workdir = argv[++i];
    }
    ready = AbsPathExisting(ready);
    dict = AbsPathExisting(dict);
    std::string cmd_mkdir = "mkdir -p '" + workdir + "'";
    std::system(cmd_mkdir.c_str());
    workdir = AbsPathExisting(workdir);
    std::string ready_in_work = workdir + "/.ready4cmix";
    if (ready != ready_in_work)
      CopyFile(ready, ready_in_work);
    RunSizes sz;
    sz.ready = FileSize(ready_in_work);
    /* Peel sizes unknown without M1; seal official enwik9 defaults. */
    sz.intro = kEnwik9IntroBytes;
    sz.coda = kEnwik9CodaBytes;
    return EncodeWrtAndPayload(ready_in_work, AbsOutPath(out_587), dict,
        workdir, &sz);
  }

  if (cmd == "decode") {
    if (argc < 4) {
      Usage(argv[0]);
      return 2;
    }
    std::string product = argv[2];
    std::string out_enwik9 = argv[3];
    std::string enwik9_ref;
    std::string expect_path;
    size_t intro_bytes = 0, coda_bytes = 0;
    int sizes_from_cli = 0;
    for (int i = 4; i < argc; i++) {
      if (std::strcmp(argv[i], "--dict") == 0 && i + 1 < argc)
        dict = argv[++i];
      else if (std::strcmp(argv[i], "--workdir") == 0 && i + 1 < argc)
        workdir = argv[++i];
      else if (std::strcmp(argv[i], "--enwik9") == 0 && i + 1 < argc)
        enwik9_ref = argv[++i];
      else if (std::strcmp(argv[i], "--expect") == 0 && i + 1 < argc)
        expect_path = argv[++i];
      else if (std::strcmp(argv[i], "--intro-bytes") == 0 && i + 1 < argc) {
        intro_bytes = std::strtoull(argv[++i], nullptr, 10);
        sizes_from_cli = 1;
      } else if (std::strcmp(argv[i], "--coda-bytes") == 0 && i + 1 < argc) {
        coda_bytes = std::strtoull(argv[++i], nullptr, 10);
        sizes_from_cli = 1;
      } else {
        Usage(argv[0]);
        return 2;
      }
    }
    product = AbsPathExisting(product);
    dict = AbsPathExisting(dict);
    std::string abs_out = AbsOutPath(out_enwik9);
    if (!expect_path.empty())
      expect_path = AbsPathExisting(expect_path);

    std::string cmd_mkdir = "mkdir -p '" + workdir + "'";
    if (std::system(cmd_mkdir.c_str()) != 0)
      return 1;
    workdir = AbsPathExisting(workdir);

    /* Prefer BLSMETA1 inside product; CLI / --enwik9 / built-in are fallbacks. */
    if (!intro_bytes || !coda_bytes) {
      if (!enwik9_ref.empty()) {
        enwik9_ref = AbsPathExisting(enwik9_ref);
        if (chdir(workdir.c_str()) != 0) {
          std::perror(workdir.c_str());
          return 1;
        }
        if (MeasureIntroCoda(enwik9_ref, &intro_bytes, &coda_bytes) != 0)
          return 1;
        sizes_from_cli = 1; /* force measured sizes over meta */
      } else {
        intro_bytes = kEnwik9IntroBytes;
        coda_bytes = kEnwik9CodaBytes;
        /* sizes_from_cli=0 → DecodeProduct may override from BLSMETA1 */
      }
    }

    return DecodeProduct(product, abs_out, dict, workdir, intro_bytes,
        coda_bytes, sizes_from_cli, expect_path);
  }

  if (cmd == "densify") {
    if (argc < 3) {
      Usage(argv[0]);
      return 2;
    }
    return m3_densify_file(argv[2]) == 0 ? 0 : 1;
  }

  if (cmd == "undensify") {
    if (argc < 4) {
      Usage(argv[0]);
      return 2;
    }
    const char* out = argc >= 5 ? argv[4] : nullptr;
    return m3_undensify_file(argv[2], argv[3], out) == 0 ? 0 : 1;
  }

  if (cmd == "payload-sim") {
    if (argc < 3) {
      Usage(argv[0]);
      return 2;
    }
    return m5_payload_sim_file(argv[2]) == 0 ? 0 : 1;
  }

  if (cmd == "seal-product") {
    if (argc < 4) {
      Usage(argv[0]);
      return 2;
    }
    std::string product = argv[2];
    std::string side = argv[3];
    size_t intro_b = kEnwik9IntroBytes, coda_b = kEnwik9CodaBytes,
           main_b = 0;
    for (int i = 4; i < argc; i++) {
      if (std::strcmp(argv[i], "--intro-bytes") == 0 && i + 1 < argc)
        intro_b = std::strtoull(argv[++i], nullptr, 10);
      else if (std::strcmp(argv[i], "--coda-bytes") == 0 && i + 1 < argc)
        coda_b = std::strtoull(argv[++i], nullptr, 10);
      else if (std::strcmp(argv[i], "--main-bytes") == 0 && i + 1 < argc)
        main_b = std::strtoull(argv[++i], nullptr, 10);
      else {
        Usage(argv[0]);
        return 2;
      }
    }
    BlsmcProductMeta meta{};
    meta.version = BLSMC_META_VERSION;
    meta.intro_bytes = intro_b;
    meta.coda_bytes = coda_b;
    meta.main_bytes = main_b;
    if (product_seal_append(product.c_str(), side.c_str(), &meta) != 0)
      return 1;
    std::fprintf(stderr, "scored product %s → %zu B\n", product.c_str(),
        FileSize(product));
    return 0;
  }

  if (cmd == "bpe") {
    if (argc < 3) {
      Usage(argv[0]);
      return 2;
    }
    const char *src = argv[2];
    std::string out_prefix = src;
    uint32_t vocab = BPE_DEFAULT_VOCAB;
    size_t max_train = BPE_DEFAULT_TRAIN_BYTES;
    size_t chunk = BPE_DEFAULT_CHUNK_BYTES;
    int verify = 1;
    int mode = BPE_MODE_FULL;
    for (int i = 3; i < argc; i++) {
      if (std::strcmp(argv[i], "--out-prefix") == 0 && i + 1 < argc)
        out_prefix = argv[++i];
      else if (std::strcmp(argv[i], "--vocab") == 0 && i + 1 < argc)
        vocab = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
      else if (std::strcmp(argv[i], "--max-train-bytes") == 0 && i + 1 < argc)
        max_train = (size_t)std::strtoull(argv[++i], nullptr, 10);
      else if (std::strcmp(argv[i], "--chunk-bytes") == 0 && i + 1 < argc)
        chunk = (size_t)std::strtoull(argv[++i], nullptr, 10);
      else if (std::strcmp(argv[i], "--no-verify") == 0)
        verify = 0;
      else if (std::strcmp(argv[i], "--dict-only") == 0)
        mode = BPE_MODE_DICT_ONLY;
      else if (std::strcmp(argv[i], "--encode-only") == 0)
        mode = BPE_MODE_ENCODE_ONLY;
      else {
        Usage(argv[0]);
        return 2;
      }
    }
    return bpe_prepare_file(src, out_prefix.c_str(), vocab, max_train, chunk,
               verify, mode) == 0
               ? 0
               : 1;
  }

  Usage(argv[0]);
  return 2;
}
