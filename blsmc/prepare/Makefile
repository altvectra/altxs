# blsmc_prepare — peel encode/decode (M1–M5).
# Requires: vendor/cmix-lex (./scripts/fetch_vendors.sh)
#
# Uses cmix-lex readalike_prepr (forked PHDA9 + .new_article_order), NOT stock
# cmix enwik9-preproc. Shim under ./shim/ stubs predictor.h for WRT Encode.

CXX ?= clang++
CC ?= clang
REPO := $(abspath ../..)
CMIX_LEX := $(REPO)/vendor/cmix-lex
HOST_ARCH := $(shell uname -m)

CXXFLAGS ?= -std=c++17 -O2 -Wall \
	-I. \
	-Ishim \
	-I$(CMIX_LEX)/src/readalike_prepr \
	-I$(CMIX_LEX)/src
# _DARWIN_C_SOURCE: memmem on macOS when POSIX feature macros are set.
# _GNU_SOURCE: memmem on Linux (also set in m5_payload_sim.c).
CFLAGS ?= -std=c11 -O2 -Wall -I. -D_GNU_SOURCE -D_DARWIN_C_SOURCE

# One ABI for C objects and the C++ link. Copies of this tree (or Rosetta
# vs native clang) otherwise leave arm64 .o files that an x86_64 c++ rejects.
ifeq ($(shell uname -s),Darwin)
  CFLAGS += -arch $(HOST_ARCH)
  CXXFLAGS += -arch $(HOST_ARCH)
endif

.PHONY: all check-arch clean shim encode9 decode9

all: check-arch
	$(MAKE) blsmc_prepare

check-arch:
	@cur="$(HOST_ARCH)"; \
	old=$$(cat .arch 2>/dev/null || echo ""); \
	if [ "$$old" != "$$cur" ]; then \
	  echo "architecture $${old:-none} -> $$cur; rebuilding peel objects"; \
	  rm -f m3_densify.o m5_payload_sim.o product_seal.o bpe_tokenizer.o blsmc_prepare; \
	  printf '%s\n' "$$cur" > .arch; \
	fi

shim:
	@test -d "$(CMIX_LEX)" || { echo "error: missing $(CMIX_LEX); run ./scripts/fetch_vendors.sh" >&2; exit 1; }
	@mkdir -p shim/preprocess
	@test -f shim/predictor.h || cp stubs/predictor.h shim/predictor.h
	@test -L shim/ds || ln -sfn ../../../vendor/cmix-lex/src/ds shim/ds
	@test -L shim/preprocess/preprocessor.cpp || \
	  ln -sfn ../../../../vendor/cmix-lex/src/preprocess/preprocessor.cpp \
	    shim/preprocess/preprocessor.cpp
	@test -L shim/preprocess/dictionary.cpp || \
	  ln -sfn ../../../../vendor/cmix-lex/src/preprocess/dictionary.cpp \
	    shim/preprocess/dictionary.cpp
	@test -L shim/preprocess/dictionary.h || \
	  ln -sfn ../../../../vendor/cmix-lex/src/preprocess/dictionary.h \
	    shim/preprocess/dictionary.h
	@test -f shim/preprocess/preprocessor.h || \
	  cp $(CMIX_LEX)/src/preprocess/preprocessor.h shim/preprocess/preprocessor.h

m3_densify.o: m3_densify.c m3_densify.h
	$(CC) $(CFLAGS) -c -o $@ m3_densify.c

m5_payload_sim.o: m5_payload_sim.c m5_payload_sim.h
	$(CC) $(CFLAGS) -c -o $@ m5_payload_sim.c

product_seal.o: product_seal.c product_seal.h
	$(CC) $(CFLAGS) -c -o $@ product_seal.c

bpe_tokenizer.o: bpe_tokenizer.c bpe_tokenizer.h product_seal.h
	$(CC) $(CFLAGS) -c -o $@ bpe_tokenizer.c

blsmc_prepare: shim m3_densify.o m5_payload_sim.o product_seal.o bpe_tokenizer.o \
	prepare_main.cpp shim/predictor.h shim/preprocess/preprocessor.h
	$(CXX) $(CXXFLAGS) -o $@ prepare_main.cpp m3_densify.o m5_payload_sim.o \
	  product_seal.o bpe_tokenizer.o \
	  shim/preprocess/dictionary.cpp \
	  shim/preprocess/preprocessor.cpp \
	  $(CMIX_LEX)/src/r1_reorder_transform.cpp

clean:
	rm -f blsmc_prepare m3_densify.o m5_payload_sim.o product_seal.o bpe_tokenizer.o .arch

# Full enwik9 → payload_sim. Fresh workdir every run.
encode9: blsmc_prepare
	rm -rf $(REPO)/data/blsmc_prepare_work
	mkdir -p $(REPO)/data/blsmc_prepare_work
	./blsmc_prepare encode \
	  $(REPO)/data/enwik9 \
	  $(REPO)/data/enwik9.blsmc_full.m3v2.payload_sim \
	  --dict $(REPO)/dict/english.dic \
	  --order $(REPO)/dict/new_article_order \
	  --workdir $(REPO)/data/blsmc_prepare_work

# Reverse scored payload_sim → enwik9.
decode9: blsmc_prepare
	mkdir -p $(REPO)/data/blsmc_decode_work
	./blsmc_prepare decode \
	  $(REPO)/data/enwik9.blsmc_full.m3v2.payload_sim \
	  $(REPO)/data/enwik9.from_payload_sim \
	  --dict $(REPO)/dict/english.dic \
	  --workdir $(REPO)/data/blsmc_decode_work \
	  $(if $(wildcard $(REPO)/data/enwik9),--expect $(REPO)/data/enwik9,)
