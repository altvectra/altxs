"""GPU Witten–Neal–Cleary range coder (lockstep with ``arithmetic_coder_lm``).

Hot path stays on HBM: probs, symbols, and coder state never DtoH until a
segment flush. Encode and decode must both use this path.

Triton cannot compile the nested scalar renorm/pending loops (LLVM
pipeline crash on H100). The coder is a one-thread CUDA kernel instead.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import torch

_CODE_BITS = 32
_TOP = 1 << _CODE_BITS
_HALF = _TOP >> 1
_QUARTER = _HALF >> 1

_MOD: Any = None
_MOD_ERR: str | None = None

_CUDA_SRC = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <stdint.h>
#include <math.h>

#define CODE_BITS 32
#define TOP (1LL << CODE_BITS)
#define HALF (TOP >> 1)
#define QUARTER (HALF >> 1)

__device__ __forceinline__ int64_t scale_p(float p) {
    double s = floor((double)p * 1000000.0 + 0.5);
    int64_t v = (int64_t)s;
    return v < 1 ? 1 : v;
}

__device__ __forceinline__ void emit(
    int64_t* bits, int64_t* nb, int64_t max_bits, int64_t bit
) {
    if (*nb < max_bits) {
        bits[*nb] = bit;
        *nb += 1;
    }
}

__device__ void renorm_encode(
    int64_t* low, int64_t* high, int64_t* pending,
    int64_t* bits, int64_t* nb, int64_t max_bits
) {
    while (true) {
        if (*high < HALF) {
            emit(bits, nb, max_bits, 0);
            while (*pending > 0) {
                emit(bits, nb, max_bits, 1);
                *pending -= 1;
            }
        } else if (*low >= HALF) {
            emit(bits, nb, max_bits, 1);
            while (*pending > 0) {
                emit(bits, nb, max_bits, 0);
                *pending -= 1;
            }
            *low -= HALF;
            *high -= HALF;
        } else if (*low >= QUARTER && *high < 3 * QUARTER) {
            *pending += 1;
            *low -= QUARTER;
            *high -= QUARTER;
        } else {
            break;
        }
        *low <<= 1;
        *high = (*high << 1) | 1;
    }
}

extern "C" __global__ void encode_sym_kernel(
    const float* __restrict__ probs,
    const int64_t* __restrict__ sym_ptr,
    int64_t* __restrict__ low_ptr,
    int64_t* __restrict__ high_ptr,
    int64_t* __restrict__ pending_ptr,
    int64_t* __restrict__ bits,
    int64_t* __restrict__ nbits_ptr,
    int64_t V,
    int64_t max_bits
) {
    int64_t low = *low_ptr;
    int64_t high = *high_ptr;
    int64_t pending = *pending_ptr;
    int64_t nb = *nbits_ptr;
    int64_t sym = *sym_ptr;
    int64_t acc = 0, sl = 0, sh = 0;
    for (int64_t i = 0; i < V; i++) {
        int64_t s = scale_p(probs[i]);
        if (i == sym) sl = acc;
        acc += s;
        if (i == sym) sh = acc;
    }
    int64_t total = acc;
    int64_t rng = high - low + 1;
    high = low + rng * sh / total - 1;
    low = low + rng * sl / total;
    renorm_encode(&low, &high, &pending, bits, &nb, max_bits);
    *low_ptr = low;
    *high_ptr = high;
    *pending_ptr = pending;
    *nbits_ptr = nb;
}

extern "C" __global__ void flush_kernel(
    const int64_t* __restrict__ low_ptr,
    int64_t* __restrict__ pending_ptr,
    int64_t* __restrict__ bits,
    int64_t* __restrict__ nbits_ptr,
    int64_t max_bits
) {
    int64_t low = *low_ptr;
    int64_t pending = *pending_ptr + 1;
    int64_t nb = *nbits_ptr;
    if (low < QUARTER) {
        emit(bits, &nb, max_bits, 0);
        while (pending > 0) {
            emit(bits, &nb, max_bits, 1);
            pending -= 1;
        }
    } else {
        emit(bits, &nb, max_bits, 1);
        while (pending > 0) {
            emit(bits, &nb, max_bits, 0);
            pending -= 1;
        }
    }
    *nbits_ptr = nb;
}

extern "C" __global__ void seed_value_kernel(
    const int64_t* __restrict__ in_bits,
    const int64_t* __restrict__ n_in_ptr,
    int64_t* __restrict__ value_ptr
) {
    int64_t n_in = *n_in_ptr;
    int64_t value = 0;
    for (int64_t i = 0; i < CODE_BITS; i++) {
        int64_t bit = (i < n_in) ? in_bits[i] : 0;
        value = (value << 1) | bit;
    }
    *value_ptr = value;
}

extern "C" __global__ void decode_sym_kernel(
    const float* __restrict__ probs,
    int64_t* __restrict__ sym_ptr,
    int64_t* __restrict__ low_ptr,
    int64_t* __restrict__ high_ptr,
    int64_t* __restrict__ value_ptr,
    const int64_t* __restrict__ in_bits,
    int64_t* __restrict__ bpos_ptr,
    const int64_t* __restrict__ n_in_ptr,
    int64_t V
) {
    int64_t low = *low_ptr;
    int64_t high = *high_ptr;
    int64_t value = *value_ptr;
    int64_t bpos = *bpos_ptr;
    int64_t n_in = *n_in_ptr;
    int64_t acc = 0;
    for (int64_t i = 0; i < V; i++) acc += scale_p(probs[i]);
    int64_t total = acc;
    int64_t rng = high - low + 1;
    int64_t offset = ((value - low + 1) * total - 1) / rng;
    acc = 0;
    int64_t sl = 0, sh = 1, sym = 0;
    for (int64_t i = 0; i < V; i++) {
        int64_t s = scale_p(probs[i]);
        if (acc <= offset) {
            sl = acc;
            sym = i;
        }
        acc += s;
        if (i == sym) sh = acc;
    }
    if (sym < 0) sym = 0;
    if (sym >= V) sym = V - 1;
    high = low + rng * sh / total - 1;
    low = low + rng * sl / total;
    *sym_ptr = sym;
    while (true) {
        if (high < HALF) {
        } else if (low >= HALF) {
            low -= HALF;
            high -= HALF;
            value -= HALF;
        } else if (low >= QUARTER && high < 3 * QUARTER) {
            low -= QUARTER;
            high -= QUARTER;
            value -= QUARTER;
        } else {
            break;
        }
        low <<= 1;
        high = (high << 1) | 1;
        int64_t bit = (bpos < n_in) ? in_bits[bpos] : 0;
        bpos += 1;
        value = (value << 1) | bit;
    }
    *low_ptr = low;
    *high_ptr = high;
    *value_ptr = value;
    *bpos_ptr = bpos;
}

extern "C" void launch_encode_sym(
    const float* probs, const int64_t* sym, int64_t* low, int64_t* high,
    int64_t* pending, int64_t* bits, int64_t* nbits, int64_t V, int64_t max_bits
) {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    encode_sym_kernel<<<1, 1, 0, stream>>>(
        probs, sym, low, high, pending, bits, nbits, V, max_bits
    );
}

extern "C" void launch_flush(
    const int64_t* low, int64_t* pending, int64_t* bits, int64_t* nbits,
    int64_t max_bits
) {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    flush_kernel<<<1, 1, 0, stream>>>(low, pending, bits, nbits, max_bits);
}

extern "C" void launch_seed(
    const int64_t* in_bits, const int64_t* n_in, int64_t* value
) {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    seed_value_kernel<<<1, 1, 0, stream>>>(in_bits, n_in, value);
}

extern "C" void launch_decode_sym(
    const float* probs, int64_t* sym, int64_t* low, int64_t* high,
    int64_t* value, const int64_t* in_bits, int64_t* bpos, const int64_t* n_in,
    int64_t V
) {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    decode_sym_kernel<<<1, 1, 0, stream>>>(
        probs, sym, low, high, value, in_bits, bpos, n_in, V
    );
}
"""

_CPP_SRC = r"""
#include <torch/extension.h>
#include <cstdint>

extern "C" void launch_encode_sym(
    const float*, const int64_t*, int64_t*, int64_t*, int64_t*,
    int64_t*, int64_t*, int64_t, int64_t);
extern "C" void launch_flush(
    const int64_t*, int64_t*, int64_t*, int64_t*, int64_t);
extern "C" void launch_seed(const int64_t*, const int64_t*, int64_t*);
extern "C" void launch_decode_sym(
    const float*, int64_t*, int64_t*, int64_t*, int64_t*,
    const int64_t*, int64_t*, const int64_t*, int64_t);

void encode_sym(
    torch::Tensor probs,
    torch::Tensor sym,
    torch::Tensor low,
    torch::Tensor high,
    torch::Tensor pending,
    torch::Tensor bits,
    torch::Tensor nbits
) {
    launch_encode_sym(
        probs.data_ptr<float>(),
        sym.data_ptr<int64_t>(),
        low.data_ptr<int64_t>(),
        high.data_ptr<int64_t>(),
        pending.data_ptr<int64_t>(),
        bits.data_ptr<int64_t>(),
        nbits.data_ptr<int64_t>(),
        probs.numel(),
        bits.numel()
    );
}

void flush_bits(
    torch::Tensor low,
    torch::Tensor pending,
    torch::Tensor bits,
    torch::Tensor nbits
) {
    launch_flush(
        low.data_ptr<int64_t>(),
        pending.data_ptr<int64_t>(),
        bits.data_ptr<int64_t>(),
        nbits.data_ptr<int64_t>(),
        bits.numel()
    );
}

void seed_value(
    torch::Tensor in_bits,
    torch::Tensor n_in,
    torch::Tensor value
) {
    launch_seed(
        in_bits.data_ptr<int64_t>(),
        n_in.data_ptr<int64_t>(),
        value.data_ptr<int64_t>()
    );
}

void decode_sym(
    torch::Tensor probs,
    torch::Tensor sym,
    torch::Tensor low,
    torch::Tensor high,
    torch::Tensor value,
    torch::Tensor in_bits,
    torch::Tensor bpos,
    torch::Tensor n_in
) {
    launch_decode_sym(
        probs.data_ptr<float>(),
        sym.data_ptr<int64_t>(),
        low.data_ptr<int64_t>(),
        high.data_ptr<int64_t>(),
        value.data_ptr<int64_t>(),
        in_bits.data_ptr<int64_t>(),
        bpos.data_ptr<int64_t>(),
        n_in.data_ptr<int64_t>(),
        probs.numel()
    );
}
"""


def _mod():
    global _MOD, _MOD_ERR
    if _MOD is not None:
        return _MOD
    if _MOD_ERR is not None:
        raise RuntimeError(f"gpu AC CUDA extension unavailable: {_MOD_ERR}")
    try:
        from torch.utils.cpp_extension import load_inline

        _MOD = load_inline(
            name="xsa_gpu_ac",
            cpp_sources=_CPP_SRC,
            cuda_sources=_CUDA_SRC,
            functions=["encode_sym", "flush_bits", "seed_value", "decode_sym"],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    except Exception as exc:
        _MOD_ERR = repr(exc)
        raise RuntimeError(f"gpu AC CUDA extension unavailable: {_MOD_ERR}") from exc
    return _MOD


def _pack_bits(bits: torch.Tensor, nbits: int) -> bytes:
    if nbits <= 0:
        return b""
    host = bits[:nbits].detach().cpu().numpy()
    out = bytearray()
    acc = 0
    n = 0
    for b in host:
        acc = (acc << 1) | (int(b) & 1)
        n += 1
        if n == 8:
            out.append(acc)
            acc = 0
            n = 0
    if n:
        out.append(acc << (8 - n))
    return bytes(out)


def _unpack_payload(payload: bytes, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    bits: list[int] = []
    for byte in payload:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    t = torch.tensor(bits, dtype=torch.int64, device=device)
    n = torch.tensor([t.numel()], dtype=torch.int64, device=device)
    return t, n


class GpuRangeCoder:
    """Device-resident range coder. One DtoH at ``finish`` / decoded tokens."""

    def __init__(self, device: torch.device, *, max_bits: int, vocab: int):
        self.device = device
        self.vocab = int(vocab)
        self.max_bits = int(max_bits)
        self.low = torch.zeros(1, dtype=torch.int64, device=device)
        self.high = torch.tensor([_TOP - 1], dtype=torch.int64, device=device)
        self.pending = torch.zeros(1, dtype=torch.int64, device=device)
        self.value = torch.zeros(1, dtype=torch.int64, device=device)
        self.nbits = torch.zeros(1, dtype=torch.int64, device=device)
        self.bpos = torch.zeros(1, dtype=torch.int64, device=device)
        self.bits = torch.zeros(self.max_bits, dtype=torch.int64, device=device)
        self.sym = torch.zeros(1, dtype=torch.int64, device=device)
        # Keep the decode input address stable so a multi-symbol CUDA graph
        # can be replayed across segment payloads.
        self._in_bits = torch.zeros(self.max_bits, dtype=torch.int64, device=device)
        self._n_in = torch.zeros(1, dtype=torch.int64, device=device)
        self._ext = _mod()

    @staticmethod
    def _probs_arg(probs: torch.Tensor) -> torch.Tensor:
        """Pass a stable float32 vector into the 1-thread coder kernel.

        ``.float().contiguous()`` can allocate. A CUDA graph would then
        replay against that capture-time temporary instead of the live
        softmax buffer.
        """
        p = probs.view(-1)
        if p.dtype == torch.float32 and p.is_contiguous():
            return p
        return p.float().contiguous()

    def encode_symbol(self, probs: torch.Tensor, symbol: torch.Tensor) -> None:
        self.sym.copy_(symbol.view(-1)[:1].to(dtype=torch.int64))
        self._ext.encode_sym(
            self._probs_arg(probs),
            self.sym,
            self.low,
            self.high,
            self.pending,
            self.bits,
            self.nbits,
        )

    def finish(self) -> bytes:
        self._ext.flush_bits(self.low, self.pending, self.bits, self.nbits)
        n = int(self.nbits.item())
        return _pack_bits(self.bits, n)

    def begin_decode(self, payload: bytes) -> None:
        bits, n_in = _unpack_payload(payload, self.device)
        n_bits = int(bits.numel())
        if n_bits > self._in_bits.numel():
            raise ValueError(
                f"decode payload has {n_bits} bits, capacity is "
                f"{self._in_bits.numel()}"
            )
        if n_bits < self._in_bits.numel():
            self._in_bits[n_bits:].zero_()
        self._in_bits[:n_bits].copy_(bits)
        self._n_in.copy_(n_in)
        self.low.zero_()
        self.high.fill_(_TOP - 1)
        self.pending.zero_()
        self.nbits.zero_()
        self.bpos.zero_()
        self._ext.seed_value(self._in_bits, self._n_in, self.value)
        self.bpos.fill_(_CODE_BITS)

    def decode_symbol(self, probs: torch.Tensor) -> torch.Tensor:
        self._ext.decode_sym(
            self._probs_arg(probs),
            self.sym,
            self.low,
            self.high,
            self.value,
            self._in_bits,
            self.bpos,
            self._n_in,
        )
        return self.sym


def can_gpu_ac(device: torch.device) -> bool:
    if device.type != "cuda" or os.environ.get("XSA_AC_GPU_AC", "1").strip() in {
        "0",
        "false",
        "off",
        "no",
    }:
        return False
    try:
        _mod()
        return True
    except Exception as exc:
        print(
            f"[AC incr] HBM range coder unavailable ({exc}); using host AC",
            file=sys.stderr,
            flush=True,
        )
        return False
