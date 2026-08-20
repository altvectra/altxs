"""Arithmetic coding of a byte stream under a neural LM (deterministic).

Ideal payload bits ≈ sum -log2 p(x_i | x_<i). This coder materializes an
actual bitstream for round-trip checks on toy corpora.
"""

from __future__ import annotations

import hashlib
import math
import sys
from typing import Callable, Iterable, Sequence

# Witten–Neal–Cleary style range coder (32-bit).
_CODE_BITS = 32
_TOP = 1 << _CODE_BITS
_HALF = _TOP >> 1
_QUARTER = _HALF >> 1

# Prefer buffer types so encode/decode can avoid O(n²) ``bytes`` copies on long files.
Prefix = bytes | bytearray | memoryview | Sequence[int]
# list/tuple/ndarray all OK — hot path prefers contiguous float32 numpy.
ProbsFn = Callable[[int, Prefix], Sequence[float]]
# Symbol stream: bytes (alphabet 256) or int sequence / ndarray (general vocab).
Symbols = bytes | bytearray | memoryview | Sequence[int]


def _progress_range(n: int, desc: str | None) -> Iterable[int]:
    """Symbol index iterator; force tqdm even under ``tee`` (non-TTY pipes)."""
    indices = range(n)
    if desc is None:
        return indices
    from tqdm import tqdm

    # ``disable=False``: default tqdm disables when stderr is not a TTY (e.g. ``| tee``).
    return tqdm(
        indices,
        total=n,
        desc=desc,
        unit="sym",
        leave=True,
        file=sys.stderr,
        dynamic_ncols=True,
        mininterval=1.0,
        disable=False,
    )


def _scale_cum_np(
    probs: Sequence[float], *, alphabet_size: int = 256
) -> tuple["object", int]:
    """Map probs → cumulative integer masses as ``np.ndarray[int64]`` (len V+1)."""
    import numpy as np

    n = int(alphabet_size)
    if len(probs) != n:
        raise ValueError(f"probs_fn must return {n} probabilities, got {len(probs)}")
    p = np.asarray(probs, dtype=np.float64).reshape(n)
    # Match ``int(p * 1e6 + 0.5)`` for non-negative p.
    scaled_arr = np.maximum(1, np.floor(p * 1_000_000.0 + 0.5).astype(np.int64))
    cum_arr = np.empty(n + 1, dtype=np.int64)
    cum_arr[0] = 0
    np.cumsum(scaled_arr, out=cum_arr[1:])
    return cum_arr, int(cum_arr[n])


def _scale_cum(
    probs: Sequence[float], *, alphabet_size: int = 256
) -> tuple[list[int], int]:
    """Map probs → cumulative integer masses (≥1 each) and total (list API)."""
    n = int(alphabet_size)
    if len(probs) != n:
        raise ValueError(f"probs_fn must return {n} probabilities, got {len(probs)}")
    try:
        import numpy as np

        if isinstance(probs, np.ndarray):
            cum_arr, total = _scale_cum_np(probs, alphabet_size=alphabet_size)
            return cum_arr.tolist(), total
    except Exception:
        pass
    scaled = [max(1, int(p * 1_000_000 + 0.5)) for p in probs]
    total = sum(scaled)
    cum = [0]
    for s in scaled:
        cum.append(cum[-1] + s)
    return cum, total


def _renorm_encode(low: int, high: int, pending: int, out: bytearray) -> tuple[int, int, int]:
    while True:
        if high < _HALF:
            out.append(0)
            while pending:
                out.append(1)
                pending -= 1
        elif low >= _HALF:
            out.append(1)
            while pending:
                out.append(0)
                pending -= 1
            low -= _HALF
            high -= _HALF
        elif low >= _QUARTER and high < 3 * _QUARTER:
            pending += 1
            low -= _QUARTER
            high -= _QUARTER
        else:
            break
        low <<= 1
        high = (high << 1) | 1
    return low, high, pending


def _pack_bits(bits: bytearray) -> bytes:
    out = bytearray()
    acc = 0
    n = 0
    for b in bits:
        acc = (acc << 1) | (b & 1)
        n += 1
        if n == 8:
            out.append(acc)
            acc = 0
            n = 0
    if n:
        out.append(acc << (8 - n))
    return bytes(out)


def _bit_iter(data: bytes):
    """Yield MSB-first bits; then zeros (encode pads the last byte)."""
    for byte in data:
        for i in range(7, -1, -1):
            yield (byte >> i) & 1
    while True:
        yield 0


def encode_with_probs(
    symbols: Symbols,
    probs_fn: ProbsFn,
    *,
    desc: str | None = None,
    alphabet_size: int = 256,
) -> bytes:
    """Encode ``symbols`` given ``probs_fn(i, prefix) -> p[0..V)`` (sums to 1).

    If ``desc`` is set, show a tqdm bar. For ``alphabet_size==256`` and ``bytes``
    input, prefix is a ``memoryview`` slice (no per-byte ``bytes`` copy).
    Hot path keeps CDF as ``np.ndarray`` (no per-symbol ``.tolist()``).
    """
    import numpy as np

    low, high, pending = 0, _TOP - 1, 0
    bits = bytearray()
    n = len(symbols)
    use_mv = alphabet_size == 256 and isinstance(symbols, (bytes, bytearray, memoryview))
    mv = memoryview(symbols) if use_mv else None
    sym_arr = None
    if isinstance(symbols, np.ndarray):
        sym_arr = np.asarray(symbols).reshape(-1)
    if desc is not None:
        print(f"[{desc}] starting ({n} sym)…", file=sys.stderr, flush=True)
    for i in _progress_range(n, desc):
        sym = int(sym_arr[i]) if sym_arr is not None else int(symbols[i])
        if sym < 0 or sym >= alphabet_size:
            raise ValueError(f"symbol {sym} out of alphabet [0, {alphabet_size})")
        prefix: Prefix = mv[:i] if mv is not None else symbols[:i]
        probs = probs_fn(i, prefix)
        if isinstance(probs, np.ndarray):
            cum, total = _scale_cum_np(probs, alphabet_size=alphabet_size)
        else:
            cum, total = _scale_cum(probs, alphabet_size=alphabet_size)
        sym_low = int(cum[sym])
        sym_high = int(cum[sym + 1])
        range_ = high - low + 1
        high = low + (range_ * sym_high) // total - 1
        low = low + (range_ * sym_low) // total
        low, high, pending = _renorm_encode(low, high, pending, bits)

    # Flush
    pending += 1
    if low < _QUARTER:
        bits.append(0)
        while pending:
            bits.append(1)
            pending -= 1
    else:
        bits.append(1)
        while pending:
            bits.append(0)
            pending -= 1

    return _pack_bits(bits)


def decode_with_probs(
    bitstream: bytes,
    n_symbols: int,
    probs_fn: ProbsFn,
    *,
    desc: str | None = None,
    out: bytearray | list | None = None,
    hasher: "hashlib._Hash | None" = None,
    return_bytes: bool = True,
    alphabet_size: int = 256,
) -> bytes | bytearray | list[int]:
    """Decode ``n_symbols`` given the same ``probs_fn`` used at encode.

    For ``alphabet_size==256``, writes a ``bytearray`` (or ``bytes`` if
    ``return_bytes``). Larger alphabets return ``list[int]`` token ids.
    """
    import numpy as np

    bits = _bit_iter(bitstream)

    def read_bit() -> int:
        return next(bits)

    low, high = 0, _TOP - 1
    value = 0
    for _ in range(_CODE_BITS):
        value = (value << 1) | read_bit()

    wide = int(alphabet_size) > 256
    if out is not None:
        buf = out
    elif wide:
        buf = []
    else:
        buf = bytearray()
    if desc is not None:
        print(f"[{desc}] starting ({n_symbols} sym)…", file=sys.stderr, flush=True)
    for i in _progress_range(n_symbols, desc):
        # Pass the live buffer (no ``bytes(out)`` copy each step).
        probs = probs_fn(i, buf)
        if isinstance(probs, np.ndarray):
            cum, total = _scale_cum_np(probs, alphabet_size=alphabet_size)
            range_ = high - low + 1
            offset = ((value - low + 1) * total - 1) // range_
            # Largest sym with cum[sym] <= offset (searchsorted on cum[1:]).
            sym = int(np.searchsorted(cum, offset, side="right") - 1)
            sym = max(0, min(int(alphabet_size) - 1, sym))
        else:
            cum, total = _scale_cum(probs, alphabet_size=alphabet_size)
            range_ = high - low + 1
            offset = ((value - low + 1) * total - 1) // range_
            sym = 0
            lo, hi = 0, int(alphabet_size)
            while lo < hi:
                mid = (lo + hi) // 2
                if cum[mid] <= offset:
                    sym = mid
                    lo = mid + 1
                else:
                    hi = mid
        sym_low = int(cum[sym])
        sym_high = int(cum[sym + 1])
        high = low + (range_ * sym_high) // total - 1
        low = low + (range_ * sym_low) // total
        buf.append(sym)
        if hasher is not None:
            if wide:
                hasher.update(int(sym).to_bytes(2, "little"))
            else:
                hasher.update(bytes((sym,)))

        while True:
            if high < _HALF:
                pass
            elif low >= _HALF:
                low -= _HALF
                high -= _HALF
                value -= _HALF
            elif low >= _QUARTER and high < 3 * _QUARTER:
                low -= _QUARTER
                high -= _QUARTER
                value -= _QUARTER
            else:
                break
            low <<= 1
            high = (high << 1) | 1
            value = (value << 1) | read_bit()

    if wide:
        return list(buf)
    return bytes(buf) if return_bytes else buf


def round_trip_with_probs(
    symbols: Symbols,
    encode_probs_fn: ProbsFn,
    decode_probs_fn: ProbsFn | None = None,
    *,
    progress: bool = False,
    alphabet_size: int = 256,
) -> tuple[bytes, bytes | list[int], bool]:
    """Encode then decode; return ``(bitstream, decoded, ok)``."""
    dec_fn = decode_probs_fn if decode_probs_fn is not None else encode_probs_fn
    enc_desc = "AC encode" if progress else None
    dec_desc = "AC decode" if progress else None
    bitstream = encode_with_probs(
        symbols, encode_probs_fn, desc=enc_desc, alphabet_size=alphabet_size
    )
    decoded = decode_with_probs(
        bitstream,
        len(symbols),
        dec_fn,
        desc=dec_desc,
        alphabet_size=alphabet_size,
    )
    if alphabet_size > 256:
        import numpy as np

        ok = bool(
            np.array_equal(
                np.asarray(decoded, dtype=np.int64).reshape(-1),
                np.asarray(symbols, dtype=np.int64).reshape(-1),
            )
        )
    else:
        ok = decoded == (
            bytes(symbols)
            if not isinstance(symbols, (bytes, bytearray))
            else symbols
        )
    return bitstream, decoded, ok


def encode_segment(
    symbols: Symbols,
    probs_rows: "np.ndarray",
    *,
    alphabet_size: int = 256,
) -> bytes:
    """Encode one segment from precomputed ``probs_rows`` (n, V) — no decode."""
    import numpy as np

    n = len(symbols)
    v = int(alphabet_size)
    if probs_rows.shape != (n, v):
        raise ValueError(f"probs_rows shape {probs_rows.shape} != ({n}, {v})")
    rows = np.asarray(probs_rows)

    def probs_fn(i: int, _prefix):
        return rows[i]

    return encode_with_probs(
        symbols, probs_fn, desc=None, alphabet_size=alphabet_size
    )


def encode_decode_segment(
    symbols: Symbols,
    probs_rows: "np.ndarray",
    *,
    alphabet_size: int = 256,
) -> tuple[bytes, bytes | list[int]]:
    """Encode and decode one segment from precomputed ``probs_rows`` (n, V).

    Used for full-corpus SHA checks without holding all chunk probs in RAM:
    each retrain window is coded independently (small flush overhead).
    """
    import numpy as np

    n = len(symbols)
    v = int(alphabet_size)
    if probs_rows.shape != (n, v):
        raise ValueError(f"probs_rows shape {probs_rows.shape} != ({n}, {v})")
    rows = np.asarray(probs_rows)

    def probs_fn(i: int, _prefix):
        return rows[i]

    payload = encode_with_probs(
        symbols, probs_fn, desc=None, alphabet_size=alphabet_size
    )
    decoded = decode_with_probs(
        payload, n, probs_fn, desc=None, alphabet_size=alphabet_size
    )
    return payload, decoded


def ideal_bits_from_logprobs(nll_nats: float) -> float:
    """Convert mean NLL in nats to bits (total bits = nll_nats / ln(2) * N when mean)."""
    return float(nll_nats) / math.log(2.0)
