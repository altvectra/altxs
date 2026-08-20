"""Load the locked payload_sim byte stream (V=256) for AC encode/decode."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .config import EXPECTED_587_BYTES

if TYPE_CHECKING:
    import numpy as np

_SRC = Path(__file__).resolve().parents[1]
_REPO = _SRC.parent
# Match blsmc prepare seal (avoid importing bpe → lzma on slim Pythons).
M3_SIDE_FOOTER = b"M3SIDFTR"
META_FOOTER = b"BLSMETA1"

# Locked LTCB path is the byte payload_sim product (V=256). BPE leftovers
# are accepted if present but are not the ranked encode/decode stream.
_BYTE_CANDIDATES = (
    "data/enwik9.blsmc_full.m3v2.payload_sim",
    "data/payload_sim.product",
    "data/enwik9",
)

_BPE_CANDIDATES: tuple[str, ...] = ()

_BPE_NAME_RE = re.compile(r"\.bpe(\d+)$")


@dataclass(frozen=True)
class CorpusInfo:
    path: Path
    kind: str  # "bpe_tokens" | "bytes"
    vocab_size: int
    n_symbols: int
    source_bytes: int | None
    meta: dict


def _sidecar_for(path: Path) -> Path | None:
    cand = Path(str(path) + ".json")
    if cand.is_file():
        return cand
    return None


def _vocab_from_bpe_name(name: str) -> int | None:
    m = _BPE_NAME_RE.search(name)
    return int(m.group(1)) if m else None


def _is_bpe_token_path(path: Path, meta: dict) -> bool:
    if _vocab_from_bpe_name(path.name) is not None:
        return True
    return meta.get("format") in ("xsa_ttt-bpe-v1", "blsmc-bpe-v1")


def resolve_data_path(explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        p = Path(explicit)
        if not p.is_file():
            raise FileNotFoundError(
                f"data not found: {p}\n"
                "Peel enwik9 with scripts/encode.sh (or pass --data / XSA_DATA)."
            )
        return p
    env = os.environ.get("XSA_DATA")
    if env:
        p = Path(env)
        if p.is_file():
            return p
    for rel in _BPE_CANDIDATES:
        p = _REPO / rel
        if p.is_file():
            return p
    for rel in _BYTE_CANDIDATES:
        p = _REPO / rel
        if p.is_file():
            return p
    raw = _REPO / "data" / "enwik9"
    if raw.is_file():
        return raw
    raise FileNotFoundError(
        "No corpus found. Expected data/enwik9.blsmc_full.m3v2.payload_sim "
        "or data/enwik9.\n"
        "  ./scripts/fetch_enwik9.sh && ./scripts/encode.sh --peel-only\n"
        "Or set XSA_DATA / pass --data."
    )


def load_meta(path: Path) -> dict:
    side = _sidecar_for(path)
    if side is None:
        return {}
    return json.loads(side.read_text(encoding="utf8"))


def describe_corpus(path: Path) -> CorpusInfo:
    meta = load_meta(path)
    if _is_bpe_token_path(path, meta):
        if meta.get("encode_pending"):
            raise FileNotFoundError(
                f"BPE dict ready but tokens missing: {path}\n"
                "  python -m xsa_ttt.prepare_bpe --src <product> --encode-only --no-verify"
            )
        name_v = _vocab_from_bpe_name(path.name)
        vocab = int(meta.get("vocab_size", name_v or 4096))
        n = path.stat().st_size // 2
        if meta.get("n_tokens") is not None:
            n = int(meta["n_tokens"])
        return CorpusInfo(
            path=path,
            kind="bpe_tokens",
            vocab_size=vocab,
            n_symbols=n,
            source_bytes=meta.get("source_bytes"),
            meta=meta,
        )
    n = _byte_stream_len(path)
    return CorpusInfo(
        path=path,
        kind="bytes",
        vocab_size=256,
        n_symbols=n,
        source_bytes=meta.get("source_bytes", n),
        meta=meta,
    )


def _byte_stream_len(path: Path) -> int:
    """File size minus BLSMETA1 + M3 densify trailers when present."""
    file_n = path.stat().st_size
    end = file_n

    def _strip_footer(end_n: int, foot: bytes) -> int:
        need = len(foot) + 8
        if end_n < need:
            return end_n
        with open(path, "rb") as f:
            f.seek(end_n - need)
            tail = f.read(need)
        if tail[: len(foot)] != foot:
            return end_n
        blob_len = int.from_bytes(tail[len(foot) :], "little")
        stream_n = end_n - need - blob_len
        if stream_n < 0 or blob_len > end_n - need:
            return end_n
        return stream_n

    end = _strip_footer(end, META_FOOTER)
    end = _strip_footer(end, M3_SIDE_FOOTER)
    return end


def load_symbols(
    path: str | Path | None = None,
    *,
    max_symbols: int | None = None,
    mmap: bool = True,
) -> tuple[np.ndarray, CorpusInfo]:
    """Load token ids (uint16) or bytes (uint8). Returns (array, info)."""
    import numpy as np

    p = resolve_data_path(path)
    info = describe_corpus(p)
    if info.kind == "bpe_tokens":
        n = info.n_symbols
        if max_symbols is not None:
            n = min(n, int(max_symbols))
        if mmap:
            arr = np.memmap(p, dtype="<u2", mode="r")[:n]
        else:
            arr = np.frombuffer(p.read_bytes()[: n * 2], dtype="<u2").copy()
        return arr, info

    n = int(info.n_symbols)
    file_n = p.stat().st_size
    if max_symbols is not None:
        n = min(n, int(max_symbols))
    if mmap:
        arr = np.memmap(p, dtype=np.uint8, mode="r")[:n]
    else:
        with open(p, "rb") as f:
            arr = np.frombuffer(f.read(n), dtype=np.uint8).copy()
    meta = dict(info.meta)
    if n < file_n:
        meta["m3_trailer_stripped"] = int(file_n - n)
    info = CorpusInfo(
        path=p,
        kind="bytes",
        vocab_size=256,
        n_symbols=int(n),
        source_bytes=int(n),
        meta=meta,
    )
    return arr, info


def load_bytes(
    path: str | Path | None = None,
    *,
    max_bytes: int | None = None,
    mmap: bool = True,
) -> np.ndarray:
    """Back-compat: load symbols (bytes or BPE tokens) as a 1-D array."""
    arr, _info = load_symbols(path, max_symbols=max_bytes, mmap=mmap)
    return arr


def describe_data(path: Path) -> str:
    info = describe_corpus(path)
    if info.kind == "bpe_tokens":
        src = (
            f", source_bytes={info.source_bytes:,}"
            if info.source_bytes is not None
            else ""
        )
        return (
            f"{path} (BPE tokens n={info.n_symbols:,}, vocab={info.vocab_size}{src}; "
            "payload_sim/dense-peel → BPE → xsa_ttt)"
        )
    n = info.n_symbols
    if n == EXPECTED_587_BYTES:
        note = "cmix-lex post–payload_lex checkpoint (legacy byte path)"
    elif n == 1_000_000_000:
        note = "WARNING: raw enwik9 (1 GB)"
    elif path.name.endswith(".dense_peel"):
        note = "dense-peel product (pre-BPE)"
    else:
        note = "byte stream"
    return f"{path} ({n:,} B; {note})"
