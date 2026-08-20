"""Capture-safe packed-KV commit for the W=1 decode-row CUDA graph."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .model import StaticState
    from .persistent_step import _PersistWS

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _commit_packed_kv(
        K_PACK,
        V_PACK,
        POS,
        MAX_LEN,
        HD,
        N_ITEM,
        BLOCK: tl.constexpr,
    ):
        item = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = item < N_ITEM
        group = item // HD
        dim = item - group * HD
        pos = tl.load(POS).to(tl.int64)
        group64 = group.to(tl.int64)
        dim64 = dim.to(tl.int64)
        src = (group64 * MAX_LEN + (MAX_LEN - 1)) * HD + dim64
        dst = (group64 * MAX_LEN + pos) * HD + dim64
        kval = tl.load(K_PACK + src, mask=mask, other=0.0)
        vval = tl.load(V_PACK + src, mask=mask, other=0.0)
        tl.store(K_PACK + dst, kval, mask=mask)
        tl.store(V_PACK + dst, vval, mask=mask)

else:  # pragma: no cover
    _commit_packed_kv = None


def can_fused_row_commit(ws: "_PersistWS | None") -> bool:
    return ws is not None and hasattr(ws, "k_pack") and hasattr(ws, "v_pack")


def commit_packed_row(ws: "_PersistWS", state: "StaticState") -> bool:
    """Copy each slot's tail K/V to ``state.pos``.

    Default is ``index_copy_`` on the same permuted views ``commit_window``
    writes. That is bit-identical to ``run_decode_row``. The flat Triton
    ``k_pack`` walk is opt-in (``XSA_AC_TRITON_COMMIT=1``).
    """
    if not can_fused_row_commit(ws):
        return False
    if int(state.win) != 1:
        return False
    want_triton = os.environ.get("XSA_AC_TRITON_COMMIT", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if want_triton and _HAS_TRITON and _commit_packed_kv is not None:
        n_item = int(ws.n_slots) * int(ws.n_kv) * int(ws.hd)
        block = 256
        _commit_packed_kv[(triton.cdiv(n_item, block),)](
            ws.k_pack,
            ws.v_pack,
            state.pos,
            ws.max_len,
            ws.hd,
            n_item,
            BLOCK=block,
            num_warps=4,
            num_stages=1,
        )
        return True
    dest = state.pos
    for sk in state.slots:
        if sk.k is None:
            continue
        assert sk.v is not None
        sk.k.index_copy_(1, dest, sk.k[:, -1:])
        sk.v.index_copy_(1, dest, sk.v[:, -1:])
    return True


def advance_row_state(
    state: "StaticState",
    symbol: torch.Tensor,
    decoded: torch.Tensor,
    out_index: int,
) -> None:
    """Promote smear/token state and increment the device position.

    Same ops as ``run_decode_row`` (``copy_`` / ``add_``).
    """
    state.prev_raw.copy_(state.cand_prev_raw)
    sym = symbol.view(-1)[:1].to(dtype=state.token.dtype)
    state.token.view(-1)[:1].copy_(sym)
    decoded[int(out_index) : int(out_index) + 1].copy_(sym)
    state.pos.add_(1)
