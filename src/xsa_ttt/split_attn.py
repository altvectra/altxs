"""Shared split-prefix attention for W=1 decode and W>1 encode.

The old attention kernel assigns one CTA to each query head and serially
scans the entire prefix.  This path partitions that scan across several
CTAs, writes fp32 online-softmax partials, then combines the partitions in
a fixed left-to-right order.  Encode and decode call these exact kernels so
their integer arithmetic-coder frequencies remain lockstep.
"""

from __future__ import annotations

import os
from typing import Any

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _HAS_TRITON = False


def attn_splits() -> int:
    """Power-of-two prefix partitions; one restores the serial kernel."""
    if os.environ.get("XSA_AC_ATTN_DOT", "1").strip().lower() in {
        "0",
        "false",
        "off",
        "no",
        "",
    }:
        return 1
    try:
        requested = int(os.environ.get("XSA_AC_ATTN_SPLITS", "8"))
    except ValueError:
        requested = 8
    requested = min(max(requested, 1), 8)
    value = 1
    while value < requested:
        value *= 2
    return value


def can_split_attn() -> bool:
    return _HAS_TRITON and attn_splits() > 1


if _HAS_TRITON:

    @triton.jit
    def _split_attn_part(
        Q,
        K_PACK,
        V_PACK,
        PART_M,
        PART_LSE,
        PART_ACC,
        POS,
        slot,
        rows,
        MAX_LEN,
        HD,
        N_HEADS,
        N_KV,
        BLOCK_H: tl.constexpr,
        BLOCK_L: tl.constexpr,
        N_SPLITS: tl.constexpr,
    ):
        head = tl.program_id(0)
        row = tl.program_id(1)
        split = tl.program_id(2)
        if (head >= N_HEADS) | (row >= rows) | (split >= N_SPLITS):
            return
        pos = tl.load(POS).to(tl.int32) + row
        kv_h = head // (N_HEADS // N_KV)
        ho = tl.arange(0, BLOCK_H)
        hm = ho < HD
        qv = tl.load(
            Q + (row * N_HEADS + head) * HD + ho, mask=hm, other=0.0
        ).to(tl.float32)
        scale = 1.0 / tl.sqrt(tl.cast(HD, tl.float32))
        col16 = tl.arange(0, 16)
        q_tc = tl.zeros([BLOCK_H, 16], dtype=tl.bfloat16)
        q_tc = tl.where(
            (col16[None, :] == 0) & hm[:, None],
            qv.to(tl.bfloat16)[:, None],
            q_tc,
        )

        n_tiles = tl.cdiv(pos, BLOCK_L)
        active = tl.minimum(N_SPLITS, tl.maximum(n_tiles, 1))
        tile0 = (split * n_tiles) // active
        tile1 = ((split + 1) * n_tiles) // active
        tile0 = tl.where(split < active, tile0, n_tiles)
        tile1 = tl.where(split < active, tile1, n_tiles)
        t0 = tile0 * BLOCK_L
        t_end = tl.minimum(tile1 * BLOCK_L, pos)
        pm = -1.0e9
        pl = 0.0
        pa = tl.zeros([BLOCK_H], dtype=tl.float32)
        while t0 < t_end:
            to = t0 + tl.arange(0, BLOCK_L)
            tm = to < t_end
            kp = (
                ((slot * N_KV + kv_h) * MAX_LEN + to[:, None]) * HD
                + ho[None, :]
            )
            mask = tm[:, None] & hm[None, :]
            kv = tl.load(K_PACK + kp, mask=mask, other=0.0)
            vv = tl.load(V_PACK + kp, mask=mask, other=0.0)
            s16 = tl.dot(kv.to(tl.bfloat16), q_tc)
            scores = scale * tl.sum(
                s16.to(tl.float32)
                * (col16[None, :] == 0).to(tl.float32),
                axis=1,
            )
            scores = tl.where(tm, scores, -1.0e9)
            mt = tl.max(scores, axis=0)
            mn = tl.maximum(pm, mt)
            alpha = tl.exp(pm - mn)
            probs = tl.exp(scores - mn)
            probs = tl.where(tm, probs, 0.0)
            pl = pl * alpha + tl.sum(probs, axis=0)
            p_tc = tl.zeros([16, BLOCK_L], dtype=tl.bfloat16)
            p_tc = tl.where(
                col16[:, None] == 0,
                probs.to(tl.bfloat16)[None, :],
                p_tc,
            )
            acc16 = tl.dot(p_tc, vv.to(tl.bfloat16))
            pv = tl.sum(
                acc16.to(tl.float32)
                * (col16[:, None] == 0).to(tl.float32),
                axis=0,
            )
            pa = pa * alpha + pv
            pm = mn
            t0 += BLOCK_L

        pi = (row * N_HEADS + head) * N_SPLITS + split
        tl.store(PART_M + pi, pm)
        tl.store(PART_LSE + pi, pl)
        tl.store(
            PART_ACC + pi * HD + ho,
            pa,
            mask=hm,
        )

    @triton.jit
    def _split_attn_reduce(
        Q,
        K_PACK,
        V_PACK,
        Y,
        V_CUR,
        ATTN_IN,
        GATE_W,
        PART_M,
        PART_LSE,
        PART_ACC,
        POS,
        slot,
        rows,
        MAX_LEN,
        HD,
        N_HEADS,
        N_KV,
        GATE_N,
        D,
        layer,
        use_xsa,
        use_gate,
        NEPS,
        V_CUR_ROW_STRIDE,
        TAIL_AT_POS: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_L: tl.constexpr,
        BLOCK_G: tl.constexpr,
        N_SPLITS: tl.constexpr,
    ):
        head = tl.program_id(0)
        row = tl.program_id(1)
        if (head >= N_HEADS) | (row >= rows):
            return
        pos = tl.load(POS).to(tl.int32) + row
        kv_h = head // (N_HEADS // N_KV)
        ho = tl.arange(0, BLOCK_H)
        hm = ho < HD
        qv = tl.load(
            Q + (row * N_HEADS + head) * HD + ho, mask=hm, other=0.0
        ).to(tl.float32)
        scale = 1.0 / tl.sqrt(tl.cast(HD, tl.float32))
        m = -1.0e9
        lse = 0.0
        acc = tl.zeros([BLOCK_H], dtype=tl.float32)
        n_tiles = tl.cdiv(pos, BLOCK_L)
        # One tile (pos < BLOCK_L) must match serial mega-dot bit-for-bit.
        # Merging a single MMA partial compiled in the 3D part-grid drifted
        # on the second layer-5 decode visit (probe step 1 slot 11).
        if n_tiles <= 1:
            col16 = tl.arange(0, 16)
            q_tc = tl.zeros([BLOCK_H, 16], dtype=tl.bfloat16)
            q_tc = tl.where(
                (col16[None, :] == 0) & hm[:, None],
                qv.to(tl.bfloat16)[:, None],
                q_tc,
            )
            t0 = 0
            while t0 < pos:
                to = t0 + tl.arange(0, BLOCK_L)
                tm = to < pos
                kp = (
                    ((slot * N_KV + kv_h) * MAX_LEN + to[:, None]) * HD
                    + ho[None, :]
                )
                mask = tm[:, None] & hm[None, :]
                kv = tl.load(K_PACK + kp, mask=mask, other=0.0)
                vv = tl.load(V_PACK + kp, mask=mask, other=0.0)
                s16 = tl.dot(kv.to(tl.bfloat16), q_tc)
                scores = scale * tl.sum(
                    s16.to(tl.float32)
                    * (col16[None, :] == 0).to(tl.float32),
                    axis=1,
                )
                scores = tl.where(tm, scores, -1.0e9)
                mt = tl.max(scores, axis=0)
                mn = tl.maximum(m, mt)
                alpha = tl.exp(m - mn)
                probs = tl.exp(scores - mn)
                probs = tl.where(tm, probs, 0.0)
                lse = lse * alpha + tl.sum(probs, axis=0)
                p_tc = tl.zeros([16, BLOCK_L], dtype=tl.bfloat16)
                p_tc = tl.where(
                    col16[:, None] == 0,
                    probs.to(tl.bfloat16)[None, :],
                    p_tc,
                )
                acc16 = tl.dot(p_tc, vv.to(tl.bfloat16))
                pv = tl.sum(
                    acc16.to(tl.float32)
                    * (col16[:, None] == 0).to(tl.float32),
                    axis=0,
                )
                acc = acc * alpha + pv
                m = mn
                t0 += BLOCK_L
        else:
            active = tl.minimum(N_SPLITS, n_tiles)
            pi0 = (row * N_HEADS + head) * N_SPLITS
            m = tl.load(PART_M + pi0)
            lse = tl.load(PART_LSE + pi0)
            acc = tl.load(PART_ACC + pi0 * HD + ho, mask=hm, other=0.0)
            for split in range(1, N_SPLITS):
                if split < active:
                    pi = pi0 + split
                    pm = tl.load(PART_M + pi)
                    pl = tl.load(PART_LSE + pi)
                    pa = tl.load(
                        PART_ACC + pi * HD + ho, mask=hm, other=0.0
                    )
                    mn = tl.maximum(m, pm)
                    alpha = tl.exp(m - mn)
                    beta = tl.exp(pm - mn)
                    lse = lse * alpha + pl * beta
                    acc = acc * alpha + pa * beta
                    m = mn

        tail = MAX_LEN - 1
        if TAIL_AT_POS:
            tail = pos
        kp = ((slot * N_KV + kv_h) * MAX_LEN + tail) * HD + ho
        k_cur = tl.load(K_PACK + kp, mask=hm, other=0.0).to(tl.float32)
        v_cur = tl.load(V_PACK + kp, mask=hm, other=0.0).to(tl.float32)
        score = scale * tl.sum(qv * k_cur, axis=0)
        mn = tl.maximum(m, score)
        alpha = tl.exp(m - mn)
        prob = tl.exp(score - mn)
        lse = lse * alpha + prob
        acc = acc * alpha + prob * v_cur
        yv = acc / lse
        if use_xsa != 0:
            vb = tl.load(
                V_CUR + row * V_CUR_ROW_STRIDE + kv_h * HD + ho,
                mask=hm,
                other=0.0,
            ).to(tl.float32)
            vnrm = tl.maximum(tl.sqrt(tl.sum(vb * vb, axis=0)), NEPS)
            vn = vb / vnrm
            yv = yv - tl.sum(yv * vn, axis=0) * vn
        if use_gate != 0:
            go = tl.arange(0, BLOCK_G)
            gm = go < GATE_N
            xv = tl.load(
                ATTN_IN + row * D + go, mask=gm, other=0.0
            ).to(tl.float32)
            gw = tl.load(
                GATE_W + (layer * N_HEADS + head) * GATE_N + go,
                mask=gm,
                other=0.0,
            ).to(tl.float32)
            yv = yv * (1.0 / (1.0 + tl.exp(-tl.sum(xv * gw, axis=0))))
        tl.store(
            Y + (row * N_HEADS + head) * HD + ho,
            yv.to(Y.dtype.element_ty),
            mask=hm,
        )

else:  # pragma: no cover
    _split_attn_part = None
    _split_attn_reduce = None


def ensure_split_workspace(ws: Any, rows: int) -> None:
    """Allocate static fp32 partials once on an encode or decode workspace."""
    if not can_split_attn():
        return
    splits = attn_splits()
    ws.attn_splits = splits
    shape = (int(rows), int(ws.n_heads), splits)
    current = getattr(ws, "split_m", None)
    if current is not None and tuple(current.shape) == shape:
        return
    device = ws.q.device
    ws.split_m = torch.zeros(shape, device=device, dtype=torch.float32)
    ws.split_lse = torch.zeros_like(ws.split_m)
    ws.split_acc = torch.zeros(
        *shape, int(ws.hd), device=device, dtype=torch.float32
    )
    from .model import _mark_static_address

    _mark_static_address(ws.split_m)
    _mark_static_address(ws.split_lse)
    _mark_static_address(ws.split_acc)


def run_split_attn(
    *,
    ws: Any,
    pos: torch.Tensor,
    slot: int,
    rows: int,
    layer: int,
    use_xsa: int,
    use_gate: int,
    v_cur: torch.Tensor,
    v_cur_row_stride: int,
    tail_at_pos: bool,
    norm_eps: float,
    block_l: int,
) -> bool:
    """Launch shared split partial/reduce kernels. False means use serial."""
    splits = int(getattr(ws, "attn_splits", 1))
    if not _HAS_TRITON or splits <= 1:
        return False
    ensure_split_workspace(ws, rows)
    splits = int(ws.attn_splits)
    block_h = 1
    while block_h < int(ws.hd):
        block_h *= 2
    block_g = 1
    while block_g < max(int(ws.gate_n), 1):
        block_g *= 2
    # Same warp/stage as ``_mega_attn_dot`` / ``_enc_attn_dot``. 4-warp
    # reduce compiled a different MMA pipeline and drifted on layer-5
    # decode (probe step 1 slot 11) even when the math matched.
    launch = dict(num_warps=8, num_stages=3)
    _split_attn_part[(ws.n_heads, rows, splits)](
        ws.q,
        ws.k_pack,
        ws.v_pack,
        ws.split_m,
        ws.split_lse,
        ws.split_acc,
        pos,
        slot,
        rows,
        ws.max_len,
        ws.hd,
        ws.n_heads,
        ws.n_kv,
        BLOCK_H=block_h,
        BLOCK_L=block_l,
        N_SPLITS=splits,
        **launch,
    )
    # W=1: 1-D grid matches mega-dot ``(n_heads,)``. pid1 is 0.
    reduce_grid = (ws.n_heads,) if int(rows) == 1 else (ws.n_heads, rows)
    _split_attn_reduce[reduce_grid](
        ws.q,
        ws.k_pack,
        ws.v_pack,
        ws.y,
        v_cur,
        ws.attn_in,
        ws.gate_w,
        ws.split_m,
        ws.split_lse,
        ws.split_acc,
        pos,
        slot,
        rows,
        ws.max_len,
        ws.hd,
        ws.n_heads,
        ws.n_kv,
        ws.gate_n,
        ws.d,
        layer,
        use_xsa,
        use_gate,
        norm_eps,
        v_cur_row_stride,
        TAIL_AT_POS=tail_at_pos,
        BLOCK_H=block_h,
        BLOCK_L=block_l,
        BLOCK_G=block_g,
        N_SPLITS=splits,
        **launch,
    )
    return True
