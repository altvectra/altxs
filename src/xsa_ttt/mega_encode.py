"""W>1 encode megakernel whose per-row numerics match W=1 decode mega.

Separate from ``mega_step.py`` (decode stays the W=1 8-CTA path). Encode
teacher-forces a full window:

  * embed: sequential ``persist_embed`` (PyTorch smear + 1D RMS) per row;
  * per slot: Triton prologue + QKV/RoPE copied from mega STAGE=2, grid
    ``(n_heads, W)``; GQA group leaders alone project shared K/V;
  * attn: shared split-prefix attention (default) with
    ``pos = win_start+row`` and self-key at that dest index, matching W=1
    decode's partial/reduction order. ``XSA_AC_ATTN_SPLITS=1`` restores the
    serial ``_enc_attn_dot`` path. GROUP=4 is one CTA × 4 rows there:
    shared ``BLOCK_L`` walk, four independent padded-16 ``tl.dot`` scans
    (own K/V loads, same iteration count per row as ``_enc_attn_dot``).
    Do not pack Qs into one MMA. Do not GEMM Q[W] against K.
  * FFN / logits: default is tiled Triton GEMV shared with decode
    (``ac_gemv.gemv_rows``). ``XSA_AC_ENCODE_FFN=loop`` is 64× cuBLAS
    ``persist_ffn_slot`` (old gold, slow).

K/V for row r are written to prefix index ``win_start+r`` *before* that
row's attn (QKV kernel completes, then attn launches), so the scan sees
the same key order as W=1 after each commit. Tail ``max_len-W+r`` is
also written so ``commit_window`` stays valid. Do not share flags or
kernels with persist ``_attn_heads``.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from .model import _mark_static_address
from .persistent_step import (
    _RMS_EPS,
    _NORM_EPS,
    _attn_block_l,
    _itinerary,
    _launch_meta,
    _rms,
    _stack_weights,
    persist_ffn_slot,
    ffn_slot_rows,
    logits_rows,
)

if TYPE_CHECKING:
    from .model import GPT, StaticState

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _HAS_TRITON = False


def _env_on(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
        "",
    }


def _enc_attn_group(win: int) -> int:
    """4 = one ``_enc_attn_dot_g4`` launch (4 rows/CTA); else per-row ``_enc_attn_dot``."""
    try:
        g = int(os.environ.get("XSA_AC_ENC_ATTN_GROUP", "1"))
    except ValueError:
        g = 1
    if g == 4 and int(win) % 4 == 0:
        return 4
    return 1


def can_mega_encode(model: "GPT", state: "StaticState") -> bool:
    if not _HAS_TRITON or state.token.device.type != "cuda":
        return False
    if int(state.win) < 2:
        return False
    if not _env_on("XSA_AC_MEGA_ENCODE", "1"):
        return False
    if not _env_on("XSA_AC_MEGA", "1"):
        return False
    if getattr(model, "_ac_qkv_bank", None) is None:
        return False
    if getattr(model, "_encode_disabled", False):
        return False
    return True


class _FfnRowView:
    """1D views so encode FFN is literally ``persist_ffn_slot`` (cuBLAS GEMV)."""

    __slots__ = (
        "itin_host",
        "ln",
        "out_w",
        "a_sc",
        "m_sc",
        "pr",
        "pp",
        "x",
        "xin",
        "attn_in",
        "y",
        "lane0",
        "lane1",
        "skips",
    )


if _HAS_TRITON:

    @triton.jit
    def _enc_rms(vec, D, eps):
        return vec * tl.rsqrt(tl.sum(vec * vec, axis=0) / D + eps)

    @triton.jit
    def _enc_gemv_own(X, W, Y, n_out, k, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        eg_acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        eg_no = tl.arange(0, BLOCK_N)
        eg_nm = eg_no < n_out
        eg_kk = 0
        while eg_kk < k:
            eg_ko = eg_kk + tl.arange(0, BLOCK_K)
            eg_km = eg_ko < k
            eg_xv = tl.load(X + eg_ko, mask=eg_km, other=0.0).to(tl.float32)
            eg_wv = tl.load(
                W + eg_no[:, None] * k + eg_ko[None, :],
                mask=eg_nm[:, None] & eg_km[None, :],
                other=0.0,
            ).to(tl.float32)
            eg_acc += tl.sum(eg_wv * eg_xv[None, :], axis=1)
            eg_kk += BLOCK_K
        tl.store(Y + eg_no, eg_acc.to(Y.dtype.element_ty), mask=eg_nm)

    @triton.jit
    def _enc_gemv_row_tiles(
        X,
        W,
        Y,
        n_out,
        k,
        win,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """One GEMV per window row. Independent rows — not an M=W GEMM."""
        gvr_row = tl.program_id(0)
        gvr_tile = tl.program_id(1)
        if gvr_row >= win:
            return
        gvr_acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        gvr_no = gvr_tile * BLOCK_N + tl.arange(0, BLOCK_N)
        gvr_nm = gvr_no < n_out
        gvr_xp = X + gvr_row * k
        gvr_yp = Y + gvr_row * n_out
        gvr_kk = 0
        while gvr_kk < k:
            gvr_ko = gvr_kk + tl.arange(0, BLOCK_K)
            gvr_km = gvr_ko < k
            gvr_xv = tl.load(gvr_xp + gvr_ko, mask=gvr_km, other=0.0).to(tl.float32)
            gvr_wv = tl.load(
                W + gvr_no[:, None] * k + gvr_ko[None, :],
                mask=gvr_nm[:, None] & gvr_km[None, :],
                other=0.0,
            ).to(tl.float32)
            gvr_acc += tl.sum(gvr_wv * gvr_xv[None, :], axis=1)
            gvr_kk += BLOCK_K
        tl.store(
            gvr_yp + gvr_no, gvr_acc.to(Y.dtype.element_ty), mask=gvr_nm
        )

    @triton.jit
    def _enc_rope_head(VEC, COS, SIN, HD, RH, BLOCK: tl.constexpr):
        er_offs = tl.arange(0, BLOCK)
        er_lo = tl.load(VEC + er_offs, mask=er_offs < RH, other=0.0).to(tl.float32)
        er_hi = tl.load(VEC + RH + er_offs, mask=er_offs < RH, other=0.0).to(
            tl.float32
        )
        er_c = tl.load(COS + er_offs, mask=er_offs < RH, other=0.0).to(tl.float32)
        er_s = tl.load(SIN + er_offs, mask=er_offs < RH, other=0.0).to(tl.float32)
        tl.store(
            VEC + er_offs,
            (er_lo * er_c + er_hi * er_s).to(VEC.dtype.element_ty),
            mask=er_offs < RH,
        )
        tl.store(
            VEC + RH + er_offs,
            (-er_lo * er_s + er_hi * er_c).to(VEC.dtype.element_ty),
            mask=er_offs < RH,
        )

    @triton.jit
    def _enc_prologue(
        X,
        X0,
        XIN,
        ATTN_IN,
        LANE0,
        LANE1,
        SKIPS,
        MIX,
        LN,
        SKIP_W,
        SKIP_G,
        ITIN,
        slot,
        d,
        win,
        n_enc,
        has_skip_gate,
        BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0)
        if row >= win:
            return
        offs = tl.arange(0, BLOCK_D)
        dmask = offs < d
        layer = tl.load(ITIN + slot * 7 + 0)
        kind = tl.load(ITIN + slot * 7 + 1)
        skip_src = tl.load(ITIN + slot * 7 + 2)
        skip_wi = tl.load(ITIN + slot * 7 + 4)
        par = (kind == 2) | (kind == 3)
        x = tl.load(X + row * d + offs, mask=dmask, other=0.0).to(tl.float32)
        if kind == 2:
            tl.store(LANE0 + row * d + offs, x.to(LANE0.dtype.element_ty), mask=dmask)
            tl.store(LANE1 + row * d + offs, x.to(LANE1.dtype.element_ty), mask=dmask)
        if par:
            x = tl.load(LANE0 + row * d + offs, mask=dmask, other=0.0).to(tl.float32)
        if skip_src >= 0:
            sk = tl.load(
                SKIPS + (skip_src * win + row) * d + offs, mask=dmask, other=0.0
            ).to(tl.float32)
            skw = tl.load(SKIP_W + skip_wi * d + offs, mask=dmask, other=0.0).to(
                tl.float32
            )
            scaled = skw * sk
            if has_skip_gate != 0:
                gg = 1.0 / (
                    1.0
                    + tl.exp(
                        -tl.load(
                            SKIP_G + skip_wi * d + offs, mask=dmask, other=0.0
                        ).to(tl.float32)
                    )
                )
                x = scaled * (1.0 - gg) + x * gg
            else:
                x = x + scaled
        x0 = tl.load(X0 + row * d + offs, mask=dmask, other=0.0).to(tl.float32)
        m0 = tl.load(MIX + (layer * 2 + 0) * d + offs, mask=dmask, other=0.0).to(
            tl.float32
        )
        m1 = tl.load(MIX + (layer * 2 + 1) * d + offs, mask=dmask, other=0.0).to(
            tl.float32
        )
        xin = m0 * x + m1 * x0
        ln = tl.load(LN + layer)
        attn = _enc_rms(xin, d, 1e-6) * ln
        tl.store(XIN + row * d + offs, xin.to(XIN.dtype.element_ty), mask=dmask)
        tl.store(ATTN_IN + row * d + offs, attn.to(ATTN_IN.dtype.element_ty), mask=dmask)
        if par:
            tl.store(LANE0 + row * d + offs, x.to(LANE0.dtype.element_ty), mask=dmask)
        else:
            tl.store(X + row * d + offs, x.to(X.dtype.element_ty), mask=dmask)

    @triton.jit
    def _enc_qkv(
        ATTN_IN,
        Q,
        KBUF,
        VBUF,
        K_PACK,
        V_PACK,
        QKV,
        Q_GAIN,
        COS,
        SIN,
        ITIN,
        POS,
        VXSA,
        slot,
        d,
        hd,
        n_heads,
        n_kv,
        qkv_dim,
        max_len,
        win,
        rh,
        BLOCK_H: tl.constexpr,
        BLOCK_K: tl.constexpr,
        DEDUP_KV: tl.constexpr,
    ):
        head = tl.program_id(0)
        row = tl.program_id(1)
        if (head >= n_heads) | (row >= win):
            return
        layer = tl.load(ITIN + slot * 7 + 0)
        pos0 = tl.load(POS).to(tl.int32)
        dest = pos0 + row
        tail_i = max_len - win + row
        kv_h = head // (n_heads // n_kv)
        ho = tl.arange(0, BLOCK_H)
        hmask = ho < hd
        attn_ptr = ATTN_IN + row * d
        q_ptr = Q + (row * n_heads + head) * hd
        q_w = QKV + (layer * qkv_dim + head * hd) * d
        _enc_gemv_own(attn_ptr, q_w, q_ptr, hd, d, BLOCK_H, BLOCK_K)
        qraw = tl.load(q_ptr + ho, mask=hmask, other=0.0).to(tl.float32)
        qn = qraw * tl.rsqrt(tl.sum(qraw * qraw, axis=0) / hd + 1e-6)
        qn = qn * tl.load(Q_GAIN + layer * n_heads + head)
        tl.store(q_ptr + ho, qn.to(Q.dtype.element_ty), mask=hmask)
        _enc_rope_head(q_ptr, COS + dest * rh, SIN + dest * rh, hd, rh, BLOCK_H)
        ksc = KBUF + (row * n_heads + head) * hd
        vsc = VBUF + (row * n_heads + head) * hd
        group_leader = (head % (n_heads // n_kv)) == 0
        if (DEDUP_KV == 0) | group_leader:
            k_w = QKV + (layer * qkv_dim + d + kv_h * hd) * d
            v_w = QKV + (
                layer * qkv_dim + d + n_kv * hd + kv_h * hd
            ) * d
            _enc_gemv_own(attn_ptr, k_w, ksc, hd, d, BLOCK_H, BLOCK_K)
            _enc_gemv_own(attn_ptr, v_w, vsc, hd, d, BLOCK_H, BLOCK_K)
            kraw = tl.load(ksc + ho, mask=hmask, other=0.0).to(tl.float32)
            kn = kraw * tl.rsqrt(
                tl.sum(kraw * kraw, axis=0) / hd + 1e-6
            )
            tl.store(ksc + ho, kn.to(KBUF.dtype.element_ty), mask=hmask)
            _enc_rope_head(
                ksc, COS + dest * rh, SIN + dest * rh, hd, rh, BLOCK_H
            )
        if group_leader:
            ksrc = tl.load(ksc + ho, mask=hmask, other=0.0)
            vsrc = tl.load(vsc + ho, mask=hmask, other=0.0)
            tl.store(
                K_PACK + ((slot * n_kv + kv_h) * max_len + dest) * hd + ho,
                ksrc,
                mask=hmask,
            )
            tl.store(
                V_PACK + ((slot * n_kv + kv_h) * max_len + dest) * hd + ho,
                vsrc,
                mask=hmask,
            )
            tl.store(
                K_PACK + ((slot * n_kv + kv_h) * max_len + tail_i) * hd + ho,
                ksrc,
                mask=hmask,
            )
            tl.store(
                V_PACK + ((slot * n_kv + kv_h) * max_len + tail_i) * hd + ho,
                vsrc,
                mask=hmask,
            )
            tl.store(
                VXSA + (row * n_kv + kv_h) * hd + ho,
                vsrc,
                mask=hmask,
            )

    @triton.jit
    def _enc_attn(
        Q,
        K_PACK,
        V_PACK,
        Y,
        VXSA,
        ATTN_IN,
        GATE_W,
        POS,
        slot,
        win,
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
        BLOCK_H: tl.constexpr,
        BLOCK_L: tl.constexpr,
        BLOCK_G: tl.constexpr,
    ):
        head = tl.program_id(0)
        row = tl.program_id(1)
        if (head >= N_HEADS) | (row >= win):
            return
        pos0 = tl.load(POS).to(tl.int32)
        pos = pos0 + row
        tail = pos
        kv_h = head // (N_HEADS // N_KV)
        hoffs = tl.arange(0, BLOCK_H)
        hmask = hoffs < HD
        qv = tl.load(
            Q + (row * N_HEADS + head) * HD + hoffs, mask=hmask, other=0.0
        ).to(tl.float32)
        scale = 1.0 / tl.sqrt(tl.cast(HD, tl.float32))
        m = -1.0e9
        lse = 0.0
        acc = tl.zeros([BLOCK_H], dtype=tl.float32)
        t0 = 0
        while t0 < pos:
            toffs = t0 + tl.arange(0, BLOCK_L)
            tmask = toffs < pos
            t_b = toffs[:, None]
            d_b = hoffs[None, :]
            kptr = ((slot * N_KV + kv_h) * MAX_LEN + t_b) * HD + d_b
            vmask = tmask[:, None] & (d_b < HD)
            kv = tl.load(K_PACK + kptr, mask=vmask, other=0.0).to(tl.float32)
            vv = tl.load(V_PACK + kptr, mask=vmask, other=0.0).to(tl.float32)
            s = scale * tl.sum(kv * qv[None, :], axis=1)
            s = tl.where(tmask, s, -1.0e9)
            m_tile = tl.max(s, axis=0)
            m_new = tl.maximum(m, m_tile)
            alpha = tl.exp(m - m_new)
            p = tl.exp(s - m_new)
            p = tl.where(tmask, p, 0.0)
            lse = lse * alpha + tl.sum(p, axis=0)
            acc = acc * alpha + tl.sum(p[:, None] * vv, axis=0)
            m = m_new
            t0 += BLOCK_L
        kb = ((slot * N_KV + kv_h) * MAX_LEN + tail) * HD
        kv = tl.load(K_PACK + kb + hoffs, mask=hmask, other=0.0).to(tl.float32)
        vv = tl.load(V_PACK + kb + hoffs, mask=hmask, other=0.0).to(tl.float32)
        s = scale * tl.sum(qv * kv, axis=0)
        m_new = tl.maximum(m, s)
        alpha = tl.exp(m - m_new)
        p = tl.exp(s - m_new)
        lse = lse * alpha + p
        acc = acc * alpha + p * vv
        yv = acc / lse
        if use_xsa != 0:
            vb = tl.load(
                VXSA + (row * N_KV + kv_h) * HD + hoffs, mask=hmask, other=0.0
            ).to(tl.float32)
            vnorm = tl.maximum(tl.sqrt(tl.sum(vb * vb, axis=0)), NEPS)
            vn = vb / vnorm
            yv = yv - tl.sum(yv * vn, axis=0) * vn
        if use_gate != 0:
            go = tl.arange(0, BLOCK_G)
            gmask = go < GATE_N
            xv = tl.load(ATTN_IN + row * D + go, mask=gmask, other=0.0).to(tl.float32)
            gw = tl.load(
                GATE_W + (layer * N_HEADS + head) * GATE_N + go,
                mask=gmask,
                other=0.0,
            ).to(tl.float32)
            yv = yv * (1.0 / (1.0 + tl.exp(-tl.sum(xv * gw, axis=0))))
        tl.store(
            Y + (row * N_HEADS + head) * HD + hoffs,
            yv.to(Y.dtype.element_ty),
            mask=hmask,
        )

    @triton.jit
    def _enc_attn_dot(
        Q,
        K_PACK,
        V_PACK,
        Y,
        VXSA,
        ATTN_IN,
        GATE_W,
        POS,
        slot,
        win,
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
        row_base,
        BLOCK_H: tl.constexpr,
        BLOCK_L: tl.constexpr,
        BLOCK_G: tl.constexpr,
    ):
        head = tl.program_id(0)
        row = tl.program_id(1) + tl.load(row_base).to(tl.int32)
        if (head >= N_HEADS) | (row >= win):
            return
        pos0 = tl.load(POS).to(tl.int32)
        pos = pos0 + row
        tail = pos
        kv_h = head // (N_HEADS // N_KV)
        hoffs = tl.arange(0, BLOCK_H)
        hmask = hoffs < HD
        qv = tl.load(
            Q + (row * N_HEADS + head) * HD + hoffs, mask=hmask, other=0.0
        ).to(tl.float32)
        scale = 1.0 / tl.sqrt(tl.cast(HD, tl.float32))
        dt_m = -1.0e9
        dt_lse = 0.0
        dt_acc = tl.zeros([BLOCK_H], dtype=tl.float32)
        col16 = tl.arange(0, 16)
        q_tc = tl.zeros([BLOCK_H, 16], dtype=tl.bfloat16)
        q_tc = tl.where(
            (col16[None, :] == 0) & hmask[:, None],
            qv.to(tl.bfloat16)[:, None],
            q_tc,
        )
        t0 = 0
        while t0 < pos:
            toffs = t0 + tl.arange(0, BLOCK_L)
            tmask = toffs < pos
            t_b = toffs[:, None]
            d_b = hoffs[None, :]
            kptr = ((slot * N_KV + kv_h) * MAX_LEN + t_b) * HD + d_b
            vmask = tmask[:, None] & (d_b < HD)
            kv = tl.load(K_PACK + kptr, mask=vmask, other=0.0)
            vv = tl.load(V_PACK + kptr, mask=vmask, other=0.0)
            s16 = tl.dot(kv.to(tl.bfloat16), q_tc)
            dt_s = scale * tl.sum(
                s16.to(tl.float32) * (col16[None, :] == 0).to(tl.float32),
                axis=1,
            )
            dt_s = tl.where(tmask, dt_s, -1.0e9)
            dt_mt = tl.max(dt_s, axis=0)
            dt_mn = tl.maximum(dt_m, dt_mt)
            dt_al = tl.exp(dt_m - dt_mn)
            dt_p = tl.exp(dt_s - dt_mn)
            dt_p = tl.where(tmask, dt_p, 0.0)
            dt_lse = dt_lse * dt_al + tl.sum(dt_p, axis=0)
            p_tc = tl.zeros([16, BLOCK_L], dtype=tl.bfloat16)
            p_tc = tl.where(
                col16[:, None] == 0,
                dt_p.to(tl.bfloat16)[None, :],
                p_tc,
            )
            acc16 = tl.dot(p_tc, vv.to(tl.bfloat16))
            dt_pv = tl.sum(
                acc16.to(tl.float32) * (col16[:, None] == 0).to(tl.float32),
                axis=0,
            )
            dt_acc = dt_acc * dt_al + dt_pv
            dt_m = dt_mn
            t0 += BLOCK_L
        kb = ((slot * N_KV + kv_h) * MAX_LEN + tail) * HD
        kv = tl.load(K_PACK + kb + hoffs, mask=hmask, other=0.0).to(tl.float32)
        vv = tl.load(V_PACK + kb + hoffs, mask=hmask, other=0.0).to(tl.float32)
        dt_s = scale * tl.sum(qv * kv, axis=0)
        dt_mn = tl.maximum(dt_m, dt_s)
        dt_al = tl.exp(dt_m - dt_mn)
        dt_p = tl.exp(dt_s - dt_mn)
        dt_lse = dt_lse * dt_al + dt_p
        dt_acc = dt_acc * dt_al + dt_p * vv
        yv = dt_acc / dt_lse
        if use_xsa != 0:
            vb = tl.load(
                VXSA + (row * N_KV + kv_h) * HD + hoffs, mask=hmask, other=0.0
            ).to(tl.float32)
            vnorm = tl.maximum(tl.sqrt(tl.sum(vb * vb, axis=0)), NEPS)
            vn = vb / vnorm
            yv = yv - tl.sum(yv * vn, axis=0) * vn
        if use_gate != 0:
            go = tl.arange(0, BLOCK_G)
            gmask = go < GATE_N
            xv = tl.load(ATTN_IN + row * D + go, mask=gmask, other=0.0).to(tl.float32)
            gw = tl.load(
                GATE_W + (layer * N_HEADS + head) * GATE_N + go,
                mask=gmask,
                other=0.0,
            ).to(tl.float32)
            yv = yv * (1.0 / (1.0 + tl.exp(-tl.sum(xv * gw, axis=0))))
        tl.store(
            Y + (row * N_HEADS + head) * HD + hoffs,
            yv.to(Y.dtype.element_ty),
            mask=hmask,
        )

    @triton.jit
    def _enc_attn_dot_g4(
        Q,
        K_PACK,
        V_PACK,
        Y,
        VXSA,
        ATTN_IN,
        GATE_W,
        POS,
        slot,
        win,
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
        BLOCK_H: tl.constexpr,
        BLOCK_L: tl.constexpr,
        BLOCK_G: tl.constexpr,
    ):
        """4 rows / CTA. Shared BLOCK_L walk; each row has its own K/V load + padded-16 tl.dot.

        Iteration count per row matches ``_enc_attn_dot`` (phased whiles, no extra
        masked tiles). Do not pack Qs into one MMA.
        """
        head = tl.program_id(0)
        row0 = tl.program_id(1) * 4
        if (head >= N_HEADS) | ((row0 + 3) >= win):
            return
        pos_base = tl.load(POS).to(tl.int32)
        kv_h = head // (N_HEADS // N_KV)
        hoffs = tl.arange(0, BLOCK_H)
        hmask = hoffs < HD
        scale = 1.0 / tl.sqrt(tl.cast(HD, tl.float32))
        col16 = tl.arange(0, 16)
        qv_a = tl.load(
            Q + ((row0) * N_HEADS + head) * HD + hoffs, mask=hmask, other=0.0
        ).to(tl.float32)
        q_tc_a = tl.zeros([BLOCK_H, 16], dtype=tl.bfloat16)
        q_tc_a = tl.where(
            (col16[None, :] == 0) & hmask[:, None],
            qv_a.to(tl.bfloat16)[:, None],
            q_tc_a,
        )
        pos_a = pos_base + row0
        dt_m_a = -1.0e9
        dt_lse_a = 0.0
        dt_acc_a = tl.zeros([BLOCK_H], dtype=tl.float32)
        qv_b = tl.load(
            Q + ((row0 + 1) * N_HEADS + head) * HD + hoffs, mask=hmask, other=0.0
        ).to(tl.float32)
        q_tc_b = tl.zeros([BLOCK_H, 16], dtype=tl.bfloat16)
        q_tc_b = tl.where(
            (col16[None, :] == 0) & hmask[:, None],
            qv_b.to(tl.bfloat16)[:, None],
            q_tc_b,
        )
        pos_b = pos_base + row0 + 1
        dt_m_b = -1.0e9
        dt_lse_b = 0.0
        dt_acc_b = tl.zeros([BLOCK_H], dtype=tl.float32)
        qv_c = tl.load(
            Q + ((row0 + 2) * N_HEADS + head) * HD + hoffs, mask=hmask, other=0.0
        ).to(tl.float32)
        q_tc_c = tl.zeros([BLOCK_H, 16], dtype=tl.bfloat16)
        q_tc_c = tl.where(
            (col16[None, :] == 0) & hmask[:, None],
            qv_c.to(tl.bfloat16)[:, None],
            q_tc_c,
        )
        pos_c = pos_base + row0 + 2
        dt_m_c = -1.0e9
        dt_lse_c = 0.0
        dt_acc_c = tl.zeros([BLOCK_H], dtype=tl.float32)
        qv_d = tl.load(
            Q + ((row0 + 3) * N_HEADS + head) * HD + hoffs, mask=hmask, other=0.0
        ).to(tl.float32)
        q_tc_d = tl.zeros([BLOCK_H, 16], dtype=tl.bfloat16)
        q_tc_d = tl.where(
            (col16[None, :] == 0) & hmask[:, None],
            qv_d.to(tl.bfloat16)[:, None],
            q_tc_d,
        )
        pos_d = pos_base + row0 + 3
        dt_m_d = -1.0e9
        dt_lse_d = 0.0
        dt_acc_d = tl.zeros([BLOCK_H], dtype=tl.float32)
        t0 = 0
        while t0 < pos_a:
            toffs0 = t0 + tl.arange(0, BLOCK_L)
            t_b0 = toffs0[:, None]
            d_b0 = hoffs[None, :]
            kptr0 = ((slot * N_KV + kv_h) * MAX_LEN + t_b0) * HD + d_b0
            tmask_a0 = toffs0 < pos_a
            vmask_a0 = tmask_a0[:, None] & (d_b0 < HD)
            kv_a0 = tl.load(K_PACK + kptr0, mask=vmask_a0, other=0.0)
            vv_a0 = tl.load(V_PACK + kptr0, mask=vmask_a0, other=0.0)
            s16_a0 = tl.dot(kv_a0.to(tl.bfloat16), q_tc_a)
            dt_s_a0 = scale * tl.sum(
                s16_a0.to(tl.float32) * (col16[None, :] == 0).to(tl.float32),
                axis=1,
            )
            dt_s_a0 = tl.where(tmask_a0, dt_s_a0, -1.0e9)
            dt_mt_a0 = tl.max(dt_s_a0, axis=0)
            dt_mn_a0 = tl.maximum(dt_m_a, dt_mt_a0)
            dt_al_a0 = tl.exp(dt_m_a - dt_mn_a0)
            dt_p_a0 = tl.exp(dt_s_a0 - dt_mn_a0)
            dt_p_a0 = tl.where(tmask_a0, dt_p_a0, 0.0)
            dt_lse_a = dt_lse_a * dt_al_a0 + tl.sum(dt_p_a0, axis=0)
            p_tc_a0 = tl.zeros([16, BLOCK_L], dtype=tl.bfloat16)
            p_tc_a0 = tl.where(
                col16[:, None] == 0,
                dt_p_a0.to(tl.bfloat16)[None, :],
                p_tc_a0,
            )
            acc16_a0 = tl.dot(p_tc_a0, vv_a0.to(tl.bfloat16))
            dt_pv_a0 = tl.sum(
                acc16_a0.to(tl.float32) * (col16[:, None] == 0).to(tl.float32),
                axis=0,
            )
            dt_acc_a = dt_acc_a * dt_al_a0 + dt_pv_a0
            dt_m_a = dt_mn_a0
            tmask_b0 = toffs0 < pos_b
            vmask_b0 = tmask_b0[:, None] & (d_b0 < HD)
            kv_b0 = tl.load(K_PACK + kptr0, mask=vmask_b0, other=0.0)
            vv_b0 = tl.load(V_PACK + kptr0, mask=vmask_b0, other=0.0)
            s16_b0 = tl.dot(kv_b0.to(tl.bfloat16), q_tc_b)
            dt_s_b0 = scale * tl.sum(
                s16_b0.to(tl.float32) * (col16[None, :] == 0).to(tl.float32),
                axis=1,
            )
            dt_s_b0 = tl.where(tmask_b0, dt_s_b0, -1.0e9)
            dt_mt_b0 = tl.max(dt_s_b0, axis=0)
            dt_mn_b0 = tl.maximum(dt_m_b, dt_mt_b0)
            dt_al_b0 = tl.exp(dt_m_b - dt_mn_b0)
            dt_p_b0 = tl.exp(dt_s_b0 - dt_mn_b0)
            dt_p_b0 = tl.where(tmask_b0, dt_p_b0, 0.0)
            dt_lse_b = dt_lse_b * dt_al_b0 + tl.sum(dt_p_b0, axis=0)
            p_tc_b0 = tl.zeros([16, BLOCK_L], dtype=tl.bfloat16)
            p_tc_b0 = tl.where(
                col16[:, None] == 0,
                dt_p_b0.to(tl.bfloat16)[None, :],
                p_tc_b0,
            )
            acc16_b0 = tl.dot(p_tc_b0, vv_b0.to(tl.bfloat16))
            dt_pv_b0 = tl.sum(
                acc16_b0.to(tl.float32) * (col16[:, None] == 0).to(tl.float32),
                axis=0,
            )
            dt_acc_b = dt_acc_b * dt_al_b0 + dt_pv_b0
            dt_m_b = dt_mn_b0
            tmask_c0 = toffs0 < pos_c
            vmask_c0 = tmask_c0[:, None] & (d_b0 < HD)
            kv_c0 = tl.load(K_PACK + kptr0, mask=vmask_c0, other=0.0)
            vv_c0 = tl.load(V_PACK + kptr0, mask=vmask_c0, other=0.0)
            s16_c0 = tl.dot(kv_c0.to(tl.bfloat16), q_tc_c)
            dt_s_c0 = scale * tl.sum(
                s16_c0.to(tl.float32) * (col16[None, :] == 0).to(tl.float32),
                axis=1,
            )
            dt_s_c0 = tl.where(tmask_c0, dt_s_c0, -1.0e9)
            dt_mt_c0 = tl.max(dt_s_c0, axis=0)
            dt_mn_c0 = tl.maximum(dt_m_c, dt_mt_c0)
            dt_al_c0 = tl.exp(dt_m_c - dt_mn_c0)
            dt_p_c0 = tl.exp(dt_s_c0 - dt_mn_c0)
            dt_p_c0 = tl.where(tmask_c0, dt_p_c0, 0.0)
            dt_lse_c = dt_lse_c * dt_al_c0 + tl.sum(dt_p_c0, axis=0)
            p_tc_c0 = tl.zeros([16, BLOCK_L], dtype=tl.bfloat16)
            p_tc_c0 = tl.where(
                col16[:, None] == 0,
                dt_p_c0.to(tl.bfloat16)[None, :],
                p_tc_c0,
            )
            acc16_c0 = tl.dot(p_tc_c0, vv_c0.to(tl.bfloat16))
            dt_pv_c0 = tl.sum(
                acc16_c0.to(tl.float32) * (col16[:, None] == 0).to(tl.float32),
                axis=0,
            )
            dt_acc_c = dt_acc_c * dt_al_c0 + dt_pv_c0
            dt_m_c = dt_mn_c0
            tmask_d0 = toffs0 < pos_d
            vmask_d0 = tmask_d0[:, None] & (d_b0 < HD)
            kv_d0 = tl.load(K_PACK + kptr0, mask=vmask_d0, other=0.0)
            vv_d0 = tl.load(V_PACK + kptr0, mask=vmask_d0, other=0.0)
            s16_d0 = tl.dot(kv_d0.to(tl.bfloat16), q_tc_d)
            dt_s_d0 = scale * tl.sum(
                s16_d0.to(tl.float32) * (col16[None, :] == 0).to(tl.float32),
                axis=1,
            )
            dt_s_d0 = tl.where(tmask_d0, dt_s_d0, -1.0e9)
            dt_mt_d0 = tl.max(dt_s_d0, axis=0)
            dt_mn_d0 = tl.maximum(dt_m_d, dt_mt_d0)
            dt_al_d0 = tl.exp(dt_m_d - dt_mn_d0)
            dt_p_d0 = tl.exp(dt_s_d0 - dt_mn_d0)
            dt_p_d0 = tl.where(tmask_d0, dt_p_d0, 0.0)
            dt_lse_d = dt_lse_d * dt_al_d0 + tl.sum(dt_p_d0, axis=0)
            p_tc_d0 = tl.zeros([16, BLOCK_L], dtype=tl.bfloat16)
            p_tc_d0 = tl.where(
                col16[:, None] == 0,
                dt_p_d0.to(tl.bfloat16)[None, :],
                p_tc_d0,
            )
            acc16_d0 = tl.dot(p_tc_d0, vv_d0.to(tl.bfloat16))
            dt_pv_d0 = tl.sum(
                acc16_d0.to(tl.float32) * (col16[:, None] == 0).to(tl.float32),
                axis=0,
            )
            dt_acc_d = dt_acc_d * dt_al_d0 + dt_pv_d0
            dt_m_d = dt_mn_d0
            t0 += BLOCK_L
        while t0 < pos_b:
            toffs1 = t0 + tl.arange(0, BLOCK_L)
            t_b1 = toffs1[:, None]
            d_b1 = hoffs[None, :]
            kptr1 = ((slot * N_KV + kv_h) * MAX_LEN + t_b1) * HD + d_b1
            tmask_b1 = toffs1 < pos_b
            vmask_b1 = tmask_b1[:, None] & (d_b1 < HD)
            kv_b1 = tl.load(K_PACK + kptr1, mask=vmask_b1, other=0.0)
            vv_b1 = tl.load(V_PACK + kptr1, mask=vmask_b1, other=0.0)
            s16_b1 = tl.dot(kv_b1.to(tl.bfloat16), q_tc_b)
            dt_s_b1 = scale * tl.sum(
                s16_b1.to(tl.float32) * (col16[None, :] == 0).to(tl.float32),
                axis=1,
            )
            dt_s_b1 = tl.where(tmask_b1, dt_s_b1, -1.0e9)
            dt_mt_b1 = tl.max(dt_s_b1, axis=0)
            dt_mn_b1 = tl.maximum(dt_m_b, dt_mt_b1)
            dt_al_b1 = tl.exp(dt_m_b - dt_mn_b1)
            dt_p_b1 = tl.exp(dt_s_b1 - dt_mn_b1)
            dt_p_b1 = tl.where(tmask_b1, dt_p_b1, 0.0)
            dt_lse_b = dt_lse_b * dt_al_b1 + tl.sum(dt_p_b1, axis=0)
            p_tc_b1 = tl.zeros([16, BLOCK_L], dtype=tl.bfloat16)
            p_tc_b1 = tl.where(
                col16[:, None] == 0,
                dt_p_b1.to(tl.bfloat16)[None, :],
                p_tc_b1,
            )
            acc16_b1 = tl.dot(p_tc_b1, vv_b1.to(tl.bfloat16))
            dt_pv_b1 = tl.sum(
                acc16_b1.to(tl.float32) * (col16[:, None] == 0).to(tl.float32),
                axis=0,
            )
            dt_acc_b = dt_acc_b * dt_al_b1 + dt_pv_b1
            dt_m_b = dt_mn_b1
            tmask_c1 = toffs1 < pos_c
            vmask_c1 = tmask_c1[:, None] & (d_b1 < HD)
            kv_c1 = tl.load(K_PACK + kptr1, mask=vmask_c1, other=0.0)
            vv_c1 = tl.load(V_PACK + kptr1, mask=vmask_c1, other=0.0)
            s16_c1 = tl.dot(kv_c1.to(tl.bfloat16), q_tc_c)
            dt_s_c1 = scale * tl.sum(
                s16_c1.to(tl.float32) * (col16[None, :] == 0).to(tl.float32),
                axis=1,
            )
            dt_s_c1 = tl.where(tmask_c1, dt_s_c1, -1.0e9)
            dt_mt_c1 = tl.max(dt_s_c1, axis=0)
            dt_mn_c1 = tl.maximum(dt_m_c, dt_mt_c1)
            dt_al_c1 = tl.exp(dt_m_c - dt_mn_c1)
            dt_p_c1 = tl.exp(dt_s_c1 - dt_mn_c1)
            dt_p_c1 = tl.where(tmask_c1, dt_p_c1, 0.0)
            dt_lse_c = dt_lse_c * dt_al_c1 + tl.sum(dt_p_c1, axis=0)
            p_tc_c1 = tl.zeros([16, BLOCK_L], dtype=tl.bfloat16)
            p_tc_c1 = tl.where(
                col16[:, None] == 0,
                dt_p_c1.to(tl.bfloat16)[None, :],
                p_tc_c1,
            )
            acc16_c1 = tl.dot(p_tc_c1, vv_c1.to(tl.bfloat16))
            dt_pv_c1 = tl.sum(
                acc16_c1.to(tl.float32) * (col16[:, None] == 0).to(tl.float32),
                axis=0,
            )
            dt_acc_c = dt_acc_c * dt_al_c1 + dt_pv_c1
            dt_m_c = dt_mn_c1
            tmask_d1 = toffs1 < pos_d
            vmask_d1 = tmask_d1[:, None] & (d_b1 < HD)
            kv_d1 = tl.load(K_PACK + kptr1, mask=vmask_d1, other=0.0)
            vv_d1 = tl.load(V_PACK + kptr1, mask=vmask_d1, other=0.0)
            s16_d1 = tl.dot(kv_d1.to(tl.bfloat16), q_tc_d)
            dt_s_d1 = scale * tl.sum(
                s16_d1.to(tl.float32) * (col16[None, :] == 0).to(tl.float32),
                axis=1,
            )
            dt_s_d1 = tl.where(tmask_d1, dt_s_d1, -1.0e9)
            dt_mt_d1 = tl.max(dt_s_d1, axis=0)
            dt_mn_d1 = tl.maximum(dt_m_d, dt_mt_d1)
            dt_al_d1 = tl.exp(dt_m_d - dt_mn_d1)
            dt_p_d1 = tl.exp(dt_s_d1 - dt_mn_d1)
            dt_p_d1 = tl.where(tmask_d1, dt_p_d1, 0.0)
            dt_lse_d = dt_lse_d * dt_al_d1 + tl.sum(dt_p_d1, axis=0)
            p_tc_d1 = tl.zeros([16, BLOCK_L], dtype=tl.bfloat16)
            p_tc_d1 = tl.where(
                col16[:, None] == 0,
                dt_p_d1.to(tl.bfloat16)[None, :],
                p_tc_d1,
            )
            acc16_d1 = tl.dot(p_tc_d1, vv_d1.to(tl.bfloat16))
            dt_pv_d1 = tl.sum(
                acc16_d1.to(tl.float32) * (col16[:, None] == 0).to(tl.float32),
                axis=0,
            )
            dt_acc_d = dt_acc_d * dt_al_d1 + dt_pv_d1
            dt_m_d = dt_mn_d1
            t0 += BLOCK_L
        while t0 < pos_c:
            toffs2 = t0 + tl.arange(0, BLOCK_L)
            t_b2 = toffs2[:, None]
            d_b2 = hoffs[None, :]
            kptr2 = ((slot * N_KV + kv_h) * MAX_LEN + t_b2) * HD + d_b2
            tmask_c2 = toffs2 < pos_c
            vmask_c2 = tmask_c2[:, None] & (d_b2 < HD)
            kv_c2 = tl.load(K_PACK + kptr2, mask=vmask_c2, other=0.0)
            vv_c2 = tl.load(V_PACK + kptr2, mask=vmask_c2, other=0.0)
            s16_c2 = tl.dot(kv_c2.to(tl.bfloat16), q_tc_c)
            dt_s_c2 = scale * tl.sum(
                s16_c2.to(tl.float32) * (col16[None, :] == 0).to(tl.float32),
                axis=1,
            )
            dt_s_c2 = tl.where(tmask_c2, dt_s_c2, -1.0e9)
            dt_mt_c2 = tl.max(dt_s_c2, axis=0)
            dt_mn_c2 = tl.maximum(dt_m_c, dt_mt_c2)
            dt_al_c2 = tl.exp(dt_m_c - dt_mn_c2)
            dt_p_c2 = tl.exp(dt_s_c2 - dt_mn_c2)
            dt_p_c2 = tl.where(tmask_c2, dt_p_c2, 0.0)
            dt_lse_c = dt_lse_c * dt_al_c2 + tl.sum(dt_p_c2, axis=0)
            p_tc_c2 = tl.zeros([16, BLOCK_L], dtype=tl.bfloat16)
            p_tc_c2 = tl.where(
                col16[:, None] == 0,
                dt_p_c2.to(tl.bfloat16)[None, :],
                p_tc_c2,
            )
            acc16_c2 = tl.dot(p_tc_c2, vv_c2.to(tl.bfloat16))
            dt_pv_c2 = tl.sum(
                acc16_c2.to(tl.float32) * (col16[:, None] == 0).to(tl.float32),
                axis=0,
            )
            dt_acc_c = dt_acc_c * dt_al_c2 + dt_pv_c2
            dt_m_c = dt_mn_c2
            tmask_d2 = toffs2 < pos_d
            vmask_d2 = tmask_d2[:, None] & (d_b2 < HD)
            kv_d2 = tl.load(K_PACK + kptr2, mask=vmask_d2, other=0.0)
            vv_d2 = tl.load(V_PACK + kptr2, mask=vmask_d2, other=0.0)
            s16_d2 = tl.dot(kv_d2.to(tl.bfloat16), q_tc_d)
            dt_s_d2 = scale * tl.sum(
                s16_d2.to(tl.float32) * (col16[None, :] == 0).to(tl.float32),
                axis=1,
            )
            dt_s_d2 = tl.where(tmask_d2, dt_s_d2, -1.0e9)
            dt_mt_d2 = tl.max(dt_s_d2, axis=0)
            dt_mn_d2 = tl.maximum(dt_m_d, dt_mt_d2)
            dt_al_d2 = tl.exp(dt_m_d - dt_mn_d2)
            dt_p_d2 = tl.exp(dt_s_d2 - dt_mn_d2)
            dt_p_d2 = tl.where(tmask_d2, dt_p_d2, 0.0)
            dt_lse_d = dt_lse_d * dt_al_d2 + tl.sum(dt_p_d2, axis=0)
            p_tc_d2 = tl.zeros([16, BLOCK_L], dtype=tl.bfloat16)
            p_tc_d2 = tl.where(
                col16[:, None] == 0,
                dt_p_d2.to(tl.bfloat16)[None, :],
                p_tc_d2,
            )
            acc16_d2 = tl.dot(p_tc_d2, vv_d2.to(tl.bfloat16))
            dt_pv_d2 = tl.sum(
                acc16_d2.to(tl.float32) * (col16[:, None] == 0).to(tl.float32),
                axis=0,
            )
            dt_acc_d = dt_acc_d * dt_al_d2 + dt_pv_d2
            dt_m_d = dt_mn_d2
            t0 += BLOCK_L
        while t0 < pos_d:
            toffs3 = t0 + tl.arange(0, BLOCK_L)
            t_b3 = toffs3[:, None]
            d_b3 = hoffs[None, :]
            kptr3 = ((slot * N_KV + kv_h) * MAX_LEN + t_b3) * HD + d_b3
            tmask_d3 = toffs3 < pos_d
            vmask_d3 = tmask_d3[:, None] & (d_b3 < HD)
            kv_d3 = tl.load(K_PACK + kptr3, mask=vmask_d3, other=0.0)
            vv_d3 = tl.load(V_PACK + kptr3, mask=vmask_d3, other=0.0)
            s16_d3 = tl.dot(kv_d3.to(tl.bfloat16), q_tc_d)
            dt_s_d3 = scale * tl.sum(
                s16_d3.to(tl.float32) * (col16[None, :] == 0).to(tl.float32),
                axis=1,
            )
            dt_s_d3 = tl.where(tmask_d3, dt_s_d3, -1.0e9)
            dt_mt_d3 = tl.max(dt_s_d3, axis=0)
            dt_mn_d3 = tl.maximum(dt_m_d, dt_mt_d3)
            dt_al_d3 = tl.exp(dt_m_d - dt_mn_d3)
            dt_p_d3 = tl.exp(dt_s_d3 - dt_mn_d3)
            dt_p_d3 = tl.where(tmask_d3, dt_p_d3, 0.0)
            dt_lse_d = dt_lse_d * dt_al_d3 + tl.sum(dt_p_d3, axis=0)
            p_tc_d3 = tl.zeros([16, BLOCK_L], dtype=tl.bfloat16)
            p_tc_d3 = tl.where(
                col16[:, None] == 0,
                dt_p_d3.to(tl.bfloat16)[None, :],
                p_tc_d3,
            )
            acc16_d3 = tl.dot(p_tc_d3, vv_d3.to(tl.bfloat16))
            dt_pv_d3 = tl.sum(
                acc16_d3.to(tl.float32) * (col16[:, None] == 0).to(tl.float32),
                axis=0,
            )
            dt_acc_d = dt_acc_d * dt_al_d3 + dt_pv_d3
            dt_m_d = dt_mn_d3
            t0 += BLOCK_L
        kb_a = ((slot * N_KV + kv_h) * MAX_LEN + pos_a) * HD
        kv_at = tl.load(K_PACK + kb_a + hoffs, mask=hmask, other=0.0).to(tl.float32)
        vv_at = tl.load(V_PACK + kb_a + hoffs, mask=hmask, other=0.0).to(tl.float32)
        dt_s_at = scale * tl.sum(qv_a * kv_at, axis=0)
        dt_mn_at = tl.maximum(dt_m_a, dt_s_at)
        dt_al_at = tl.exp(dt_m_a - dt_mn_at)
        dt_p_at = tl.exp(dt_s_at - dt_mn_at)
        dt_lse_a = dt_lse_a * dt_al_at + dt_p_at
        dt_acc_a = dt_acc_a * dt_al_at + dt_p_at * vv_at
        yv_a = dt_acc_a / dt_lse_a
        if use_xsa != 0:
            vb_a = tl.load(
                VXSA + ((row0) * N_KV + kv_h) * HD + hoffs, mask=hmask, other=0.0
            ).to(tl.float32)
            vnorm_a = tl.maximum(tl.sqrt(tl.sum(vb_a * vb_a, axis=0)), NEPS)
            vn_a = vb_a / vnorm_a
            yv_a = yv_a - tl.sum(yv_a * vn_a, axis=0) * vn_a
        if use_gate != 0:
            go_a = tl.arange(0, BLOCK_G)
            gmask_a = go_a < GATE_N
            xv_a = tl.load(ATTN_IN + (row0) * D + go_a, mask=gmask_a, other=0.0).to(
                tl.float32
            )
            gw_a = tl.load(
                GATE_W + (layer * N_HEADS + head) * GATE_N + go_a,
                mask=gmask_a,
                other=0.0,
            ).to(tl.float32)
            yv_a = yv_a * (1.0 / (1.0 + tl.exp(-tl.sum(xv_a * gw_a, axis=0))))
        tl.store(
            Y + ((row0) * N_HEADS + head) * HD + hoffs,
            yv_a.to(Y.dtype.element_ty),
            mask=hmask,
        )
        kb_b = ((slot * N_KV + kv_h) * MAX_LEN + pos_b) * HD
        kv_bt = tl.load(K_PACK + kb_b + hoffs, mask=hmask, other=0.0).to(tl.float32)
        vv_bt = tl.load(V_PACK + kb_b + hoffs, mask=hmask, other=0.0).to(tl.float32)
        dt_s_bt = scale * tl.sum(qv_b * kv_bt, axis=0)
        dt_mn_bt = tl.maximum(dt_m_b, dt_s_bt)
        dt_al_bt = tl.exp(dt_m_b - dt_mn_bt)
        dt_p_bt = tl.exp(dt_s_bt - dt_mn_bt)
        dt_lse_b = dt_lse_b * dt_al_bt + dt_p_bt
        dt_acc_b = dt_acc_b * dt_al_bt + dt_p_bt * vv_bt
        yv_b = dt_acc_b / dt_lse_b
        if use_xsa != 0:
            vb_b = tl.load(
                VXSA + ((row0 + 1) * N_KV + kv_h) * HD + hoffs, mask=hmask, other=0.0
            ).to(tl.float32)
            vnorm_b = tl.maximum(tl.sqrt(tl.sum(vb_b * vb_b, axis=0)), NEPS)
            vn_b = vb_b / vnorm_b
            yv_b = yv_b - tl.sum(yv_b * vn_b, axis=0) * vn_b
        if use_gate != 0:
            go_b = tl.arange(0, BLOCK_G)
            gmask_b = go_b < GATE_N
            xv_b = tl.load(ATTN_IN + (row0 + 1) * D + go_b, mask=gmask_b, other=0.0).to(
                tl.float32
            )
            gw_b = tl.load(
                GATE_W + (layer * N_HEADS + head) * GATE_N + go_b,
                mask=gmask_b,
                other=0.0,
            ).to(tl.float32)
            yv_b = yv_b * (1.0 / (1.0 + tl.exp(-tl.sum(xv_b * gw_b, axis=0))))
        tl.store(
            Y + ((row0 + 1) * N_HEADS + head) * HD + hoffs,
            yv_b.to(Y.dtype.element_ty),
            mask=hmask,
        )
        kb_c = ((slot * N_KV + kv_h) * MAX_LEN + pos_c) * HD
        kv_ct = tl.load(K_PACK + kb_c + hoffs, mask=hmask, other=0.0).to(tl.float32)
        vv_ct = tl.load(V_PACK + kb_c + hoffs, mask=hmask, other=0.0).to(tl.float32)
        dt_s_ct = scale * tl.sum(qv_c * kv_ct, axis=0)
        dt_mn_ct = tl.maximum(dt_m_c, dt_s_ct)
        dt_al_ct = tl.exp(dt_m_c - dt_mn_ct)
        dt_p_ct = tl.exp(dt_s_ct - dt_mn_ct)
        dt_lse_c = dt_lse_c * dt_al_ct + dt_p_ct
        dt_acc_c = dt_acc_c * dt_al_ct + dt_p_ct * vv_ct
        yv_c = dt_acc_c / dt_lse_c
        if use_xsa != 0:
            vb_c = tl.load(
                VXSA + ((row0 + 2) * N_KV + kv_h) * HD + hoffs, mask=hmask, other=0.0
            ).to(tl.float32)
            vnorm_c = tl.maximum(tl.sqrt(tl.sum(vb_c * vb_c, axis=0)), NEPS)
            vn_c = vb_c / vnorm_c
            yv_c = yv_c - tl.sum(yv_c * vn_c, axis=0) * vn_c
        if use_gate != 0:
            go_c = tl.arange(0, BLOCK_G)
            gmask_c = go_c < GATE_N
            xv_c = tl.load(ATTN_IN + (row0 + 2) * D + go_c, mask=gmask_c, other=0.0).to(
                tl.float32
            )
            gw_c = tl.load(
                GATE_W + (layer * N_HEADS + head) * GATE_N + go_c,
                mask=gmask_c,
                other=0.0,
            ).to(tl.float32)
            yv_c = yv_c * (1.0 / (1.0 + tl.exp(-tl.sum(xv_c * gw_c, axis=0))))
        tl.store(
            Y + ((row0 + 2) * N_HEADS + head) * HD + hoffs,
            yv_c.to(Y.dtype.element_ty),
            mask=hmask,
        )
        kb_d = ((slot * N_KV + kv_h) * MAX_LEN + pos_d) * HD
        kv_dt = tl.load(K_PACK + kb_d + hoffs, mask=hmask, other=0.0).to(tl.float32)
        vv_dt = tl.load(V_PACK + kb_d + hoffs, mask=hmask, other=0.0).to(tl.float32)
        dt_s_dt = scale * tl.sum(qv_d * kv_dt, axis=0)
        dt_mn_dt = tl.maximum(dt_m_d, dt_s_dt)
        dt_al_dt = tl.exp(dt_m_d - dt_mn_dt)
        dt_p_dt = tl.exp(dt_s_dt - dt_mn_dt)
        dt_lse_d = dt_lse_d * dt_al_dt + dt_p_dt
        dt_acc_d = dt_acc_d * dt_al_dt + dt_p_dt * vv_dt
        yv_d = dt_acc_d / dt_lse_d
        if use_xsa != 0:
            vb_d = tl.load(
                VXSA + ((row0 + 3) * N_KV + kv_h) * HD + hoffs, mask=hmask, other=0.0
            ).to(tl.float32)
            vnorm_d = tl.maximum(tl.sqrt(tl.sum(vb_d * vb_d, axis=0)), NEPS)
            vn_d = vb_d / vnorm_d
            yv_d = yv_d - tl.sum(yv_d * vn_d, axis=0) * vn_d
        if use_gate != 0:
            go_d = tl.arange(0, BLOCK_G)
            gmask_d = go_d < GATE_N
            xv_d = tl.load(ATTN_IN + (row0 + 3) * D + go_d, mask=gmask_d, other=0.0).to(
                tl.float32
            )
            gw_d = tl.load(
                GATE_W + (layer * N_HEADS + head) * GATE_N + go_d,
                mask=gmask_d,
                other=0.0,
            ).to(tl.float32)
            yv_d = yv_d * (1.0 / (1.0 + tl.exp(-tl.sum(xv_d * gw_d, axis=0))))
        tl.store(
            Y + ((row0 + 3) * N_HEADS + head) * HD + hoffs,
            yv_d.to(Y.dtype.element_ty),
            mask=hmask,
        )


else:  # pragma: no cover
    _enc_prologue = None
    _enc_qkv = None
    _enc_attn = None
    _enc_attn_dot = None
    _enc_attn_dot_g4 = None
    _enc_gemv_row_tiles = None


@dataclass
class _EncodeWS:
    win: int
    x: torch.Tensor
    x0: torch.Tensor
    xin: torch.Tensor
    attn_in: torch.Tensor
    q: torch.Tensor
    y: torch.Tensor
    hid: torch.Tensor
    scr: torch.Tensor
    logit_buf: torch.Tensor
    kbuf: torch.Tensor
    vbuf: torch.Tensor
    vxsa: torch.Tensor
    lane0: torch.Tensor
    lane1: torch.Tensor
    skips: torch.Tensor
    k_pack: torch.Tensor
    v_pack: torch.Tensor
    mix: torch.Tensor
    a_sc: torch.Tensor
    m_sc: torch.Tensor
    q_gain: torch.Tensor
    gate_w: torch.Tensor
    ln: torch.Tensor
    pr: torch.Tensor
    pp: torch.Tensor
    itin: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor
    smear_w: torch.Tensor
    skip_w: torch.Tensor
    skip_g: torch.Tensor
    out_w: torch.Tensor
    tok_emb: torch.Tensor
    lm_w: torch.Tensor
    n_enc: int
    has_par: bool
    has_smear: bool
    has_skip_gate: bool
    qkv_dim: int
    mlp_dim: int
    n_slots: int
    max_len: int
    d: int
    hd: int
    n_heads: int
    n_kv: int
    vocab: int
    rope_half: int
    gate_n: int
    soft_pos: float
    soft_neg: float
    smear_lam: float
    final_mode: int
    asym_cap: bool
    itin_host: list


def prepare_mega_encode(model: "GPT", state: "StaticState") -> _EncodeWS | None:
    if not can_mega_encode(model, state):
        return None
    device = state.token.device
    dtype = getattr(model, "_ac_compute_dtype", None) or state.prev_raw.dtype
    d = int(model.cfg.model_dim)
    hd = int(model.blocks[0].attn.head_dim)
    n_kv = int(model.blocks[0].attn.num_kv_heads)
    n_heads = int(model.blocks[0].attn.num_heads)
    n_slots = len(state.slots)
    max_len = int(state.max_len)
    win = int(state.win)
    qkv = model._ac_qkv_bank
    qkv_dim = int(qkv.shape[1])
    mlp_dim = int(model._ac_mlp_up.shape[1])
    itin, n_enc, has_par = _itinerary(model)
    if itin.shape[0] != n_slots:
        return None
    packed = _stack_weights(model, device, dtype)
    like = torch.zeros(1, 1, n_kv, hd, device=device, dtype=dtype)
    for sk in state.slots:
        sk.ensure(like)
    k_pack = torch.zeros(n_slots, n_kv, max_len, hd, device=device, dtype=dtype)
    v_pack = torch.zeros_like(k_pack)
    for i, sk in enumerate(state.slots):
        assert sk.k is not None and sk.v is not None
        if sk.k.data_ptr() != k_pack[i].data_ptr():
            k_pack[i].copy_(
                sk.k.reshape(max_len, n_kv, hd).permute(1, 0, 2).contiguous()
            )
            v_pack[i].copy_(
                sk.v.reshape(max_len, n_kv, hd).permute(1, 0, 2).contiguous()
            )
            sk.k = k_pack[i].permute(1, 0, 2).unsqueeze(0)
            sk.v = v_pack[i].permute(1, 0, 2).unsqueeze(0)
            _mark_static_address(sk.k)
            _mark_static_address(sk.v)

    def _z(*shape, dt=dtype):
        t = torch.zeros(*shape, device=device, dtype=dt)
        _mark_static_address(t)
        return t

    tok_emb, lm_w, out_w, soft_pos, soft_neg, smear_lam, final_mode = _launch_meta(
        model
    )
    ws = _EncodeWS(
        win=win,
        x=_z(win, d),
        x0=_z(win, d),
        xin=_z(win, d),
        attn_in=_z(win, d),
        q=_z(win, n_heads * hd),
        y=_z(win, n_heads * hd),
        hid=_z(win, d),
        scr=_z(win, mlp_dim),
        logit_buf=_z(win, int(model.cfg.vocab_size)),
        kbuf=_z(win, n_heads * hd),
        vbuf=_z(win, n_heads * hd),
        vxsa=_z(win, n_kv * hd),
        lane0=_z(win, d),
        lane1=_z(win, d),
        skips=_z(max(n_enc, 1), win, d),
        k_pack=k_pack,
        v_pack=v_pack,
        mix=packed["mix"],
        a_sc=packed["a_sc"],
        m_sc=packed["m_sc"],
        q_gain=packed["q_gain"],
        gate_w=packed["gate_w"],
        ln=packed["ln"],
        pr=packed["pr"],
        pp=packed["pp"],
        itin=itin.to(device=device),
        cos=packed["cos"],
        sin=packed["sin"],
        smear_w=packed["smear_w"],
        skip_w=packed["skip_w"],
        skip_g=packed["skip_g"],
        out_w=out_w,
        tok_emb=tok_emb,
        lm_w=lm_w,
        n_enc=n_enc,
        has_par=has_par,
        has_smear=bool(model.smear_gate_enabled),
        has_skip_gate=model.skip_gates is not None,
        qkv_dim=qkv_dim,
        mlp_dim=mlp_dim,
        n_slots=n_slots,
        max_len=max_len,
        d=d,
        hd=hd,
        n_heads=n_heads,
        n_kv=n_kv,
        vocab=int(model.cfg.vocab_size),
        rope_half=packed["rope_half"],
        gate_n=packed["gate_n"],
        soft_pos=soft_pos,
        soft_neg=soft_neg,
        smear_lam=smear_lam,
        final_mode=final_mode,
        asym_cap=bool(model.asym_logit_enabled),
        itin_host=[tuple(int(v) for v in row) for row in itin.tolist()],
    )
    ws._ffn_row = _FfnRowView()
    row = ws._ffn_row
    row.itin_host = ws.itin_host
    row.ln = ws.ln
    row.out_w = ws.out_w
    row.a_sc = ws.a_sc
    row.m_sc = ws.m_sc
    row.pr = ws.pr
    row.pp = ws.pp
    if (
        not _want_ffn_loop()
        and model._ac_mlp_up is not None
        and model._ac_mlp_down is not None
    ):
        try:
            encode_gemv_rows(ws.y, ws.out_w[0], ws.hid)
            encode_gemv_rows(ws.attn_in, model._ac_mlp_up[0], ws.scr)
            encode_gemv_rows(ws.scr, model._ac_mlp_down[0], ws.hid)
            encode_gemv_rows(ws.hid, ws.lm_w, ws.logit_buf)
        except Exception as exc:
            ws._enc_gemv_rows_failed = True  # type: ignore[attr-defined]
            cause = exc.args[-1] if exc.args else exc
            kind = "tiled" if _ffn_mode() == "rows" else "batched"
            print(
                f"[AC incr] W-encode {kind} GEMV FFN unavailable ({cause!r}); "
                "using per-row cuBLAS GEMV FFN",
                file=sys.stderr,
                flush=True,
            )
        else:
            ws.hid.zero_()
            ws.scr.zero_()
            ws.logit_buf.zero_()
    _mark_static_address(ws.k_pack)
    _mark_static_address(ws.v_pack)
    _mark_static_address(ws.itin)
    ws.attn_row_base = torch.zeros(1, dtype=torch.int32, device=device)
    _mark_static_address(ws.attn_row_base)
    from .split_attn import ensure_split_workspace

    ensure_split_workspace(ws, win)
    return ws


def refresh_encode_weights(model: "GPT") -> None:
    ws: _EncodeWS | None = getattr(model, "_encode_ws", None)
    if ws is None:
        return
    device = ws.x.device
    dtype = ws.x.dtype
    packed = _stack_weights(model, device, dtype)
    ws.mix.copy_(packed["mix"])
    ws.a_sc.copy_(packed["a_sc"])
    ws.m_sc.copy_(packed["m_sc"])
    ws.q_gain.copy_(packed["q_gain"])
    ws.gate_w.copy_(packed["gate_w"])
    ws.ln.copy_(packed["ln"])
    ws.pr.copy_(packed["pr"])
    ws.pp.copy_(packed["pp"])
    ws.cos.copy_(packed["cos"])
    ws.sin.copy_(packed["sin"])
    ws.skip_w.copy_(packed["skip_w"])
    ws.skip_g.copy_(packed["skip_g"])
    ws.smear_w.copy_(packed["smear_w"])
    tok_emb, lm_w, out_w, soft_pos, soft_neg, smear_lam, final_mode = _launch_meta(
        model
    )
    ws.tok_emb = tok_emb
    ws.lm_w = lm_w
    ws.out_w = out_w
    row = getattr(ws, "_ffn_row", None)
    if row is not None:
        row.out_w = out_w
        row.ln = ws.ln
        row.a_sc = ws.a_sc
        row.m_sc = ws.m_sc
        row.pr = ws.pr
        row.pp = ws.pp
    ws.soft_pos = soft_pos
    ws.soft_neg = soft_neg
    ws.smear_lam = smear_lam
    ws.final_mode = final_mode
    ws.asym_cap = bool(model.asym_logit_enabled)


def _rms_rows_pt(x: torch.Tensor) -> torch.Tensor:
    """Per-row RMS. Same formula as ``_rms`` on each 1D row."""
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + _RMS_EPS).to(
        dtype=x.dtype
    )


def _ffn_mode() -> str:
    """``rows`` (default, shared tiled GEMV), ``loop``, or ``bmm``."""
    raw = os.environ.get("XSA_AC_ENCODE_FFN", "rows").strip().lower()
    if raw in {"loop", "persist", "1d", "cublas"}:
        return "loop"
    if raw in {"bmm", "batched"}:
        return "bmm"
    return "rows"


def _want_ffn_loop() -> bool:
    """True → per-row cuBLAS ``persist_ffn_slot`` (decode match)."""
    return _ffn_mode() == "loop"


def _ffn_banner(ws: object | None = None) -> str:
    if ws is not None and (
        getattr(ws, "_enc_gemv_rows_failed", False)
        or getattr(ws, "_enc_gemv_bmm_failed", False)
    ):
        return "cuBLAS GEMV FFN"
    mode = _ffn_mode()
    if mode == "rows":
        from .ac_gemv import want_stage_fusion

        if want_stage_fusion():
            return "3-stage fused FFN"
        return "tiled GEMV FFN"
    if mode == "bmm":
        return "cuBLAS batched-GEMV FFN"
    return "cuBLAS GEMV FFN"


def encode_gemv_bmm(x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor) -> None:
    """``out[r] = x[r] @ weight.T`` via batched M=1 GEMV. Not an M=W GEMM."""
    win = int(x.shape[0])
    k = int(weight.shape[1])
    n_out = int(weight.shape[0])
    if int(x.shape[1]) != k or tuple(out.shape) != (win, n_out):
        raise ValueError(
            f"gemv_bmm shape mismatch x={tuple(x.shape)} w={tuple(weight.shape)} "
            f"out={tuple(out.shape)}"
        )
    # [W,1,K] @ [W,K,N] with B broadcast (stride 0). Same OP_T view as F.linear.
    torch.bmm(
        x.unsqueeze(1),
        weight.transpose(0, 1).unsqueeze(0).expand(win, k, n_out),
        out=out.unsqueeze(1),
    )


def encode_gemv_rows(x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor) -> None:
    """``out[r] = weight @ x[r]`` in one launch. Rows do not mix (not M=W GEMM)."""
    if _ffn_mode() == "bmm":
        encode_gemv_bmm(x, weight, out)
        return
    from .ac_gemv import gemv_rows

    gemv_rows(x, weight, out)


def encode_embed(state: "StaticState", ws: _EncodeWS) -> None:
    """Window smear: sequential ``persist_embed`` on each row (decode match)."""
    raw = F.embedding(state.token.view(-1), ws.tok_emb)
    win = ws.win
    state.cand_prev_raw.view(-1).copy_(raw[win - 1])
    prev = state.prev_raw.view(-1)
    for r in range(win):
        row = raw[r]
        if ws.has_smear:
            g = ws.smear_lam * torch.sigmoid(
                (row[: ws.gate_n].float() * ws.smear_w[: ws.gate_n].float()).sum()
            )
            x = _rms(row + g.to(ws.x.dtype) * prev, _RMS_EPS)
        else:
            x = _rms(row, _RMS_EPS)
        ws.x[r].copy_(x)
        ws.x0[r].copy_(x)
        prev = row


def encode_ffn_slot(model: "GPT", ws: _EncodeWS, slot: int) -> None:
    """Out-proj + MLP for all W rows.

    Default is tiled Triton GEMV (same kernel as decode). ``loop`` is 64×
    ``persist_ffn_slot`` cuBLAS. ``bmm`` is batched M=1 cuBLAS.
    """
    from .ac_gemv import can_gemv_rows

    use_loop = (
        _want_ffn_loop()
        or getattr(ws, "_enc_gemv_rows_failed", False)
        or (_ffn_mode() == "rows" and not can_gemv_rows())
    )
    if not use_loop:
        try:
            _encode_ffn_slot_rows(model, ws, slot)
            return
        except Exception as exc:
            ws._enc_gemv_rows_failed = True  # type: ignore[attr-defined]
            if not getattr(ws, "_enc_gemv_rows_logged", False):
                ws._enc_gemv_rows_logged = True  # type: ignore[attr-defined]
                cause = exc.args[-1] if exc.args else exc
                kind = "tiled" if _ffn_mode() == "rows" else "batched"
                print(
                    f"[AC incr] W-encode {kind} GEMV FFN unavailable ({cause!r}); "
                    "using per-row cuBLAS GEMV FFN",
                    file=sys.stderr,
                    flush=True,
                )
    row = ws._ffn_row
    for r in range(ws.win):
        row.x = ws.x[r]
        row.xin = ws.xin[r]
        row.attn_in = ws.attn_in[r]
        row.y = ws.y[r]
        row.lane0 = ws.lane0[r]
        row.lane1 = ws.lane1[r]
        row.skips = ws.skips[:, r]
        persist_ffn_slot(model, row, slot)  # type: ignore[arg-type]


def _encode_ffn_slot_rows(model: "GPT", ws: _EncodeWS, slot: int) -> None:
    ffn_slot_rows(model, ws, slot)


def encode_logits(state: "StaticState", ws: _EncodeWS) -> None:
    if _want_ffn_loop() or getattr(ws, "_enc_gemv_rows_failed", False):
        if ws.has_par:
            if ws.final_mode == 1:
                src = ws.lane0
            elif ws.final_mode == 2:
                src = ws.lane1
            else:
                src = 0.5 * (ws.lane0 + ws.lane1)
        else:
            src = ws.x
        for r in range(ws.win):
            hid = _rms(src[r], _RMS_EPS)
            logits = F.linear(hid, ws.lm_w)
            if ws.asym_cap:
                logits = torch.where(
                    logits >= 0,
                    ws.soft_pos * torch.tanh(logits / ws.soft_pos),
                    ws.soft_neg * torch.tanh(logits / ws.soft_neg),
                )
            else:
                logits = ws.soft_pos * torch.tanh(logits / ws.soft_pos)
            state.logits[0, r].copy_(logits.float())
        return
    logits_rows(state, ws)


def _next_pow2(n: int) -> int:
    p = 1
    while p < int(n):
        p *= 2
    return max(p, 1)


def _enc_attn_slot(state: "StaticState", ws: _EncodeWS, slot: int) -> None:
    layer, _kind, _ss, _sd, _sw, use_xsa, use_gate = ws.itin_host[slot]
    block_h = _next_pow2(ws.hd)
    block_g = _next_pow2(max(ws.gate_n, 1))
    grid = (ws.n_heads, ws.win)
    common = (
        ws.q,
        ws.k_pack,
        ws.v_pack,
        ws.y,
        ws.vxsa,
        ws.attn_in,
        ws.gate_w,
        state.pos.view(-1),
        slot,
        ws.win,
        ws.max_len,
        ws.hd,
        ws.n_heads,
        ws.n_kv,
        ws.gate_n,
        ws.d,
        layer,
        use_xsa,
        use_gate,
        _NORM_EPS,
    )
    want_dot = _env_on("XSA_AC_ATTN_DOT", "1") and not getattr(
        ws, "_enc_attn_dot_failed", False
    )
    group = _enc_attn_group(ws.win)
    row_base = getattr(ws, "attn_row_base", None)
    if row_base is None:
        row_base = torch.zeros(1, dtype=torch.int32, device=ws.q.device)
        _mark_static_address(row_base)
        ws.attn_row_base = row_base
    from .split_attn import run_split_attn

    split_count = int(getattr(ws, "attn_splits", 1))
    if split_count > 1 and not getattr(ws, "_enc_attn_split_failed", False):
        try:
            if run_split_attn(
                ws=ws,
                pos=state.pos.view(-1),
                slot=slot,
                rows=ws.win,
                layer=layer,
                use_xsa=use_xsa,
                use_gate=use_gate,
                v_cur=ws.vxsa,
                v_cur_row_stride=ws.n_kv * ws.hd,
                tail_at_pos=True,
                norm_eps=_NORM_EPS,
                block_l=_attn_block_l(),
            ):
                ws.mega_attn = f"split{split_count}"  # type: ignore[attr-defined]
                return
        except Exception as exc:
            ws._enc_attn_split_failed = True  # type: ignore[attr-defined]
            if not getattr(ws, "_enc_attn_split_logged", False):
                ws._enc_attn_split_logged = True  # type: ignore[attr-defined]
                cause = exc.args[-1] if exc.args else exc
                print(
                    f"[AC incr] split-prefix encode attn unavailable "
                    f"({cause!r}); using serial encode attn",
                    file=sys.stderr,
                    flush=True,
                )
    if want_dot:
        block_l = _attn_block_l()
        if (
            group == 4
            and _enc_attn_dot_g4 is not None
            and not getattr(ws, "_enc_attn_dot_g4_failed", False)
        ):
            try:
                _enc_attn_dot_g4[(ws.n_heads, ws.win // 4)](
                    *common,
                    BLOCK_H=block_h,
                    BLOCK_L=block_l,
                    BLOCK_G=block_g,
                    num_warps=8,
                    num_stages=3,
                )
                ws.mega_attn = "dot-g4"  # type: ignore[attr-defined]
                return
            except Exception as exc:
                ws._enc_attn_dot_g4_failed = True  # type: ignore[attr-defined]
                if not getattr(ws, "_enc_attn_dot_g4_logged", False):
                    ws._enc_attn_dot_g4_logged = True  # type: ignore[attr-defined]
                    cause = exc.args[-1] if exc.args else exc
                    print(
                        f"[AC incr] W-encode 4-row mega attn unavailable "
                        f"({cause!r}); using per-row encode attn",
                        file=sys.stderr,
                        flush=True,
                    )
        try:
            row_base.zero_()
            _enc_attn_dot[grid](
                *common,
                row_base,
                BLOCK_H=block_h,
                BLOCK_L=block_l,
                BLOCK_G=block_g,
                num_warps=8,
                num_stages=3,
            )
            ws.mega_attn = "dot"  # type: ignore[attr-defined]
            return
        except Exception as exc:
            ws._enc_attn_dot_failed = True  # type: ignore[attr-defined]
            if not getattr(ws, "_enc_attn_dot_logged", False):
                ws._enc_attn_dot_logged = True  # type: ignore[attr-defined]
                cause = exc.args[-1] if exc.args else exc
                print(
                    f"[AC incr] W-encode mega attn tl.dot unavailable ({cause!r}); "
                    "using fp32 encode attn",
                    file=sys.stderr,
                    flush=True,
                )
    _enc_attn[grid](
        *common,
        BLOCK_H=block_h,
        BLOCK_L=128,
        BLOCK_G=block_g,
        num_warps=4,
    )
    ws.mega_attn = "fp32"  # type: ignore[attr-defined]


def try_mega_encode(model: "GPT", state: "StaticState") -> bool:
    """W-token encode step. False → fused slots."""
    if not can_mega_encode(model, state) or getattr(model, "_encode_disabled", False):
        return False
    ws: _EncodeWS | None = getattr(model, "_encode_ws", None)
    if (
        ws is None
        or ws.k_pack.shape[0] != len(state.slots)
        or int(ws.win) != int(state.win)
        or ws.k_pack.shape[2] != state.max_len
    ):
        try:
            ws = prepare_mega_encode(model, state)
        except Exception:
            return False
        if ws is None:
            return False
        model._encode_ws = ws
    qkv = model._ac_qkv_bank
    if qkv is None or model._ac_mlp_up is None or model._ac_mlp_down is None:
        return False
    if not _HAS_TRITON:
        return False
    d = ws.d
    hd = ws.hd
    block_d = _next_pow2(d)
    block_h = _next_pow2(hd)
    try:
        encode_embed(state, ws)
        for slot in range(ws.n_slots):
            _enc_prologue[(ws.win,)](
                ws.x,
                ws.x0,
                ws.xin,
                ws.attn_in,
                ws.lane0,
                ws.lane1,
                ws.skips,
                ws.mix,
                ws.ln,
                ws.skip_w,
                ws.skip_g,
                ws.itin,
                slot,
                d,
                ws.win,
                ws.n_enc,
                int(ws.has_skip_gate),
                BLOCK_D=block_d,
                num_warps=4,
            )
            _enc_qkv[(ws.n_heads, ws.win)](
                ws.attn_in,
                ws.q,
                ws.kbuf,
                ws.vbuf,
                ws.k_pack,
                ws.v_pack,
                qkv,
                ws.q_gain,
                ws.cos,
                ws.sin,
                ws.itin,
                state.pos.view(-1),
                ws.vxsa,
                slot,
                d,
                hd,
                ws.n_heads,
                ws.n_kv,
                ws.qkv_dim,
                ws.max_len,
                ws.win,
                ws.rope_half,
                BLOCK_H=block_h,
                BLOCK_K=128,
                DEDUP_KV=1 if _env_on("XSA_AC_GQA_DEDUP", "1") else 0,
                num_warps=4,
            )
            _enc_attn_slot(state, ws, slot)
            encode_ffn_slot(model, ws, slot)
        encode_logits(state, ws)
        ws.mega_mode = "encode"
    except Exception as exc:
        model._encode_disabled = True
        cause = (
            exc.args[-1]
            if type(exc).__name__ == "CompilationError" and exc.args
            else exc
        )
        print(
            f"[AC incr] W-encode megakernel failed ({cause!r}); "
            "using per-slot fused steps",
            file=sys.stderr,
            flush=True,
        )
        return False
    return True
