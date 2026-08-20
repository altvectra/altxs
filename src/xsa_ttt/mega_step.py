"""W=1 megakernel path: QKV/RoPE/pack + split-prefix attention + tiled FFN.

GQA K/V use dedicated per-head scratch (not ``SCR``). Only group leaders
project and write shared K/V (``XSA_AC_GQA_DEDUP=0`` restores redundant
projection for A/B). The default prefix scan is shared with W-encode
and partitions each head across CTAs before a fixed-order fp32 reduction.
``XSA_AC_ATTN_SPLITS=1`` restores the serial thin kernel in this file.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .persistent_step import _PersistWS
    from .model import GPT, StaticState

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _HAS_TRITON = False


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, str(default))))
    except ValueError:
        return default


def _env_on(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
        "",
    }


DUMP_ATTN_IN = 0
DUMP_Y = 1
DUMP_X = 2
DUMP_LANE1 = 3
DUMP_ATTN_OUT = 4
DUMP_NFIELD = 5


def mega_dump_on() -> bool:
    return os.environ.get("XSA_AC_MEGA_DUMP", "0").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
        "",
    }


def attach_layer_dumps(ws: "_PersistWS", device: torch.device | None = None) -> None:
    """Per-slot snapshots for ``scripts/probe_mega_layers.py``. Cheap (~100KB)."""
    if getattr(ws, "dump_d", None) is not None:
        return
    if device is None:
        device = ws.x.device
    n_slots = int(ws.n_slots)
    d = int(ws.d)
    n_heads = int(ws.n_heads)
    hd = int(ws.hd)
    dtype = ws.x.dtype
    ws.dump_d = torch.zeros(n_slots, DUMP_NFIELD, d, device=device, dtype=dtype)
    ws.dump_q = torch.zeros(n_slots, n_heads * hd, device=device, dtype=dtype)
    from .model import _mark_static_address

    _mark_static_address(ws.dump_d)
    _mark_static_address(ws.dump_q)


def mega_cta_count(device: torch.device, n_heads: int = 8) -> int:
    if device.type != "cuda":
        return 1
    n_sm = int(torch.cuda.get_device_properties(device).multi_processor_count)
    # 8 heads × 8 KV tiles. Cutting to 8 CTAs serialized the 16k scan.
    _ = n_heads
    return min(n_sm, _env_int("XSA_AC_MEGA_CTA", 64))


if _HAS_TRITON:

    @triton.jit
    def _ld_vol(ptr):
        return tl.inline_asm_elementwise(
            "ld.volatile.global.s32 $0, [$1];",
            "=r, l",
            [ptr],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )

    @triton.jit
    def _grid_sync(FLAGS, pid, n_cta):
        # Intra-CTA only. Split-launch path joins at the kernel boundary.
        tl.debug_barrier()

    def _grid_sync_ptx_impl(ARRIVED, SENSE, n_cta):
        # Width-1 dummy so the asm is not cloned across BLOCK_D lanes when
        # Triton honors noinline. Do not rewrite as a Triton ``while`` on a
        # volatile flag — that hangs compile.
        _ = tl.arange(0, 1)
        tl.inline_asm_elementwise(
            """{
            .reg .pred p_tid, p_last, p_eq;
            .reg .s32 tid, n, old, cur, last;
            bar.sync 0;
            ld.global.s32 old, [$2];
            mov.u32 tid, %tid.x;
            setp.eq.s32 p_tid, tid, 0;
            mov.s32 n, -1;
            @p_tid atom.global.add.s32 n, [$1], 1;
            add.s32 last, $3, -1;
            setp.eq.s32 p_last, n, last;
            @p_last st.global.s32 [$1], 0;
            @p_last add.s32 cur, old, 1;
            @p_last st.global.s32 [$2], cur;
            membar.gl;
            bar.sync 0;
            WAIT:
            ld.volatile.global.s32 cur, [$2];
            setp.eq.s32 p_eq, cur, old;
            @p_eq bra WAIT;
            bar.sync 0;
            mov.s32 $0, 0;
            }""",
            "=r, l, l, r",
            [ARRIVED, SENSE, n_cta],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )

    try:
        _grid_sync_ptx = triton.jit(_grid_sync_ptx_impl, noinline=True)
    except TypeError:  # pragma: no cover - older Triton
        _grid_sync_ptx = triton.jit(_grid_sync_ptx_impl)

    @triton.jit
    def _tanh(tx):
        return 2.0 / (1.0 + tl.exp(-2.0 * tx)) - 1.0

    @triton.jit
    def _rms_vec(vec, D, eps):
        return vec * tl.rsqrt(tl.sum(vec * vec, axis=0) / D + eps)

    @triton.jit
    def _gemv_blk(
        X,
        W,
        Y,
        n_out,
        k,
        pid,
        n_cta,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        FUSE_KIND: tl.constexpr,
        soft_pos,
        soft_neg,
        asym_cap,
    ):
        blk_row0 = pid * BLOCK_N
        blk_step = n_cta * BLOCK_N
        while blk_row0 < n_out:
            blk_acc = tl.zeros([BLOCK_N], dtype=tl.float32)
            blk_no = blk_row0 + tl.arange(0, BLOCK_N)
            blk_nm = blk_no < n_out
            blk_kk = 0
            while blk_kk < k:
                blk_ko = blk_kk + tl.arange(0, BLOCK_K)
                blk_km = blk_ko < k
                blk_xv = tl.load(X + blk_ko, mask=blk_km, other=0.0).to(tl.float32)
                blk_wv = tl.load(
                    W + blk_no[:, None] * k + blk_ko[None, :],
                    mask=blk_nm[:, None] & blk_km[None, :],
                    other=0.0,
                ).to(tl.float32)
                blk_acc += tl.sum(blk_wv * blk_xv[None, :], axis=1)
                blk_kk += BLOCK_K
            if FUSE_KIND == 1:
                blk_acc = tl.where(blk_acc >= 0, blk_acc, blk_acc * 0.5)
                blk_acc = blk_acc * blk_acc
            if FUSE_KIND == 2:
                if asym_cap != 0:
                    blk_acc = tl.where(
                        blk_acc >= 0,
                        soft_pos * _tanh(blk_acc / soft_pos),
                        soft_neg * _tanh(blk_acc / soft_neg),
                    )
                else:
                    blk_acc = soft_pos * _tanh(blk_acc / soft_pos)
            tl.store(Y + blk_no, blk_acc.to(Y.dtype.element_ty), mask=blk_nm)
            blk_row0 += blk_step

    @triton.jit
    def _gemv_own_blk(X, W, Y, n_out, k, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        own_acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        own_no = tl.arange(0, BLOCK_N)
        own_nm = own_no < n_out
        own_kk = 0
        while own_kk < k:
            own_ko = own_kk + tl.arange(0, BLOCK_K)
            own_km = own_ko < k
            own_xv = tl.load(X + own_ko, mask=own_km, other=0.0).to(tl.float32)
            own_wv = tl.load(
                W + own_no[:, None] * k + own_ko[None, :],
                mask=own_nm[:, None] & own_km[None, :],
                other=0.0,
            ).to(tl.float32)
            own_acc += tl.sum(own_wv * own_xv[None, :], axis=1)
            own_kk += BLOCK_K
        tl.store(Y + own_no, own_acc.to(Y.dtype.element_ty), mask=own_nm)

    @triton.jit
    def _rope_head(VEC, COS, SIN, HD, RH, BLOCK: tl.constexpr):
        rope_offs = tl.arange(0, BLOCK)
        rope_lo = tl.load(VEC + rope_offs, mask=rope_offs < RH, other=0.0).to(
            tl.float32
        )
        rope_hi = tl.load(VEC + RH + rope_offs, mask=rope_offs < RH, other=0.0).to(
            tl.float32
        )
        rope_c = tl.load(COS + rope_offs, mask=rope_offs < RH, other=0.0).to(
            tl.float32
        )
        rope_s = tl.load(SIN + rope_offs, mask=rope_offs < RH, other=0.0).to(
            tl.float32
        )
        tl.store(
            VEC + rope_offs,
            (rope_lo * rope_c + rope_hi * rope_s).to(VEC.dtype.element_ty),
            mask=rope_offs < RH,
        )
        tl.store(
            VEC + RH + rope_offs,
            (-rope_lo * rope_s + rope_hi * rope_c).to(VEC.dtype.element_ty),
            mask=rope_offs < RH,
        )

    @triton.jit
    def _mega_kernel(
        TOKEN,
        PREV,
        CAND,
        POS,
        LOGITS,
        X,
        X0,
        XIN,
        ATTN_IN,
        SCR,
        Q,
        KBUF,
        VBUF,
        Y,
        HID,
        LANE0,
        LANE1,
        SKIPS,
        K_PACK,
        V_PACK,
        MIX,
        A_SC,
        M_SC,
        Q_GAIN,
        GATE_W,
        LN,
        PR,
        PP,
        ITIN,
        COS,
        SIN,
        SMEAR_W,
        SKIP_W,
        SKIP_G,
        QKV,
        UP,
        DOWN,
        OUT_W,
        TOK_EMB,
        LM_W,
        PART_M,
        PART_LSE,
        PART_ACC,
        ARRIVED,
        SENSE,
        BAR_FLAGS,
        DUMP_D,
        DUMP_Q,
        SLOT_SEL,
        n_cta,
        n_slots,
        n_enc,
        d,
        hd,
        n_heads,
        n_kv,
        qkv_dim,
        mlp_dim,
        vocab,
        max_len,
        rh,
        gate_n,
        has_smear,
        has_skip_gate,
        has_par,
        final_mode,
        asym_cap,
        smear_lam,
        soft_pos,
        soft_neg,
        BLOCK_D: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_L: tl.constexpr,
        BLOCK_G: tl.constexpr,
        DUMP: tl.constexpr,
        STAGE: tl.constexpr,
        DEDUP_KV: tl.constexpr,
    ):
        pid = tl.program_id(0)
        tail = max_len - 1
        offs = tl.arange(0, BLOCK_D)
        dmask = offs < d
        if STAGE == 1:
            tok = tl.load(TOKEN).to(tl.int32)
            raw = tl.load(TOK_EMB + tok * d + offs, mask=dmask, other=0.0).to(
                tl.float32
            )
            tl.store(CAND + offs, raw.to(CAND.dtype.element_ty), mask=dmask)
            x = raw
            if has_smear != 0:
                smear_go = tl.arange(0, BLOCK_G)
                smear_mask = smear_go < gate_n
                smear_rv = tl.load(CAND + smear_go, mask=smear_mask, other=0.0).to(
                    tl.float32
                )
                smw = tl.load(SMEAR_W + smear_go, mask=smear_mask, other=0.0).to(
                    tl.float32
                )
                smear_g = smear_lam * (
                    1.0 / (1.0 + tl.exp(-tl.sum(smear_rv * smw, axis=0)))
                )
                prev = tl.load(PREV + offs, mask=dmask, other=0.0).to(tl.float32)
                x = raw + smear_g * prev
            x = _rms_vec(x, d, 1e-6)
            tl.store(X + offs, x.to(X.dtype.element_ty), mask=dmask)
            tl.store(X0 + offs, x.to(X0.dtype.element_ty), mask=dmask)

        if (STAGE == 0) | (STAGE == 2) | (STAGE == 3):
            slot = SLOT_SEL
            n_left = 1
            if STAGE == 0:
                slot = 0
                n_left = n_slots
            while n_left > 0:
                layer = tl.load(ITIN + slot * 7 + 0)
                kind = tl.load(ITIN + slot * 7 + 1)
                skip_src = tl.load(ITIN + slot * 7 + 2)
                skip_dst = tl.load(ITIN + slot * 7 + 3)
                skip_wi = tl.load(ITIN + slot * 7 + 4)
                use_xsa = tl.load(ITIN + slot * 7 + 5)
                use_gate = tl.load(ITIN + slot * 7 + 6)
                par = (kind == 2) | (kind == 3)
                if (STAGE == 0) | (STAGE == 2):
                    x = tl.load(X + offs, mask=dmask, other=0.0).to(tl.float32)
                    if kind == 2:
                        tl.store(LANE0 + offs, x.to(LANE0.dtype.element_ty), mask=dmask)
                        tl.store(LANE1 + offs, x.to(LANE1.dtype.element_ty), mask=dmask)
                    if par:
                        x = tl.load(LANE0 + offs, mask=dmask, other=0.0).to(tl.float32)
                    if skip_src >= 0:
                        sk = tl.load(SKIPS + skip_src * d + offs, mask=dmask, other=0.0).to(
                            tl.float32
                        )
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
                    x0 = tl.load(X0 + offs, mask=dmask, other=0.0).to(tl.float32)
                    m0 = tl.load(MIX + (layer * 2 + 0) * d + offs, mask=dmask, other=0.0).to(
                        tl.float32
                    )
                    m1 = tl.load(MIX + (layer * 2 + 1) * d + offs, mask=dmask, other=0.0).to(
                        tl.float32
                    )
                    xin = m0 * x + m1 * x0
                    ln = tl.load(LN + layer)
                    attn = _rms_vec(xin, d, 1e-6) * ln
                    tl.store(XIN + offs, xin.to(XIN.dtype.element_ty), mask=dmask)
                    tl.store(ATTN_IN + offs, attn.to(ATTN_IN.dtype.element_ty), mask=dmask)
                    if par:
                        tl.store(LANE0 + offs, x.to(LANE0.dtype.element_ty), mask=dmask)
                    else:
                        tl.store(X + offs, x.to(X.dtype.element_ty), mask=dmask)
                    if DUMP:
                        tl.store(
                            DUMP_D + (slot * 5 + 0) * d + offs,
                            attn.to(DUMP_D.dtype.element_ty),
                            mask=dmask,
                        )

                    # One persist ``_attn_heads`` scan per head. ``head`` walks by
                    # ``n_cta`` so ``XSA_AC_MEGA_CTA=1`` still covers all heads.
                    # KBUF/VBUF here are per-head mega scratch (not the n_kv persist
                    # bufs). Do not reuse SCR — MLP up-proj writes it.
                    head = pid
                    while head < n_heads:
                        kv_h = head // (n_heads // n_kv)
                        q_w = QKV + (layer * qkv_dim + head * hd) * d
                        q_ptr = Q + head * hd
                        _gemv_own_blk(ATTN_IN, q_w, q_ptr, hd, d, BLOCK_H, BLOCK_K)
                        ho = tl.arange(0, BLOCK_H)
                        hmask = ho < hd
                        qraw = tl.load(q_ptr + ho, mask=hmask, other=0.0).to(tl.float32)
                        qn = qraw * tl.rsqrt(tl.sum(qraw * qraw, axis=0) / hd + 1e-6)
                        qn = qn * tl.load(Q_GAIN + layer * n_heads + head)
                        tl.store(q_ptr + ho, qn.to(Q.dtype.element_ty), mask=hmask)
                        pos = tl.load(POS).to(tl.int32)
                        _rope_head(q_ptr, COS + pos * rh, SIN + pos * rh, hd, rh, BLOCK_H)
                        qv = tl.load(q_ptr + ho, mask=hmask, other=0.0).to(tl.float32)
                        group_leader = (head % (n_heads // n_kv)) == 0
                        # Split-launch decode has a kernel boundary before
                        # attention, so only the GQA leader needs to project
                        # shared K/V. STAGE=0 keeps the historical per-head
                        # work because its inline attention has no join here.
                        ksc = KBUF + head * hd
                        vsc = VBUF + head * hd
                        if (DEDUP_KV == 0) | (STAGE == 0) | group_leader:
                            k_w = QKV + (layer * qkv_dim + d + kv_h * hd) * d
                            v_w = QKV + (
                                layer * qkv_dim + d + n_kv * hd + kv_h * hd
                            ) * d
                            _gemv_own_blk(
                                ATTN_IN, k_w, ksc, hd, d, BLOCK_H, BLOCK_K
                            )
                            _gemv_own_blk(
                                ATTN_IN, v_w, vsc, hd, d, BLOCK_H, BLOCK_K
                            )
                            kraw = tl.load(
                                ksc + ho, mask=hmask, other=0.0
                            ).to(tl.float32)
                            kn = kraw * tl.rsqrt(
                                tl.sum(kraw * kraw, axis=0) / hd + 1e-6
                            )
                            tl.store(
                                ksc + ho,
                                kn.to(KBUF.dtype.element_ty),
                                mask=hmask,
                            )
                            _rope_head(
                                ksc,
                                COS + pos * rh,
                                SIN + pos * rh,
                                hd,
                                rh,
                                BLOCK_H,
                            )
                        if group_leader:
                            tl.store(
                                K_PACK + ((slot * n_kv + kv_h) * max_len + tail) * hd + ho,
                                tl.load(ksc + ho, mask=hmask, other=0.0),
                                mask=hmask,
                            )
                            tl.store(
                                V_PACK + ((slot * n_kv + kv_h) * max_len + tail) * hd + ho,
                                tl.load(vsc + ho, mask=hmask, other=0.0),
                                mask=hmask,
                            )
                        # Prefix+current softmax lives in persist ``_attn_heads``.
                        # Inlining it here matched pos=0/1 and blew up at pos>=2
                        # (first RoPE'd prefix key) — fat-kernel miscompile.
                        if STAGE == 0:
                            scale = 1.0 / tl.sqrt(tl.cast(hd, tl.float32))
                            at_m = -1.0e9
                            at_lse = 0.0
                            at_acc = tl.zeros([BLOCK_H], dtype=tl.float32)
                            at_t0 = 0
                            while at_t0 < pos:
                                at_toffs = at_t0 + tl.arange(0, BLOCK_L)
                                at_tmask = at_toffs < pos
                                at_tb = at_toffs[:, None]
                                at_db = ho[None, :]
                                at_kptr = (
                                    (slot * n_kv + kv_h) * max_len + at_tb
                                ) * hd + at_db
                                at_vm = at_tmask[:, None] & (at_db < hd)
                                at_kv = tl.load(
                                    K_PACK + at_kptr, mask=at_vm, other=0.0
                                ).to(tl.float32)
                                at_vv = tl.load(
                                    V_PACK + at_kptr, mask=at_vm, other=0.0
                                ).to(tl.float32)
                                at_s = scale * tl.sum(at_kv * qv[None, :], axis=1)
                                at_s = tl.where(at_tmask, at_s, -1.0e9)
                                at_mt = tl.max(at_s, axis=0)
                                at_mn = tl.maximum(at_m, at_mt)
                                at_al = tl.exp(at_m - at_mn)
                                at_p = tl.exp(at_s - at_mn)
                                at_p = tl.where(at_tmask, at_p, 0.0)
                                at_lse = at_lse * at_al + tl.sum(at_p, axis=0)
                                at_acc = at_acc * at_al + tl.sum(
                                    at_p[:, None] * at_vv, axis=0
                                )
                                at_m = at_mn
                                at_t0 += BLOCK_L
                            kb = ((slot * n_kv + kv_h) * max_len + tail) * hd
                            k_cur = tl.load(
                                K_PACK + kb + ho, mask=hmask, other=0.0
                            ).to(tl.float32)
                            v_cur = tl.load(
                                V_PACK + kb + ho, mask=hmask, other=0.0
                            ).to(tl.float32)
                            at_s = scale * tl.sum(qv * k_cur, axis=0)
                            at_mn = tl.maximum(at_m, at_s)
                            at_al = tl.exp(at_m - at_mn)
                            at_p = tl.exp(at_s - at_mn)
                            at_lse = at_lse * at_al + at_p
                            at_acc = at_acc * at_al + at_p * v_cur
                            at_yv = at_acc / at_lse
                            if use_xsa != 0:
                                at_vn = tl.maximum(
                                    tl.sqrt(tl.sum(v_cur * v_cur, axis=0)), 1e-12
                                )
                                at_hat = v_cur / at_vn
                                at_yv = at_yv - tl.sum(at_yv * at_hat, axis=0) * at_hat
                            if use_gate != 0:
                                at_go = tl.arange(0, BLOCK_G)
                                at_gm = at_go < gate_n
                                at_xv = tl.load(
                                    ATTN_IN + at_go, mask=at_gm, other=0.0
                                ).to(tl.float32)
                                at_gw = tl.load(
                                    GATE_W + (layer * n_heads + head) * gate_n + at_go,
                                    mask=at_gm,
                                    other=0.0,
                                ).to(tl.float32)
                                at_yv = at_yv * (
                                    1.0
                                    / (1.0 + tl.exp(-tl.sum(at_xv * at_gw, axis=0)))
                                )
                            tl.store(
                                Y + head * hd + ho,
                                at_yv.to(Y.dtype.element_ty),
                                mask=hmask,
                            )
                        head += n_cta
                if STAGE == 0:
                    _grid_sync_ptx(ARRIVED, SENSE, n_cta)

                if (STAGE == 0) | (STAGE == 3):
                    if DUMP:
                        tl.store(
                            DUMP_D + (slot * 5 + 1) * d + offs,
                            tl.load(Y + offs, mask=dmask, other=0.0),
                            mask=dmask,
                        )
                        tl.store(
                            DUMP_Q + slot * n_heads * hd + offs,
                            tl.load(Q + offs, mask=dmask, other=0.0),
                            mask=dmask,
                        )
                    _gemv_blk(
                        Y,
                        OUT_W + layer * d * d,
                        HID,
                        d,
                        d,
                        pid,
                        n_cta,
                        16,
                        BLOCK_K,
                        0,
                        soft_pos,
                        soft_neg,
                        asym_cap,
                    )
                    if STAGE == 0:
                        _grid_sync_ptx(ARRIVED, SENSE, n_cta)

                    attn_out = tl.load(HID + offs, mask=dmask, other=0.0).to(tl.float32)
                    a_sc = tl.load(A_SC + layer * d + offs, mask=dmask, other=0.0).to(tl.float32)
                    ln = tl.load(LN + layer)
                    if DUMP:
                        if pid == 0:
                            tl.store(
                                DUMP_D + (slot * 5 + 4) * d + offs,
                                attn_out.to(DUMP_D.dtype.element_ty),
                                mask=dmask,
                            )
                    if par:
                        tl.store(Y + offs, (attn_out * a_sc).to(Y.dtype.element_ty), mask=dmask)
                        lane1 = tl.load(LANE1 + offs, mask=dmask, other=0.0).to(tl.float32)
                        tl.store(
                            ATTN_IN + offs,
                            (_rms_vec(lane1, d, 1e-6) * ln).to(ATTN_IN.dtype.element_ty),
                            mask=dmask,
                        )
                    else:
                        xin = tl.load(XIN + offs, mask=dmask, other=0.0).to(tl.float32)
                        xout = xin + a_sc * attn_out
                        tl.store(XIN + offs, xout.to(XIN.dtype.element_ty), mask=dmask)
                        tl.store(
                            ATTN_IN + offs,
                            (_rms_vec(xout, d, 1e-6) * ln).to(ATTN_IN.dtype.element_ty),
                            mask=dmask,
                        )

                    _gemv_blk(
                        ATTN_IN,
                        UP + layer * mlp_dim * d,
                        SCR,
                        mlp_dim,
                        d,
                        pid,
                        n_cta,
                        32,
                        BLOCK_K,
                        1,
                        soft_pos,
                        soft_neg,
                        asym_cap,
                    )
                    if STAGE == 0:
                        _grid_sync_ptx(ARRIVED, SENSE, n_cta)
                    _gemv_blk(
                        SCR,
                        DOWN + layer * d * mlp_dim,
                        HID,
                        d,
                        mlp_dim,
                        pid,
                        n_cta,
                        16,
                        BLOCK_K,
                        0,
                        soft_pos,
                        soft_neg,
                        asym_cap,
                    )
                    if STAGE == 0:
                        _grid_sync_ptx(ARRIVED, SENSE, n_cta)

                    mlp = tl.load(HID + offs, mask=dmask, other=0.0).to(tl.float32)
                    m_sc = tl.load(M_SC + layer * d + offs, mask=dmask, other=0.0).to(tl.float32)
                    if par:
                        attn_s = tl.load(Y + offs, mask=dmask, other=0.0).to(tl.float32)
                        mlp_s = mlp * m_sc
                        pr0 = tl.load(PR + layer * 2 + 0)
                        pr1 = tl.load(PR + layer * 2 + 1)
                        pp0 = tl.load(PP + layer * 4 + 0)
                        pp1 = tl.load(PP + layer * 4 + 1)
                        pp2 = tl.load(PP + layer * 4 + 2)
                        pp3 = tl.load(PP + layer * 4 + 3)
                        l0 = tl.load(LANE0 + offs, mask=dmask, other=0.0).to(tl.float32)
                        l1 = tl.load(LANE1 + offs, mask=dmask, other=0.0).to(tl.float32)
                        tl.store(
                            LANE0 + offs,
                            (pr0 * l0 + pp0 * attn_s + pp2 * mlp_s).to(LANE0.dtype.element_ty),
                            mask=dmask,
                        )
                        tl.store(
                            LANE1 + offs,
                            (pr1 * l1 + pp1 * attn_s + pp3 * mlp_s).to(LANE1.dtype.element_ty),
                            mask=dmask,
                        )
                    else:
                        xout = tl.load(XIN + offs, mask=dmask, other=0.0).to(tl.float32)
                        xout = xout + m_sc * mlp
                        tl.store(X + offs, xout.to(X.dtype.element_ty), mask=dmask)
                        if skip_dst >= 0:
                            tl.store(
                                SKIPS + skip_dst * d + offs,
                                xout.to(SKIPS.dtype.element_ty),
                                mask=dmask,
                            )
                    if DUMP:
                        if par:
                            tl.store(
                                DUMP_D + (slot * 5 + 2) * d + offs,
                                tl.load(LANE0 + offs, mask=dmask, other=0.0),
                                mask=dmask,
                            )
                            tl.store(
                                DUMP_D + (slot * 5 + 3) * d + offs,
                                tl.load(LANE1 + offs, mask=dmask, other=0.0),
                                mask=dmask,
                            )
                        else:
                            tl.store(
                                DUMP_D + (slot * 5 + 2) * d + offs,
                                tl.load(X + offs, mask=dmask, other=0.0),
                                mask=dmask,
                            )
                if STAGE == 0:
                    _grid_sync_ptx(ARRIVED, SENSE, n_cta)
                slot += 1
                n_left -= 1
        if (STAGE == 0) | (STAGE == 4):
            if has_par != 0:
                l0 = tl.load(LANE0 + offs, mask=dmask, other=0.0).to(tl.float32)
                l1 = tl.load(LANE1 + offs, mask=dmask, other=0.0).to(tl.float32)
                if final_mode == 1:
                    x = l0
                elif final_mode == 2:
                    x = l1
                else:
                    x = 0.5 * (l0 + l1)
            else:
                x = tl.load(X + offs, mask=dmask, other=0.0).to(tl.float32)
            x = _rms_vec(x, d, 1e-6)
            tl.store(X + offs, x.to(X.dtype.element_ty), mask=dmask)
            _gemv_blk(
                X,
                LM_W,
                LOGITS,
                vocab,
                d,
                pid,
                n_cta,
                16,
                BLOCK_K,
                2,
                soft_pos,
                soft_neg,
                asym_cap,
            )

    @triton.jit
    def _mega_attn(
        Q,
        K_PACK,
        V_PACK,
        Y,
        VBUF,
        ATTN_IN,
        GATE_W,
        slot,
        POS,
        tail,
        MAX_LEN,
        HD,
        N_HEADS,
        N_KV,
        GATE_N,
        layer,
        use_xsa,
        use_gate,
        NEPS,
        BLOCK_H: tl.constexpr,
        BLOCK_L: tl.constexpr,
        BLOCK_G: tl.constexpr,
    ):
        # Same math as persist ``_attn_heads``. Lives in this module so a
        # persist-side compile experiment cannot take mega down.
        head = tl.program_id(0)
        pos = tl.load(POS)
        if head < N_HEADS:
            kv_h = head // (N_HEADS // N_KV)
            hoffs = tl.arange(0, BLOCK_H)
            hmask = hoffs < HD
            qv = tl.load(Q + head * HD + hoffs, mask=hmask, other=0.0).to(tl.float32)
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
                vb = tl.load(VBUF + kv_h * HD + hoffs, mask=hmask, other=0.0).to(
                    tl.float32
                )
                vnorm = tl.maximum(tl.sqrt(tl.sum(vb * vb, axis=0)), NEPS)
                vn = vb / vnorm
                yv = yv - tl.sum(yv * vn, axis=0) * vn
            if use_gate != 0:
                go = tl.arange(0, BLOCK_G)
                gmask = go < GATE_N
                xv = tl.load(ATTN_IN + go, mask=gmask, other=0.0).to(tl.float32)
                gw = tl.load(
                    GATE_W + (layer * N_HEADS + head) * GATE_N + go,
                    mask=gmask,
                    other=0.0,
                ).to(tl.float32)
                yv = yv * (1.0 / (1.0 + tl.exp(-tl.sum(xv * gw, axis=0))))
            tl.store(Y + head * HD + hoffs, yv.to(Y.dtype.element_ty), mask=hmask)

    @triton.jit
    def _mega_attn_dot(
        Q,
        K_PACK,
        V_PACK,
        Y,
        VBUF,
        ATTN_IN,
        GATE_W,
        slot,
        POS,
        tail,
        MAX_LEN,
        HD,
        N_HEADS,
        N_KV,
        GATE_N,
        layer,
        use_xsa,
        use_gate,
        NEPS,
        BLOCK_H: tl.constexpr,
        BLOCK_L: tl.constexpr,
        BLOCK_G: tl.constexpr,
    ):
        # Thin mega scan with tensor-core QK/PV. Same online softmax + mask
        # as ``_mega_attn``. Do not add cache_modifier (this Triton rejects .cs).
        # Do not put this body inside the QKV megakernel.
        head = tl.program_id(0)
        pos = tl.load(POS)
        if head < N_HEADS:
            kv_h = head // (N_HEADS // N_KV)
            hoffs = tl.arange(0, BLOCK_H)
            hmask = hoffs < HD
            qv = tl.load(Q + head * HD + hoffs, mask=hmask, other=0.0).to(tl.float32)
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
                vb = tl.load(VBUF + kv_h * HD + hoffs, mask=hmask, other=0.0).to(
                    tl.float32
                )
                vnorm = tl.maximum(tl.sqrt(tl.sum(vb * vb, axis=0)), NEPS)
                vn = vb / vnorm
                yv = yv - tl.sum(yv * vn, axis=0) * vn
            if use_gate != 0:
                go = tl.arange(0, BLOCK_G)
                gmask = go < GATE_N
                xv = tl.load(ATTN_IN + go, mask=gmask, other=0.0).to(tl.float32)
                gw = tl.load(
                    GATE_W + (layer * N_HEADS + head) * GATE_N + go,
                    mask=gmask,
                    other=0.0,
                ).to(tl.float32)
                yv = yv * (1.0 / (1.0 + tl.exp(-tl.sum(xv * gw, axis=0))))
            tl.store(Y + head * HD + hoffs, yv.to(Y.dtype.element_ty), mask=hmask)


def attach_mega_workspace(ws: "_PersistWS", device: torch.device) -> None:
    n_cta = mega_cta_count(device, n_heads=int(ws.n_heads))
    hd = int(ws.hd)
    block_h = 1
    while block_h < hd:
        block_h *= 2
    ws.n_cta = n_cta
    ws.sense = torch.zeros(1, dtype=torch.int32, device=device)
    ws.arrived = torch.zeros(1, dtype=torch.int32, device=device)
    ws.part_m = torch.zeros(n_cta, dtype=torch.float32, device=device)
    ws.part_lse = torch.zeros(n_cta, dtype=torch.float32, device=device)
    ws.part_acc = torch.zeros(n_cta, block_h, dtype=torch.float32, device=device)
    n_heads = int(ws.n_heads)
    ws.mega_q = torch.zeros(n_cta * hd, device=device, dtype=ws.q.dtype)
    ws.mega_k = torch.zeros(n_heads * hd, device=device, dtype=ws.q.dtype)
    ws.mega_v = torch.zeros(n_heads * hd, device=device, dtype=ws.q.dtype)
    ws.bar_flags = torch.zeros(n_cta, dtype=torch.int32, device=device)
    ws.use_mega = True
    from .model import _mark_static_address

    for t in (
        ws.sense,
        ws.arrived,
        ws.part_m,
        ws.part_lse,
        ws.part_acc,
        ws.mega_q,
        ws.mega_k,
        ws.mega_v,
        ws.bar_flags,
    ):
        _mark_static_address(t)
    from .split_attn import ensure_split_workspace

    ensure_split_workspace(ws, 1)
    if mega_dump_on():
        attach_layer_dumps(ws, device)


def mega_attn_slot(state: "StaticState", ws: "_PersistWS", slot: int) -> None:
    """Mega's thin prefix scan. Tensor-core first; fp32 if that kernel fails.

    A compile miss here must not disable mega or persist — only this scan
    falls back.
    """
    from .persistent_step import _NORM_EPS, _next_pow2

    layer, _kind, _ss, _sd, _sw, use_xsa, use_gate = ws.itin_host[slot]
    tail = ws.max_len - 1
    ws.vbuf.copy_(ws.v_pack[slot, :, tail].reshape(-1))
    ws.kbuf.copy_(ws.k_pack[slot, :, tail].reshape(-1))
    args = (
        ws.q,
        ws.k_pack,
        ws.v_pack,
        ws.y,
        ws.vbuf,
        ws.attn_in,
        ws.gate_w,
        slot,
        state.pos,
        tail,
        ws.max_len,
        ws.hd,
        ws.n_heads,
        ws.n_kv,
        ws.gate_n,
        layer,
        use_xsa,
        use_gate,
        _NORM_EPS,
    )
    block_h = _next_pow2(ws.hd)
    block_g = _next_pow2(max(ws.gate_n, 1))
    from .split_attn import run_split_attn

    split_count = int(getattr(ws, "attn_splits", 1))
    if split_count > 1 and not getattr(ws, "_mega_attn_split_failed", False):
        try:
            try:
                split_block_l = int(os.environ.get("XSA_AC_ATTN_BLOCK", "256"))
            except ValueError:
                split_block_l = 256
            split_block_l = min(max(_next_pow2(split_block_l), 16), 1024)
            if run_split_attn(
                ws=ws,
                pos=state.pos,
                slot=slot,
                rows=1,
                layer=layer,
                use_xsa=use_xsa,
                use_gate=use_gate,
                v_cur=ws.vbuf,
                v_cur_row_stride=ws.n_kv * ws.hd,
                tail_at_pos=False,
                norm_eps=_NORM_EPS,
                block_l=split_block_l,
            ):
                ws.mega_attn = f"split{split_count}"
                return
        except Exception as exc:
            ws._mega_attn_split_failed = True
            if not getattr(ws, "_mega_attn_split_logged", False):
                import sys

                ws._mega_attn_split_logged = True
                cause = exc.args[-1] if exc.args else exc
                print(
                    f"[AC incr] split-prefix decode attn unavailable "
                    f"({cause!r}); using serial mega attn",
                    file=sys.stderr,
                    flush=True,
                )
    want_dot = _env_on("XSA_AC_ATTN_DOT", "1") and not getattr(
        ws, "_mega_attn_dot_failed", False
    )
    if want_dot:
        try:
            try:
                block_l = int(os.environ.get("XSA_AC_ATTN_BLOCK", "256"))
            except ValueError:
                block_l = 256
            p = 16
            while p < block_l:
                p *= 2
            block_l = min(max(p, 16), 1024)
            _mega_attn_dot[(ws.n_heads,)](
                *args,
                BLOCK_H=block_h,
                BLOCK_L=block_l,
                BLOCK_G=block_g,
                num_warps=8,
                num_stages=3,
            )
            ws.mega_attn = "dot"
            return
        except Exception as exc:
            ws._mega_attn_dot_failed = True
            if not getattr(ws, "_mega_attn_dot_logged", False):
                import sys

                ws._mega_attn_dot_logged = True
                cause = exc.args[-1] if exc.args else exc
                print(
                    f"[AC incr] mega attn tl.dot unavailable ({cause!r}); "
                    "using fp32 mega attn (mega stays on)",
                    file=sys.stderr,
                    flush=True,
                )
    _mega_attn[(ws.n_heads,)](
        *args,
        BLOCK_H=block_h,
        BLOCK_L=128,
        BLOCK_G=block_g,
        num_warps=4,
    )
    ws.mega_attn = "fp32"


def try_mega_extend(model: "GPT", state: "StaticState", ws: "_PersistWS") -> bool:
    if not _HAS_TRITON or not getattr(ws, "use_mega", False):
        return False
    qkv = model._ac_qkv_bank
    up = model._ac_mlp_up
    down = model._ac_mlp_down
    if qkv is None or up is None or down is None:
        return False
    d = ws.d
    hd = ws.hd
    block_d = 1
    while block_d < d:
        block_d *= 2
    block_h = 1
    while block_h < hd:
        block_h *= 2
    block_g = 1
    while block_g < max(ws.gate_n, 1):
        block_g *= 2
    dump_on = getattr(ws, "dump_d", None) is not None
    dump_d = ws.dump_d if dump_on else ws.x
    dump_q = ws.dump_q if dump_on else ws.q
    n_heads = int(ws.n_heads)
    n_slots = int(ws.n_slots)

    def _launch(
        grid: int, n_cta: int, slot_sel: int, stage: int, *, cooperative: bool = False
    ) -> None:
        launch_kw: dict = dict(num_warps=4, num_stages=1)
        if cooperative:
            launch_kw["launch_cooperative_grid"] = True
        args = (
            state.token.view(-1),
            state.prev_raw.view(-1),
            state.cand_prev_raw.view(-1),
            state.pos.view(-1),
            state.logits.view(-1),
            ws.x,
            ws.x0,
            ws.xin,
            ws.attn_in,
            ws.scr,
            ws.q,
            ws.mega_k,
            ws.mega_v,
            ws.y,
            ws.hid,
            ws.lane0,
            ws.lane1,
            ws.skips,
            ws.k_pack,
            ws.v_pack,
            ws.mix,
            ws.a_sc,
            ws.m_sc,
            ws.q_gain,
            ws.gate_w,
            ws.ln,
            ws.pr,
            ws.pp,
            ws.itin,
            ws.cos,
            ws.sin,
            ws.smear_w,
            ws.skip_w,
            ws.skip_g,
            qkv,
            up,
            down,
            ws.out_w,
            ws.tok_emb,
            ws.lm_w,
            ws.part_m,
            ws.part_lse,
            ws.part_acc,
            ws.arrived,
            ws.sense,
            ws.bar_flags,
            dump_d,
            dump_q,
            slot_sel,
            n_cta,
            ws.n_slots,
            ws.n_enc,
            d,
            hd,
            ws.n_heads,
            ws.n_kv,
            ws.qkv_dim,
            ws.mlp_dim,
            ws.vocab,
            ws.max_len,
            ws.rope_half,
            ws.gate_n,
            int(ws.has_smear),
            int(ws.has_skip_gate),
            int(ws.has_par),
            ws.final_mode,
            int(ws.asym_cap),
            float(ws.smear_lam),
            float(ws.soft_pos),
            float(ws.soft_neg),
        )
        const = dict(
            BLOCK_D=block_d,
            BLOCK_H=block_h,
            BLOCK_K=128,
            BLOCK_L=128,
            BLOCK_G=block_g,
            DUMP=1 if dump_on else 0,
            STAGE=stage,
            DEDUP_KV=1 if _env_on("XSA_AC_GQA_DEDUP", "1") else 0,
        )
        try:
            _mega_kernel[(grid,)](*args, **const, **launch_kw)
        except TypeError:
            launch_kw.pop("launch_cooperative_grid", None)
            _mega_kernel[(grid,)](*args, **const, **launch_kw)

    from .persistent_step import persist_embed, persist_ffn_slot, persist_logits
    from .persistent_step import _dump_layer

    # Embed in PyTorch (same as persist). Triton STAGE=1 CAND/smear is
    # invisible at step 0 (prev=0) and poisons step 1 after commit.
    persist_embed(state, ws)

    if _env_on("XSA_AC_MEGA_ONE", "0") and not getattr(ws, "_mega_one_failed", False):
        try:
            ws.arrived.zero_()
            ws.sense.zero_()
            _launch(n_heads, n_heads, 0, 0, cooperative=True)
            ws.mega_mode = "one"
            return True
        except Exception as exc:
            ws._mega_one_failed = True
            cause = exc.args[-1] if exc.args else exc
            ws._mega_error = f"MEGA_ONE: {cause!r}"

    try:
        for slot in range(n_slots):
            _launch(n_heads, n_heads, slot, 2)
            mega_attn_slot(state, ws, slot)
            if dump_on:
                _dump_layer(ws, slot, DUMP_Y, ws.y)
                dump_q[slot].copy_(ws.q)
            persist_ffn_slot(model, ws, slot)
        persist_logits(state, ws)
        ws.mega_mode = "mega"
    except Exception as exc:
        cause = exc.args[-1] if exc.args else exc
        ws._mega_error = repr(cause)
        return False
    return True
