"""Triton causal attention for train / XM probe / segment prefill.

Same contract as ``model._causal_attn_chunk``: causal ``col <= row`` mask and
fp32 softmax (online, tiled). Encode and decode both go through
``_chunked_causal_sdpa``, so they stay lockstep. Do **not** inline this into
``mega_step.py`` or share flags with persist ``_attn_heads``.

``XSA_ATTN_TRITON=0`` falls back to the Python chunk loop.
"""

from __future__ import annotations

import math
import os
import sys

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - CPU / no-Triton wheels
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _HAS_TRITON = False

_HEAD_DIM = 64
_BLOCK_M = 64
_BLOCK_N = 64
_failed = False
_fail_reason = ""


def _env_on(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
        "",
    }


def triton_attn_enabled() -> bool:
    return (not _failed) and _HAS_TRITON and _env_on("XSA_ATTN_TRITON", "1")


def _mark_failed(exc: BaseException) -> None:
    global _failed, _fail_reason
    if _failed:
        return
    _failed = True
    _fail_reason = repr(exc)
    print(
        f"[xsa_ttt] triton causal attn failed ({_fail_reason}); "
        "using chunked fp32 softmax",
        file=sys.stderr,
        flush=True,
    )


if _HAS_TRITON:

    @triton.jit
    def _train_attn_fwd(
        Q,
        K,
        V,
        O,
        LSE,
        SCALE,
        STRIDE_QB,
        STRIDE_QH,
        STRIDE_QT,
        STRIDE_QD,
        STRIDE_KB,
        STRIDE_KH,
        STRIDE_KT,
        STRIDE_KD,
        STRIDE_VB,
        STRIDE_VH,
        STRIDE_VT,
        STRIDE_VD,
        STRIDE_OB,
        STRIDE_OH,
        STRIDE_OT,
        STRIDE_OD,
        T,
        H,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HEAD_DIM: tl.constexpr,
    ):
        start_m = tl.program_id(0) * BLOCK_M
        off_hz = tl.program_id(1)
        if start_m >= T:
            return
        off_z = off_hz // H
        off_h = off_hz % H
        q_bh = (off_z * STRIDE_QB + off_h * STRIDE_QH).to(tl.int64)
        k_bh = (off_z * STRIDE_KB + off_h * STRIDE_KH).to(tl.int64)
        v_bh = (off_z * STRIDE_VB + off_h * STRIDE_VH).to(tl.int64)
        o_bh = (off_z * STRIDE_OB + off_h * STRIDE_OH).to(tl.int64)

        offs_m = start_m + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM)
        qmask = (offs_m[:, None] < T) & (offs_d[None, :] < HEAD_DIM)
        fa_q = tl.load(
            Q + q_bh + offs_m[:, None] * STRIDE_QT + offs_d[None, :] * STRIDE_QD,
            mask=qmask,
            other=0.0,
        ).to(tl.bfloat16)

        fa_m = tl.zeros([BLOCK_M], dtype=tl.float32) - 1.0e9
        fa_l = tl.zeros([BLOCK_M], dtype=tl.float32)
        fa_acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        lim_n = tl.minimum(start_m + BLOCK_M, T)
        start_n = 0
        while start_n < lim_n:
            offs_n = start_n + tl.arange(0, BLOCK_N)
            knmask = (offs_n[:, None] < T) & (offs_d[None, :] < HEAD_DIM)
            fa_k = tl.load(
                K + k_bh + offs_n[:, None] * STRIDE_KT + offs_d[None, :] * STRIDE_KD,
                mask=knmask,
                other=0.0,
            ).to(tl.bfloat16)
            fa_v = tl.load(
                V + v_bh + offs_n[:, None] * STRIDE_VT + offs_d[None, :] * STRIDE_VD,
                mask=knmask,
                other=0.0,
            ).to(tl.bfloat16)
            fa_qk = tl.dot(fa_q, tl.trans(fa_k)) * SCALE
            fa_ok = (
                (offs_m[:, None] < T)
                & (offs_n[None, :] < T)
                & (offs_n[None, :] <= offs_m[:, None])
            )
            fa_qk = tl.where(fa_ok, fa_qk, -1.0e9)
            fa_mt = tl.max(fa_qk, axis=1)
            fa_mn = tl.maximum(fa_m, fa_mt)
            fa_al = tl.exp(fa_m - fa_mn)
            fa_p = tl.exp(fa_qk - fa_mn[:, None])
            fa_p = tl.where(fa_ok, fa_p, 0.0)
            fa_l = fa_l * fa_al + tl.sum(fa_p, axis=1)
            fa_acc = fa_acc * fa_al[:, None] + tl.dot(fa_p.to(tl.bfloat16), fa_v)
            fa_m = fa_mn
            start_n += BLOCK_N

        fa_l = tl.maximum(fa_l, 1.0e-20)
        fa_out = fa_acc / fa_l[:, None]
        omask = (offs_m[:, None] < T) & (offs_d[None, :] < HEAD_DIM)
        tl.store(
            O + o_bh + offs_m[:, None] * STRIDE_OT + offs_d[None, :] * STRIDE_OD,
            fa_out.to(O.dtype.element_ty),
            mask=omask,
        )
        tl.store(
            LSE + off_hz * T + offs_m,
            fa_m + tl.log(fa_l),
            mask=offs_m < T,
        )

    @triton.jit
    def _train_attn_bwd_pre(
        O,
        DO,
        DELTA,
        STRIDE_B,
        STRIDE_H,
        STRIDE_T,
        STRIDE_D,
        T,
        H,
        BLOCK_M: tl.constexpr,
        HEAD_DIM: tl.constexpr,
    ):
        start_m = tl.program_id(0) * BLOCK_M
        off_hz = tl.program_id(1)
        if start_m >= T:
            return
        off_z = off_hz // H
        off_h = off_hz % H
        bh = (off_z * STRIDE_B + off_h * STRIDE_H).to(tl.int64)
        offs_m = start_m + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM)
        mask = (offs_m[:, None] < T) & (offs_d[None, :] < HEAD_DIM)
        pre_o = tl.load(
            O + bh + offs_m[:, None] * STRIDE_T + offs_d[None, :] * STRIDE_D,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        pre_do = tl.load(
            DO + bh + offs_m[:, None] * STRIDE_T + offs_d[None, :] * STRIDE_D,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        tl.store(
            DELTA + off_hz * T + offs_m,
            tl.sum(pre_o * pre_do, axis=1),
            mask=offs_m < T,
        )

    @triton.jit
    def _train_attn_bwd_kv(
        Q,
        K,
        V,
        DO,
        DK,
        DV,
        LSE,
        DELTA,
        SCALE,
        STRIDE_QB,
        STRIDE_QH,
        STRIDE_QT,
        STRIDE_QD,
        STRIDE_KB,
        STRIDE_KH,
        STRIDE_KT,
        STRIDE_KD,
        STRIDE_VB,
        STRIDE_VH,
        STRIDE_VT,
        STRIDE_VD,
        STRIDE_OB,
        STRIDE_OH,
        STRIDE_OT,
        STRIDE_OD,
        T,
        H,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HEAD_DIM: tl.constexpr,
    ):
        start_n = tl.program_id(0) * BLOCK_N
        off_hz = tl.program_id(1)
        if start_n >= T:
            return
        off_z = off_hz // H
        off_h = off_hz % H
        q_bh = (off_z * STRIDE_QB + off_h * STRIDE_QH).to(tl.int64)
        k_bh = (off_z * STRIDE_KB + off_h * STRIDE_KH).to(tl.int64)
        v_bh = (off_z * STRIDE_VB + off_h * STRIDE_VH).to(tl.int64)
        o_bh = (off_z * STRIDE_OB + off_h * STRIDE_OH).to(tl.int64)

        offs_n = start_n + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, HEAD_DIM)
        knmask = (offs_n[:, None] < T) & (offs_d[None, :] < HEAD_DIM)
        kv_k = tl.load(
            K + k_bh + offs_n[:, None] * STRIDE_KT + offs_d[None, :] * STRIDE_KD,
            mask=knmask,
            other=0.0,
        ).to(tl.bfloat16)
        kv_v = tl.load(
            V + v_bh + offs_n[:, None] * STRIDE_VT + offs_d[None, :] * STRIDE_VD,
            mask=knmask,
            other=0.0,
        ).to(tl.bfloat16)
        kv_dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
        kv_dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

        start_m = start_n
        while start_m < T:
            offs_m = start_m + tl.arange(0, BLOCK_M)
            qmask = (offs_m[:, None] < T) & (offs_d[None, :] < HEAD_DIM)
            kv_q = tl.load(
                Q + q_bh + offs_m[:, None] * STRIDE_QT + offs_d[None, :] * STRIDE_QD,
                mask=qmask,
                other=0.0,
            ).to(tl.bfloat16)
            kv_do = tl.load(
                DO + o_bh + offs_m[:, None] * STRIDE_OT + offs_d[None, :] * STRIDE_OD,
                mask=qmask,
                other=0.0,
            ).to(tl.bfloat16)
            kv_lse = tl.load(LSE + off_hz * T + offs_m, mask=offs_m < T, other=0.0)
            kv_di = tl.load(DELTA + off_hz * T + offs_m, mask=offs_m < T, other=0.0)
            kv_qk = tl.dot(kv_q, tl.trans(kv_k)) * SCALE
            kv_ok = (
                (offs_m[:, None] < T)
                & (offs_n[None, :] < T)
                & (offs_n[None, :] <= offs_m[:, None])
            )
            kv_p = tl.exp(kv_qk - kv_lse[:, None])
            kv_p = tl.where(kv_ok, kv_p, 0.0)
            kv_dv += tl.dot(tl.trans(kv_p.to(tl.bfloat16)), kv_do)
            kv_dp = tl.dot(kv_do, tl.trans(kv_v)).to(tl.float32)
            kv_ds = kv_p * (kv_dp - kv_di[:, None]) * SCALE
            kv_dk += tl.dot(tl.trans(kv_ds.to(tl.bfloat16)), kv_q)
            start_m += BLOCK_M

        tl.store(
            DK + k_bh + offs_n[:, None] * STRIDE_KT + offs_d[None, :] * STRIDE_KD,
            kv_dk.to(DK.dtype.element_ty),
            mask=knmask,
        )
        tl.store(
            DV + v_bh + offs_n[:, None] * STRIDE_VT + offs_d[None, :] * STRIDE_VD,
            kv_dv.to(DV.dtype.element_ty),
            mask=knmask,
        )

    @triton.jit
    def _train_attn_bwd_q(
        Q,
        K,
        V,
        DO,
        DQ,
        LSE,
        DELTA,
        SCALE,
        STRIDE_QB,
        STRIDE_QH,
        STRIDE_QT,
        STRIDE_QD,
        STRIDE_KB,
        STRIDE_KH,
        STRIDE_KT,
        STRIDE_KD,
        STRIDE_VB,
        STRIDE_VH,
        STRIDE_VT,
        STRIDE_VD,
        STRIDE_OB,
        STRIDE_OH,
        STRIDE_OT,
        STRIDE_OD,
        T,
        H,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HEAD_DIM: tl.constexpr,
    ):
        start_m = tl.program_id(0) * BLOCK_M
        off_hz = tl.program_id(1)
        if start_m >= T:
            return
        off_z = off_hz // H
        off_h = off_hz % H
        q_bh = (off_z * STRIDE_QB + off_h * STRIDE_QH).to(tl.int64)
        k_bh = (off_z * STRIDE_KB + off_h * STRIDE_KH).to(tl.int64)
        v_bh = (off_z * STRIDE_VB + off_h * STRIDE_VH).to(tl.int64)
        o_bh = (off_z * STRIDE_OB + off_h * STRIDE_OH).to(tl.int64)

        offs_m = start_m + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM)
        qmask = (offs_m[:, None] < T) & (offs_d[None, :] < HEAD_DIM)
        bq_q = tl.load(
            Q + q_bh + offs_m[:, None] * STRIDE_QT + offs_d[None, :] * STRIDE_QD,
            mask=qmask,
            other=0.0,
        ).to(tl.bfloat16)
        bq_do = tl.load(
            DO + o_bh + offs_m[:, None] * STRIDE_OT + offs_d[None, :] * STRIDE_OD,
            mask=qmask,
            other=0.0,
        ).to(tl.bfloat16)
        bq_lse = tl.load(LSE + off_hz * T + offs_m, mask=offs_m < T, other=0.0)
        bq_di = tl.load(DELTA + off_hz * T + offs_m, mask=offs_m < T, other=0.0)
        bq_dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        lim_n = tl.minimum(start_m + BLOCK_M, T)
        start_n = 0
        while start_n < lim_n:
            offs_n = start_n + tl.arange(0, BLOCK_N)
            knmask = (offs_n[:, None] < T) & (offs_d[None, :] < HEAD_DIM)
            bq_k = tl.load(
                K + k_bh + offs_n[:, None] * STRIDE_KT + offs_d[None, :] * STRIDE_KD,
                mask=knmask,
                other=0.0,
            ).to(tl.bfloat16)
            bq_v = tl.load(
                V + v_bh + offs_n[:, None] * STRIDE_VT + offs_d[None, :] * STRIDE_VD,
                mask=knmask,
                other=0.0,
            ).to(tl.bfloat16)
            bq_qk = tl.dot(bq_q, tl.trans(bq_k)) * SCALE
            bq_ok = (
                (offs_m[:, None] < T)
                & (offs_n[None, :] < T)
                & (offs_n[None, :] <= offs_m[:, None])
            )
            bq_p = tl.exp(bq_qk - bq_lse[:, None])
            bq_p = tl.where(bq_ok, bq_p, 0.0)
            bq_dp = tl.dot(bq_do, tl.trans(bq_v)).to(tl.float32)
            bq_ds = bq_p * (bq_dp - bq_di[:, None]) * SCALE
            bq_dq += tl.dot(bq_ds.to(tl.bfloat16), bq_k)
            start_n += BLOCK_N

        tl.store(
            DQ + q_bh + offs_m[:, None] * STRIDE_QT + offs_d[None, :] * STRIDE_QD,
            bq_dq.to(DQ.dtype.element_ty),
            mask=qmask,
        )


def _can_launch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    if not triton_attn_enabled() or not q.is_cuda:
        return False
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        return False
    if q.ndim != 4 or k.shape != q.shape or v.shape != q.shape:
        return False
    if int(q.shape[-1]) != _HEAD_DIM:
        return False
    if int(q.shape[2]) < 1:
        return False
    return True


def _fwd_launch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    bsz, n_heads, seqlen, _ = q.shape
    o = torch.empty_like(q)
    lse = torch.empty(bsz, n_heads, seqlen, device=q.device, dtype=torch.float32)
    scale = 1.0 / math.sqrt(_HEAD_DIM)
    grid = (triton.cdiv(seqlen, _BLOCK_M), bsz * n_heads)
    _train_attn_fwd[grid](
        q,
        k,
        v,
        o,
        lse,
        scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        seqlen,
        n_heads,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        HEAD_DIM=_HEAD_DIM,
        num_warps=4,
        num_stages=1,
    )
    return o, lse


def _bwd_launch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    do: torch.Tensor,
):
    bsz, n_heads, seqlen, _ = q.shape
    scale = 1.0 / math.sqrt(_HEAD_DIM)
    delta = torch.empty_like(lse)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    pre_grid = (triton.cdiv(seqlen, _BLOCK_M), bsz * n_heads)
    _train_attn_bwd_pre[pre_grid](
        o,
        do,
        delta,
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        seqlen,
        n_heads,
        BLOCK_M=_BLOCK_M,
        HEAD_DIM=_HEAD_DIM,
        num_warps=4,
        num_stages=1,
    )
    kv_grid = (triton.cdiv(seqlen, _BLOCK_N), bsz * n_heads)
    _train_attn_bwd_kv[kv_grid](
        q,
        k,
        v,
        do,
        dk,
        dv,
        lse,
        delta,
        scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        seqlen,
        n_heads,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        HEAD_DIM=_HEAD_DIM,
        num_warps=4,
        num_stages=1,
    )
    q_grid = (triton.cdiv(seqlen, _BLOCK_M), bsz * n_heads)
    _train_attn_bwd_q[q_grid](
        q,
        k,
        v,
        do,
        dq,
        lse,
        delta,
        scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        seqlen,
        n_heads,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        HEAD_DIM=_HEAD_DIM,
        num_warps=4,
        num_stages=1,
    )
    return dq, dk, dv


class _TritonCausalAttn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        o, lse = _fwd_launch(q, k, v)
        # Always save. Non-reentrant gradient checkpointing
        # (``use_reentrant=False``) still records this Function in the graph
        # while inner Q/K/V often report ``requires_grad=False`` during the
        # pack pass — gating on that left ``saved_tensors`` empty and blew
        # up retrain backward (``expected 5, got 0``).
        ctx.save_for_backward(q, k, v, o, lse)
        return o

    @staticmethod
    def backward(ctx, do: torch.Tensor):
        q, k, v, o, lse = ctx.saved_tensors
        do = do.contiguous()
        dq, dk, dv = _bwd_launch(q, k, v, o, lse, do)
        return dq, dk, dv


def triton_causal_sdpa(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor | None:
    """``(B, H, T, D)`` causal attention, or ``None`` to keep the chunked path."""
    if not _can_launch(q, k, v):
        return None
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    try:
        out = _TritonCausalAttn.apply(q, k, v)
    except Exception as exc:  # pragma: no cover - compile / launch
        _mark_failed(exc)
        return None
    return out
