"""Thin fused AC slot: Triton pointwise clusters + cuBLAS GEMM/attn.

Prologue (one launch): resid mix + RMSNorm + ln scale.
Pre-attn (one launch): Q/K RMSNorm + RoPE + q-gain; K written to the tail.
Post-attn (one launch): XSA + sparse gate.
Mid / act / residual: attn-scale + MLP RMSNorm, leaky(0.5)², MLP residual.

Fat ops stay ``F.linear`` (packed QKV, out, up, down) and
``_static_window_attention``. CPU and missing-Triton paths use the eager
PyTorch equivalents so tests stay bit-identical.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from .model import StaticKV, apply_rotary_emb, _static_window_attention

if TYPE_CHECKING:
    from .model import GPT

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - CPU / no-Triton wheels
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _HAS_TRITON = False

_RMS_EPS = 1e-6
_NORM_EPS = 1e-12


def _triton_ok(x: torch.Tensor) -> bool:
    return bool(_HAS_TRITON and x.is_cuda)


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def _flat_rows(x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """``(B, W, D)`` → contiguous ``(N, D)``."""
    n, dim = int(x.numel() // x.shape[-1]), int(x.shape[-1])
    return x.reshape(n, dim).contiguous(), n, dim


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

if _HAS_TRITON:

    @triton.jit
    def _prologue_kernel(
        X,
        X0,
        MIX0,
        MIX1,
        XIN,
        ATTN,
        LN,
        EPS,
        N,
        D,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = (row < N) & (offs < D)
        x = tl.load(X + row * D + offs, mask=mask, other=0.0).to(tl.float32)
        x0 = tl.load(X0 + row * D + offs, mask=mask, other=0.0).to(tl.float32)
        m0 = tl.load(MIX0 + offs, mask=offs < D, other=0.0).to(tl.float32)
        m1 = tl.load(MIX1 + offs, mask=offs < D, other=0.0).to(tl.float32)
        xin = m0 * x + m1 * x0
        var = tl.sum(xin * xin, axis=0) / D
        rstd = tl.rsqrt(var + EPS)
        attn = xin * rstd * LN
        tl.store(XIN + row * D + offs, xin.to(XIN.dtype.element_ty), mask=mask)
        tl.store(ATTN + row * D + offs, attn.to(ATTN.dtype.element_ty), mask=mask)

    @triton.jit
    def _qk_prep_kernel(
        X,
        OUT,
        COS,
        SIN,
        POS,
        ARANGE_W,
        GAIN,
        LN_EPS,
        STRIDE_TOK,
        STRIDE_HEAD,
        COS_STRIDE,
        WIN,
        HEAD_DIM,
        ROPE_HALF,
        N_TOK,
        N_HEADS,
        HAS_GAIN: tl.constexpr,
        BLOCK: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        tok = tl.program_id(0)
        head = tl.program_id(1)
        if tok >= N_TOK or head >= N_HEADS:
            return
        offs = tl.arange(0, BLOCK)
        mask = offs < HEAD_DIM
        base = tok * STRIDE_TOK + head * STRIDE_HEAD
        x = tl.load(X + base + offs, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=0) / HEAD_DIM
        rstd = tl.rsqrt(var + LN_EPS)
        xn = x * rstd
        gain = 1.0
        if HAS_GAIN:
            gain = tl.load(GAIN + head).to(tl.float32)
            xn = xn * gain
        tl.store(OUT + base + offs, xn.to(OUT.dtype.element_ty), mask=mask)
        pos = tl.load(POS)
        w = tok % WIN
        idx = pos + tl.load(ARANGE_W + w)
        roffs = tl.arange(0, BLOCK_R)
        rmask = roffs < ROPE_HALF
        cos = tl.load(COS + idx * COS_STRIDE + roffs, mask=rmask, other=0.0).to(
            tl.float32
        )
        sin = tl.load(SIN + idx * COS_STRIDE + roffs, mask=rmask, other=0.0).to(
            tl.float32
        )
        x1 = tl.load(X + base + roffs, mask=rmask, other=0.0).to(tl.float32) * rstd
        x2 = tl.load(X + base + ROPE_HALF + roffs, mask=rmask, other=0.0).to(
            tl.float32
        ) * rstd
        if HAS_GAIN:
            x1 = x1 * gain
            x2 = x2 * gain
        y1 = x1 * cos + x2 * sin
        y2 = -x1 * sin + x2 * cos
        tl.store(OUT + base + roffs, y1.to(OUT.dtype.element_ty), mask=rmask)
        tl.store(OUT + base + ROPE_HALF + roffs, y2.to(OUT.dtype.element_ty), mask=rmask)

    @triton.jit
    def _post_attn_kernel(
        Y,
        V,
        X,
        GW,
        X_STRIDE,
        N_HEADS,
        N_KV,
        HEAD_DIM,
        GATE_W,
        GATE_SCALE,
        NORM_EPS,
        N_TOK,
        USE_XSA: tl.constexpr,
        USE_GATE: tl.constexpr,
        BLOCK: tl.constexpr,
        BLOCK_G: tl.constexpr,
    ):
        tok = tl.program_id(0)
        head = tl.program_id(1)
        if tok >= N_TOK or head >= N_HEADS:
            return
        offs = tl.arange(0, BLOCK)
        mask = offs < HEAD_DIM
        y_base = (tok * N_HEADS + head) * HEAD_DIM
        y = tl.load(Y + y_base + offs, mask=mask, other=0.0).to(tl.float32)
        if USE_XSA:
            group = N_HEADS // N_KV
            kv_h = head // group
            v = tl.load(
                V + (tok * N_KV + kv_h) * HEAD_DIM + offs, mask=mask, other=0.0
            ).to(tl.float32)
            vnorm = tl.sqrt(tl.sum(v * v, axis=0))
            vnorm = tl.maximum(vnorm, NORM_EPS)
            vn = v / vnorm
            y = y - tl.sum(y * vn, axis=0) * vn
        if USE_GATE:
            go = tl.arange(0, BLOCK_G)
            gmask = go < GATE_W
            xv = tl.load(X + tok * X_STRIDE + go, mask=gmask, other=0.0).to(tl.float32)
            gw = tl.load(GW + head * GATE_W + go, mask=gmask, other=0.0).to(tl.float32)
            acc = GATE_SCALE * tl.sum(xv * gw, axis=0)
            y = y * (1.0 / (1.0 + tl.exp(-acc)))
        tl.store(Y + y_base + offs, y.to(Y.dtype.element_ty), mask=mask)

    @triton.jit
    def _mid_kernel(
        XIN,
        ATTN,
        SCALE,
        XOUT,
        MLPIN,
        LN,
        EPS,
        N,
        D,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = (row < N) & (offs < D)
        xin = tl.load(XIN + row * D + offs, mask=mask, other=0.0).to(tl.float32)
        att = tl.load(ATTN + row * D + offs, mask=mask, other=0.0).to(tl.float32)
        sc = tl.load(SCALE + offs, mask=offs < D, other=0.0).to(tl.float32)
        xout = xin + sc * att
        var = tl.sum(xout * xout, axis=0) / D
        rstd = tl.rsqrt(var + EPS)
        tl.store(XOUT + row * D + offs, xout.to(XOUT.dtype.element_ty), mask=mask)
        tl.store(
            MLPIN + row * D + offs,
            (xout * rstd * LN).to(MLPIN.dtype.element_ty),
            mask=mask,
        )

    @triton.jit
    def _rmsnorm_kernel(
        X,
        OUT,
        LN,
        EPS,
        N,
        D,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = (row < N) & (offs < D)
        x = tl.load(X + row * D + offs, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=0) / D
        rstd = tl.rsqrt(var + EPS)
        tl.store(OUT + row * D + offs, (x * rstd * LN).to(OUT.dtype.element_ty), mask=mask)

    @triton.jit
    def _scale_kernel(
        X,
        SCALE,
        OUT,
        N,
        D,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = (row < N) & (offs < D)
        x = tl.load(X + row * D + offs, mask=mask, other=0.0).to(tl.float32)
        sc = tl.load(SCALE + offs, mask=offs < D, other=0.0).to(tl.float32)
        tl.store(OUT + row * D + offs, (x * sc).to(OUT.dtype.element_ty), mask=mask)

    @triton.jit
    def _sq_leaky_kernel(
        X,
        N,
        D,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = (row < N) & (offs < D)
        h = tl.load(X + row * D + offs, mask=mask, other=0.0).to(tl.float32)
        h = tl.where(h >= 0.0, h, 0.5 * h)
        tl.store(X + row * D + offs, (h * h).to(X.dtype.element_ty), mask=mask)

    @triton.jit
    def _resid_kernel(
        X,
        Y,
        SCALE,
        OUT,
        N,
        D,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = (row < N) & (offs < D)
        x = tl.load(X + row * D + offs, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(Y + row * D + offs, mask=mask, other=0.0).to(tl.float32)
        sc = tl.load(SCALE + offs, mask=offs < D, other=0.0).to(tl.float32)
        tl.store(OUT + row * D + offs, (x + sc * y).to(OUT.dtype.element_ty), mask=mask)

    @triton.jit
    def _parallel_mix_kernel(
        L0,
        L1,
        ATTN,
        MLP,
        PR,
        PP,
        OUT0,
        OUT1,
        N,
        D,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = (row < N) & (offs < D)
        l0 = tl.load(L0 + row * D + offs, mask=mask, other=0.0).to(tl.float32)
        l1 = tl.load(L1 + row * D + offs, mask=mask, other=0.0).to(tl.float32)
        att = tl.load(ATTN + row * D + offs, mask=mask, other=0.0).to(tl.float32)
        mlp = tl.load(MLP + row * D + offs, mask=mask, other=0.0).to(tl.float32)
        pr0 = tl.load(PR + 0).to(tl.float32)
        pr1 = tl.load(PR + 1).to(tl.float32)
        ap0 = tl.load(PP + 0).to(tl.float32)
        ap1 = tl.load(PP + 1).to(tl.float32)
        mp0 = tl.load(PP + 2).to(tl.float32)
        mp1 = tl.load(PP + 3).to(tl.float32)
        tl.store(
            OUT0 + row * D + offs,
            (pr0 * l0 + ap0 * att + mp0 * mlp).to(OUT0.dtype.element_ty),
            mask=mask,
        )
        tl.store(
            OUT1 + row * D + offs,
            (pr1 * l1 + ap1 * att + mp1 * mlp).to(OUT1.dtype.element_ty),
            mask=mask,
        )


def _prologue(
    x: torch.Tensor,
    x0: torch.Tensor,
    mix: torch.Tensor,
    ln_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _triton_ok(x):
        x_in = torch.addcmul(mix[1] * x0, x, mix[0])
        attn_in = F.rms_norm(x_in, (x_in.size(-1),)) * ln_scale
        return x_in, attn_in
    xf, n, dim = _flat_rows(x)
    x0f, _, _ = _flat_rows(x0)
    x_in = torch.empty_like(xf)
    attn_in = torch.empty_like(xf)
    block = _next_pow2(dim)
    _prologue_kernel[(n,)](
        xf,
        x0f,
        mix[0].contiguous(),
        mix[1].contiguous(),
        x_in,
        attn_in,
        float(ln_scale),
        _RMS_EPS,
        n,
        dim,
        BLOCK=block,
        num_warps=4 if block <= 512 else 8,
    )
    shape = x.shape
    return x_in.view(shape), attn_in.view(shape)


def _rope_tables(attn, max_len: int, device, dtype):
    cos_tab, sin_tab = attn.rotary(max_len, device, dtype)
    rh = int(cos_tab.shape[-1])
    return cos_tab.reshape(-1, rh), sin_tab.reshape(-1, rh), rh


def _prep_qk(
    x: torch.Tensor,
    out: torch.Tensor,
    attn,
    past_kv: StaticKV,
    *,
    gain: torch.Tensor | None,
) -> None:
    """RMSNorm + RoPE (+ optional q-gain) into ``out`` (same layout as ``x``)."""
    bsz, seqlen, n_heads, head_dim = x.shape
    n_tok = bsz * seqlen
    rd = int(attn.rope_dims)
    rope_dims = rd if 0 < rd < head_dim else head_dim
    rope_half = rope_dims // 2
    if not _triton_ok(x):
        y = F.rms_norm(x, (head_dim,))
        idx = past_kv.pos + past_kv.arange_w
        cos_tab, sin_tab = attn.rotary(past_kv.max_len, x.device, x.dtype)
        cos = cos_tab.index_select(1, idx)
        sin = sin_tab.index_select(1, idx)
        y = apply_rotary_emb(y, cos, sin, attn.rope_dims)
        if gain is not None:
            y = y * gain[None, None, :, None]
        out.copy_(y)
        return
    xf = x.contiguous()
    out_f = out.reshape_as(xf)
    cos_2d, sin_2d, table_half = _rope_tables(
        attn, past_kv.max_len, x.device, x.dtype
    )
    if table_half != rope_half:
        # Table is sized to rotary.rope_dims; keep eager if they disagree.
        y = F.rms_norm(x, (head_dim,))
        idx = past_kv.pos + past_kv.arange_w
        cos_tab, sin_tab = attn.rotary(past_kv.max_len, x.device, x.dtype)
        y = apply_rotary_emb(
            y, cos_tab.index_select(1, idx), sin_tab.index_select(1, idx), attn.rope_dims
        )
        if gain is not None:
            y = y * gain[None, None, :, None]
        out.copy_(y)
        return
    block = _next_pow2(head_dim)
    block_r = _next_pow2(rope_half)
    dummy_gain = xf.view(-1)[:1]
    _qk_prep_kernel[(n_tok, n_heads)](
        xf.view(n_tok, n_heads, head_dim),
        out_f.view(n_tok, n_heads, head_dim),
        cos_2d,
        sin_2d,
        past_kv.pos,
        past_kv.arange_w,
        gain if gain is not None else dummy_gain,
        _RMS_EPS,
        n_heads * head_dim,
        head_dim,
        cos_2d.stride(0),
        seqlen,
        head_dim,
        rope_half,
        n_tok,
        n_heads,
        HAS_GAIN=gain is not None,
        BLOCK=block,
        BLOCK_R=block_r,
        num_warps=4,
    )


def _post_attn(
    y: torch.Tensor,
    v: torch.Tensor,
    x: torch.Tensor,
    attn,
) -> torch.Tensor:
    use_xsa = bool(attn.use_xsa)
    use_gate = bool(attn.sparse_attn_gate)
    if not use_xsa and not use_gate:
        return y
    if not _triton_ok(y):
        if use_xsa:
            y = attn._xsa_efficient(y, v)
        if use_gate:
            gw = getattr(attn, "_ac_gate_w", None)
            if gw is None or gw.dtype != x.dtype:
                gw = attn.attn_gate_w.to(x.dtype)
            g = torch.sigmoid(
                attn.sparse_attn_gate_scale * F.linear(x[..., : attn.gate_window], gw)
            )
            y = y * g[..., None]
        return y
    bsz, seqlen, n_heads, head_dim = y.shape
    n_tok = bsz * seqlen
    n_kv = v.size(-2)
    gate_w = int(attn.gate_window)
    gw = getattr(attn, "_ac_gate_w", None)
    if use_gate and (gw is None or gw.dtype != x.dtype):
        gw = attn.attn_gate_w.to(x.dtype)
    if gw is None:
        gw = y.new_empty(n_heads, max(gate_w, 1))
    xf, _, dim = _flat_rows(x)
    y = y.contiguous()
    _post_attn_kernel[(n_tok, n_heads)](
        y.view(n_tok, n_heads, head_dim),
        v.contiguous().view(n_tok, n_kv, head_dim),
        xf,
        gw.contiguous(),
        dim,
        n_heads,
        n_kv,
        head_dim,
        gate_w,
        float(attn.sparse_attn_gate_scale),
        _NORM_EPS,
        n_tok,
        USE_XSA=use_xsa,
        USE_GATE=use_gate,
        BLOCK=_next_pow2(head_dim),
        BLOCK_G=_next_pow2(max(gate_w, 1)),
        num_warps=4,
    )
    return y


def _mid(
    x_in: torch.Tensor,
    attn_out: torch.Tensor,
    a_sc: torch.Tensor,
    ln_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _triton_ok(x_in):
        x_out = torch.addcmul(x_in, attn_out, a_sc)
        mlp_in = F.rms_norm(x_out, (x_out.size(-1),)) * ln_scale
        return x_out, mlp_in
    xf, n, dim = _flat_rows(x_in)
    af, _, _ = _flat_rows(attn_out)
    x_out = torch.empty_like(xf)
    mlp_in = torch.empty_like(xf)
    block = _next_pow2(dim)
    _mid_kernel[(n,)](
        xf,
        af,
        a_sc.contiguous(),
        x_out,
        mlp_in,
        float(ln_scale),
        _RMS_EPS,
        n,
        dim,
        BLOCK=block,
        num_warps=4 if block <= 512 else 8,
    )
    shape = x_in.shape
    return x_out.view(shape), mlp_in.view(shape)


def _rmsnorm_ln(x: torch.Tensor, ln_scale: float) -> torch.Tensor:
    if not _triton_ok(x):
        return F.rms_norm(x, (x.size(-1),)) * ln_scale
    xf, n, dim = _flat_rows(x)
    out = torch.empty_like(xf)
    block = _next_pow2(dim)
    _rmsnorm_kernel[(n,)](
        xf,
        out,
        float(ln_scale),
        _RMS_EPS,
        n,
        dim,
        BLOCK=block,
        num_warps=4 if block <= 512 else 8,
    )
    return out.view(x.shape)


def _scale(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if not _triton_ok(x):
        return x * scale
    xf, n, dim = _flat_rows(x)
    out = torch.empty_like(xf)
    block = _next_pow2(dim)
    _scale_kernel[(n,)](
        xf,
        scale.contiguous(),
        out,
        n,
        dim,
        BLOCK=block,
        num_warps=4 if block <= 512 else 8,
    )
    return out.view(x.shape)


def _sq_leaky(h: torch.Tensor) -> torch.Tensor:
    if not _triton_ok(h):
        return F.leaky_relu(h, negative_slope=0.5).square()
    hf, n, dim = _flat_rows(h)
    block = _next_pow2(dim)
    _sq_leaky_kernel[(n,)](
        hf,
        n,
        dim,
        BLOCK=block,
        num_warps=4 if block <= 512 else 8,
    )
    return hf.view(h.shape)


def _resid(x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if not _triton_ok(x):
        return torch.addcmul(x, y, scale)
    xf, n, dim = _flat_rows(x)
    yf, _, _ = _flat_rows(y)
    out = torch.empty_like(xf)
    block = _next_pow2(dim)
    _resid_kernel[(n,)](
        xf,
        yf,
        scale.contiguous(),
        out,
        n,
        dim,
        BLOCK=block,
        num_warps=4 if block <= 512 else 8,
    )
    return out.view(x.shape)


def _parallel_mix(
    lane0: torch.Tensor,
    lane1: torch.Tensor,
    attn_out: torch.Tensor,
    mlp_out: torch.Tensor,
    pr: torch.Tensor,
    pp: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _triton_ok(lane0):
        return (
            torch.addcmul(
                torch.addcmul(pr[0] * lane0, attn_out, pp[0, 0]),
                mlp_out,
                pp[1, 0],
            ),
            torch.addcmul(
                torch.addcmul(pr[1] * lane1, attn_out, pp[0, 1]),
                mlp_out,
                pp[1, 1],
            ),
        )
    l0, n, dim = _flat_rows(lane0)
    l1, _, _ = _flat_rows(lane1)
    af, _, _ = _flat_rows(attn_out)
    mf, _, _ = _flat_rows(mlp_out)
    o0 = torch.empty_like(l0)
    o1 = torch.empty_like(l1)
    block = _next_pow2(dim)
    _parallel_mix_kernel[(n,)](
        l0,
        l1,
        af,
        mf,
        pr.contiguous(),
        pp.contiguous().view(-1),
        o0,
        o1,
        n,
        dim,
        BLOCK=block,
        num_warps=4 if block <= 512 else 8,
    )
    return o0.view(lane0.shape), o1.view(lane1.shape)


def _cublas_mlp(x: torch.Tensor, up_w: torch.Tensor, down_w: torch.Tensor) -> torch.Tensor:
    hidden = F.linear(x, up_w)
    hidden = _sq_leaky(hidden)
    return F.linear(hidden, down_w)


def _fused_attention(
    attn,
    x: torch.Tensor,
    q_w: torch.Tensor,
    k_w: torch.Tensor,
    v_w: torch.Tensor,
    out_w: torch.Tensor,
    past_kv: StaticKV,
    qkv_w: torch.Tensor | None = None,
) -> torch.Tensor:
    """Packed (or 3-way) QKV GEMM, Triton Q/K prep, cuBLAS GQA, one out-proj."""
    bsz, seqlen, dim = x.shape
    hd = attn.head_dim
    kv_dim = attn.num_kv_heads * hd
    if qkv_w is not None:
        qkv = F.linear(x, qkv_w)
        q, k, v = qkv.split((dim, kv_dim, kv_dim), dim=-1)
        q = q.reshape(bsz, seqlen, attn.num_heads, hd)
        k = k.reshape(bsz, seqlen, attn.num_kv_heads, hd)
        v = v.reshape(bsz, seqlen, attn.num_kv_heads, hd)
    else:
        q = F.linear(x, q_w).reshape(bsz, seqlen, attn.num_heads, hd)
        k = F.linear(x, k_w).reshape(bsz, seqlen, attn.num_kv_heads, hd)
        v = F.linear(x, v_w).reshape(bsz, seqlen, attn.num_kv_heads, hd)
    qg = getattr(attn, "_ac_q_gain", None)
    if qg is None or qg.dtype != q.dtype:
        qg = attn.q_gain.to(dtype=q.dtype)
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    _prep_qk(q, q_out, attn, past_kv, gain=qg)
    _prep_qk(k, k_out, attn, past_kv, gain=None)
    past_kv.ensure(k_out)
    assert past_kv.k is not None and past_kv.v is not None
    past_kv.k[:, -past_kv.win :].copy_(k_out)
    past_kv.v[:, -past_kv.win :].copy_(v)
    y = _static_window_attention(q_out, past_kv)
    y = _post_attn(y, v, x, attn)
    return F.linear(y.reshape(bsz, seqlen, dim), out_w)


def _qkv_weight(model: "GPT", block_idx: int) -> torch.Tensor | None:
    bank = getattr(model, "_ac_qkv_bank", None)
    if bank is None:
        return None
    return bank[block_idx]


def _block_scales(block, dtype: torch.dtype):
    mix = getattr(block, "_ac_resid_mix", None)
    if mix is None or mix.dtype != dtype:
        mix = block.resid_mix.to(dtype=dtype)
    a_sc = getattr(block, "_ac_attn_scale", None)
    if a_sc is None or a_sc.dtype != dtype:
        a_sc = block.attn_scale.to(dtype=dtype)
    m_sc = getattr(block, "_ac_mlp_scale", None)
    if m_sc is None or m_sc.dtype != dtype:
        m_sc = block.mlp_scale.to(dtype=dtype)
    return mix, a_sc, m_sc


def fused_block_forward(
    model: "GPT",
    block_idx: int,
    x: torch.Tensor,
    x0: torch.Tensor,
    q_w: torch.Tensor,
    k_w: torch.Tensor,
    v_w: torch.Tensor,
    out_w: torch.Tensor,
    up_w: torch.Tensor,
    down_w: torch.Tensor,
    past_kv: StaticKV,
) -> torch.Tensor:
    block = model.blocks[block_idx]
    mix, a_sc, m_sc = _block_scales(block, x.dtype)
    x_in, attn_in = _prologue(x, x0, mix, block.ln_scale_factor)
    attn_out = _fused_attention(
        block.attn,
        attn_in,
        q_w,
        k_w,
        v_w,
        out_w,
        past_kv,
        qkv_w=_qkv_weight(model, block_idx),
    )
    x_out, mlp_in = _mid(x_in, attn_out, a_sc, block.ln_scale_factor)
    return _resid(x_out, _cublas_mlp(mlp_in, up_w, down_w), m_sc)


def fused_parallel_block(
    model: "GPT",
    block_idx: int,
    lane0: torch.Tensor,
    lane1: torch.Tensor,
    x0: torch.Tensor,
    q_w: torch.Tensor,
    k_w: torch.Tensor,
    v_w: torch.Tensor,
    out_w: torch.Tensor,
    up_w: torch.Tensor,
    down_w: torch.Tensor,
    past_kv: StaticKV,
) -> tuple[torch.Tensor, torch.Tensor]:
    block = model.blocks[block_idx]
    mix, a_sc, m_sc = _block_scales(block, lane0.dtype)
    _attn_read, attn_in = _prologue(lane0, x0, mix, block.ln_scale_factor)
    attn_out = _fused_attention(
        block.attn,
        attn_in,
        q_w,
        k_w,
        v_w,
        out_w,
        past_kv,
        qkv_w=_qkv_weight(model, block_idx),
    )
    attn_out = _scale(attn_out, a_sc)
    mlp_in = _rmsnorm_ln(lane1, block.ln_scale_factor)
    mlp_out = _scale(_cublas_mlp(mlp_in, up_w, down_w), m_sc)
    pr = getattr(model, "_ac_parallel_resid", None)
    pp = getattr(model, "_ac_parallel_post", None)
    if pr is None or pr.dtype != lane0.dtype:
        pr = model.parallel_resid_lambdas.to(dtype=lane0.dtype)
    if pp is None or pp.dtype != lane0.dtype:
        pp = model.parallel_post_lambdas.to(dtype=lane0.dtype)
    return _parallel_mix(
        lane0, lane1, attn_out, mlp_out, pr[block_idx], pp[block_idx]
    )
