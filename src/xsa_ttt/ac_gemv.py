"""Shared per-row GEMV for encode (W) and decode (W=1).

Independent rows — not an M=W GEMM. Encode and decode must call this
same kernel so integer frequencies match.
"""

from __future__ import annotations

import os

import torch

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
    def _gemv_row_tiles(
        X,
        W,
        Y,
        n_out,
        k,
        win,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
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
        tl.store(gvr_yp + gvr_no, gvr_acc.to(Y.dtype.element_ty), mask=gvr_nm)

    @triton.jit
    def _rms_rows_1d(
        X,
        Y,
        LN,
        D,
        eps,
        win,
        use_ln,
        BLOCK: tl.constexpr,
    ):
        """Per-row RMS matching ``_rms``: ``x * rsqrt(mean(x.f32²)+eps).to(x.dtype)``."""
        row = tl.program_id(0)
        if row >= win:
            return
        offs = tl.arange(0, BLOCK)
        mask = offs < D
        xb = tl.load(X + row * D + offs, mask=mask, other=0.0)
        xf = xb.to(tl.float32)
        ss = tl.sum(xf * xf, axis=0) / D
        sc = tl.rsqrt(ss + eps).to(xb.dtype)
        yb = xb * sc
        if use_ln != 0:
            # ln_scale_factor is one scalar per layer, not a D-vector.
            # Loading LN+offs walks off the [n_layers] table and explodes MLP.
            yb = yb * tl.load(LN)
        tl.store(Y + row * D + offs, yb, mask=mask)

    @triton.jit
    def _ffn_out_residual_rows(
        Y,
        W,
        HID,
        A_SC,
        XIN,
        ATTN_S,
        D,
        win,
        par,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Out projection followed by its rounded residual update."""
        row = tl.program_id(0)
        tile = tl.program_id(1)
        no = tile * BLOCK_N + tl.arange(0, BLOCK_N)
        nm = no < D
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        kk = 0
        while kk < D:
            ko = kk + tl.arange(0, BLOCK_K)
            km = ko < D
            xv = tl.load(
                Y + row * D + ko, mask=(row < win) & km, other=0.0
            ).to(tl.float32)
            wv = tl.load(
                W + no[:, None] * D + ko[None, :],
                mask=nm[:, None] & km[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(wv * xv[None, :], axis=1)
            kk += BLOCK_K
        attn = acc.to(HID.dtype.element_ty)
        tl.store(
            HID + row * D + no, attn, mask=(row < win) & nm
        )
        sc = tl.load(A_SC + no, mask=nm, other=0.0)
        scaled = (attn * sc).to(HID.dtype.element_ty)
        if par != 0:
            tl.store(
                ATTN_S + row * D + no,
                scaled.to(ATTN_S.dtype.element_ty),
                mask=(row < win) & nm,
            )
        else:
            old = tl.load(
                XIN + row * D + no,
                mask=(row < win) & nm,
                other=0.0,
            )
            updated = (old + scaled).to(XIN.dtype.element_ty)
            tl.store(
                XIN + row * D + no,
                updated,
                mask=(row < win) & nm,
            )

    @triton.jit
    def _ffn_up_activation_rows(
        SRC,
        W,
        LN,
        SCR,
        D,
        MLP,
        eps,
        win,
        BLOCK_D: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """RMS + up projection + rounded leaky-square activation."""
        row = tl.program_id(0)
        tile = tl.program_id(1)
        do = tl.arange(0, BLOCK_D)
        dm = do < D
        src = tl.load(
            SRC + row * D + do, mask=(row < win) & dm, other=0.0
        )
        srcf = src.to(tl.float32)
        scale = tl.rsqrt(tl.sum(srcf * srcf, axis=0) / D + eps).to(
            src.dtype
        )
        no = tile * BLOCK_N + tl.arange(0, BLOCK_N)
        nm = no < MLP
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        kk = 0
        while kk < D:
            ko = kk + tl.arange(0, 128)
            km = ko < D
            sv = tl.load(
                SRC + row * D + ko,
                mask=(row < win) & km,
                other=0.0,
            )
            # Scalar ln_scale_factor (same as persist ``_rms(x) * ln``).
            norm = ((sv * scale) * tl.load(LN)).to(W.dtype.element_ty)
            wv = tl.load(
                W + no[:, None] * D + ko[None, :],
                mask=nm[:, None] & km[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(wv * norm.to(tl.float32)[None, :], axis=1)
            kk += 128
        raw = acc.to(SCR.dtype.element_ty)
        neg = (raw * 0.5).to(SCR.dtype.element_ty)
        leaky = tl.where(raw >= 0, raw, neg)
        act = (leaky * leaky).to(SCR.dtype.element_ty)
        tl.store(
            SCR + row * MLP + no, act, mask=(row < win) & nm
        )

    @triton.jit
    def _ffn_down_commit_rows(
        SCR,
        W,
        HID,
        M_SC,
        XIN,
        X,
        ATTN_S,
        LANE0,
        LANE1,
        PR,
        PP,
        SKIPS,
        skip_dst,
        D,
        MLP,
        win,
        par,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Down projection followed by rounded residual/lane commit."""
        row = tl.program_id(0)
        tile = tl.program_id(1)
        no = tile * BLOCK_N + tl.arange(0, BLOCK_N)
        nm = no < D
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        kk = 0
        while kk < MLP:
            ko = kk + tl.arange(0, BLOCK_K)
            km = ko < MLP
            xv = tl.load(
                SCR + row * MLP + ko,
                mask=(row < win) & km,
                other=0.0,
            ).to(tl.float32)
            wv = tl.load(
                W + no[:, None] * MLP + ko[None, :],
                mask=nm[:, None] & km[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(wv * xv[None, :], axis=1)
            kk += BLOCK_K
        mlp = acc.to(HID.dtype.element_ty)
        tl.store(
            HID + row * D + no, mlp, mask=(row < win) & nm
        )
        msc = tl.load(M_SC + no, mask=nm, other=0.0)
        mlp_s = (mlp * msc).to(HID.dtype.element_ty)
        if par != 0:
            l0 = tl.load(
                LANE0 + row * D + no,
                mask=(row < win) & nm,
                other=0.0,
            )
            l1 = tl.load(
                LANE1 + row * D + no,
                mask=(row < win) & nm,
                other=0.0,
            )
            attn = tl.load(
                ATTN_S + row * D + no,
                mask=(row < win) & nm,
                other=0.0,
            )
            pr0 = tl.load(PR + 0)
            pr1 = tl.load(PR + 1)
            pp0 = tl.load(PP + 0)
            pp1 = tl.load(PP + 1)
            pp2 = tl.load(PP + 2)
            pp3 = tl.load(PP + 3)
            n0 = ((pr0 * l0).to(LANE0.dtype.element_ty) +
                  (pp0 * attn).to(LANE0.dtype.element_ty)).to(
                      LANE0.dtype.element_ty
                  )
            n0 = (n0 + (pp2 * mlp_s).to(LANE0.dtype.element_ty)).to(
                LANE0.dtype.element_ty
            )
            n1 = ((pr1 * l1).to(LANE1.dtype.element_ty) +
                  (pp1 * attn).to(LANE1.dtype.element_ty)).to(
                      LANE1.dtype.element_ty
                  )
            n1 = (n1 + (pp3 * mlp_s).to(LANE1.dtype.element_ty)).to(
                LANE1.dtype.element_ty
            )
            tl.store(
                LANE0 + row * D + no, n0, mask=(row < win) & nm
            )
            tl.store(
                LANE1 + row * D + no, n1, mask=(row < win) & nm
            )
        else:
            xin = tl.load(
                XIN + row * D + no,
                mask=(row < win) & nm,
                other=0.0,
            )
            updated = (
                xin + mlp_s.to(XIN.dtype.element_ty)
            ).to(X.dtype.element_ty)
            tl.store(
                X + row * D + no, updated, mask=(row < win) & nm
            )
            if skip_dst >= 0:
                tl.store(
                    SKIPS + (skip_dst * win + row) * D + no,
                    updated.to(SKIPS.dtype.element_ty),
                    mask=(row < win) & nm,
                )

    @triton.jit
    def _logits_fused_rows(
        SRC,
        W,
        LOGIT_BUF,
        D,
        V,
        eps,
        win,
        BLOCK_D: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Final RMS + LM projection with reference output rounding."""
        row = tl.program_id(0)
        tile = tl.program_id(1)
        do = tl.arange(0, BLOCK_D)
        dm = do < D
        src = tl.load(
            SRC + row * D + do, mask=(row < win) & dm, other=0.0
        )
        srcf = src.to(tl.float32)
        scale = tl.rsqrt(tl.sum(srcf * srcf, axis=0) / D + eps).to(
            src.dtype
        )
        no = tile * BLOCK_N + tl.arange(0, BLOCK_N)
        nm = no < V
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        kk = 0
        while kk < D:
            ko = kk + tl.arange(0, 128)
            km = ko < D
            sv = tl.load(
                SRC + row * D + ko,
                mask=(row < win) & km,
                other=0.0,
            )
            norm = (sv * scale).to(W.dtype.element_ty)
            wv = tl.load(
                W + no[:, None] * D + ko[None, :],
                mask=nm[:, None] & km[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(wv * norm.to(tl.float32)[None, :], axis=1)
            kk += 128
        raw = acc.to(LOGIT_BUF.dtype.element_ty)
        tl.store(
            LOGIT_BUF + row * V + no, raw, mask=(row < win) & nm
        )

else:  # pragma: no cover
    _gemv_row_tiles = None
    _rms_rows_1d = None
    _ffn_out_residual_rows = None
    _ffn_up_activation_rows = None
    _ffn_down_commit_rows = None
    _logits_fused_rows = None


def can_gemv_rows() -> bool:
    return _gemv_row_tiles is not None


def want_tiled_ffn() -> bool:
    """Tiled GEMV on both encode and decode (default). ``loop`` keeps cuBLAS."""
    raw = os.environ.get("XSA_AC_ENCODE_FFN", "rows").strip().lower()
    return raw not in {"loop", "persist", "1d", "cublas", "bmm", "batched"}


def want_stage_fusion() -> bool:
    """Use dependency-ordered FFN/logit stages shared by encode and decode."""
    return os.environ.get("XSA_AC_STAGE_FUSION", "1").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
        "",
    }


def gemv_rows(x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor) -> None:
    """``out[r] = weight @ x[r]``. ``x`` is ``[win, K]``, ``out`` is ``[win, N]``."""
    if _gemv_row_tiles is None:
        raise RuntimeError("tiled GEMV requires Triton")
    win = int(x.shape[0])
    k = int(weight.shape[1])
    n_out = int(weight.shape[0])
    if int(x.shape[1]) != k or tuple(out.shape) != (win, n_out):
        raise ValueError(
            f"gemv_rows shape mismatch x={tuple(x.shape)} w={tuple(weight.shape)} "
            f"out={tuple(out.shape)}"
        )
    block_n = 32 if n_out > 512 else 16
    n_tiles = (n_out + block_n - 1) // block_n
    _gemv_row_tiles[(win, n_tiles)](
        x,
        weight,
        out,
        n_out,
        k,
        win,
        BLOCK_N=block_n,
        BLOCK_K=128,
        num_warps=4,
    )


def gemv_vec(x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor) -> None:
    """1D ``out = weight @ x`` via the same kernel as ``gemv_rows`` (win=1)."""
    gemv_rows(x.view(1, -1), weight, out.view(1, -1))


def rms_rows_match(
    x: torch.Tensor,
    out: torch.Tensor,
    ln: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> None:
    """``out[r] = _rms(x[r])`` [``* ln``]. One launch; each row is independent."""
    if _rms_rows_1d is None:
        raise RuntimeError("rms_rows_match requires Triton")
    win, d = int(x.shape[0]), int(x.shape[1])
    if tuple(out.shape) != (win, d):
        raise ValueError(f"rms shape mismatch x={tuple(x.shape)} out={tuple(out.shape)}")
    if ln is not None and int(ln.numel()) != 1:
        raise ValueError(
            f"ln must be the per-layer scalar ln_scale_factor, got {ln.numel()} values"
        )
    block = 1
    while block < d:
        block *= 2
    _rms_rows_1d[(win,)](
        x,
        out,
        x if ln is None else ln,
        d,
        float(eps),
        win,
        0 if ln is None else 1,
        BLOCK=block,
        num_warps=4,
    )


def ffn_stage_rows(
    *,
    y: torch.Tensor,
    out_w: torch.Tensor,
    hid: torch.Tensor,
    a_sc: torch.Tensor,
    xin: torch.Tensor,
    attn_s: torch.Tensor,
    src: torch.Tensor,
    up_w: torch.Tensor,
    ln: torch.Tensor,
    scr: torch.Tensor,
    down_w: torch.Tensor,
    m_sc: torch.Tensor,
    x: torch.Tensor,
    lane0: torch.Tensor,
    lane1: torch.Tensor,
    pr: torch.Tensor,
    pp: torch.Tensor,
    skips: torch.Tensor,
    skip_dst: int,
    par: bool,
    eps: float = 1e-6,
) -> None:
    """Three stream-ordered kernels for out → up/act → down/commit."""
    if (
        _ffn_out_residual_rows is None
        or _ffn_up_activation_rows is None
        or _ffn_down_commit_rows is None
    ):
        raise RuntimeError("stage-fused FFN requires Triton")
    win, d = int(y.shape[0]), int(y.shape[1])
    mlp = int(scr.shape[1])
    ln_s = ln.reshape(-1).contiguous()
    if int(ln_s.numel()) != 1:
        raise ValueError(
            f"ln must be the per-layer scalar ln_scale_factor, got {ln.numel()} values"
        )
    if tuple(hid.shape) != (win, d) or tuple(xin.shape) != (win, d):
        raise ValueError("stage-fused FFN hidden shape mismatch")
    block_d = 1
    while block_d < d:
        block_d *= 2
    block_n = 16
    _ffn_out_residual_rows[(win, (d + block_n - 1) // block_n)](
        y,
        out_w,
        hid,
        a_sc,
        xin,
        attn_s,
        d,
        win,
        1 if par else 0,
        BLOCK_N=block_n,
        BLOCK_K=128,
        num_warps=4,
    )
    up_block_n = 32
    _ffn_up_activation_rows[
        (win, (mlp + up_block_n - 1) // up_block_n)
    ](
        src,
        up_w,
        ln_s,
        scr,
        d,
        mlp,
        float(eps),
        win,
        BLOCK_D=block_d,
        BLOCK_N=up_block_n,
        num_warps=4,
    )
    _ffn_down_commit_rows[(win, (d + block_n - 1) // block_n)](
        scr,
        down_w,
        hid,
        m_sc,
        xin,
        x,
        attn_s,
        lane0,
        lane1,
        pr,
        pp,
        skips,
        int(skip_dst),
        d,
        mlp,
        win,
        1 if par else 0,
        BLOCK_N=block_n,
        BLOCK_K=128,
        num_warps=4,
    )


def logits_stage_rows(
    src: torch.Tensor,
    weight: torch.Tensor,
    logit_buf: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> None:
    """One dependency-safe kernel for final RMS → LM projection."""
    if _logits_fused_rows is None:
        raise RuntimeError("stage-fused logits require Triton")
    win, d = int(src.shape[0]), int(src.shape[1])
    vocab = int(weight.shape[0])
    if tuple(logit_buf.shape) != (win, vocab):
        raise ValueError("stage-fused logit buffer shape mismatch")
    block_d = 1
    while block_d < d:
        block_d *= 2
    block_n = 16
    _logits_fused_rows[(win, (vocab + block_n - 1) // block_n)](
        src,
        weight,
        logit_buf,
        d,
        vocab,
        float(eps),
        win,
        BLOCK_D=block_d,
        BLOCK_N=block_n,
        num_warps=4,
    )
