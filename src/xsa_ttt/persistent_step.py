"""W=1 persistent AC stack (CUDA-graph captured).

Default path is ``mega_step`` (QKV + mega's own thin attn + shared tiled
GEMV FFN when ``XSA_AC_ENCODE_FFN=rows``).
This module's ``_attn_heads`` is the persist-control scan only — speed
experiments must not change it. If mega cannot run, the cuBLAS itinerary
here is the fallback. Encode and decode must both use this path.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from .model import _mark_static_address

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

_RMS_EPS = 1e-6
_NORM_EPS = 1e-12
_KIND_ENC = 0
_KIND_DEC = 1
_KIND_PAR0 = 2
_KIND_PAR = 3


def _attn_block_l() -> int:
    """Prefix tile length for ``_attn_heads``. Power of two, multiple of 16."""
    try:
        n = int(os.environ.get("XSA_AC_ATTN_BLOCK", "256"))
    except ValueError:
        n = 256
    p = 16
    while p < n:
        p *= 2
    return min(max(p, 16), 1024)


def _attn_use_dot() -> bool:
    return os.environ.get("XSA_AC_ATTN_DOT", "1").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def _env_on(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def can_persistent(model: "GPT", state: "StaticState") -> bool:
    if not _HAS_TRITON or state.token.device.type != "cuda":
        return False
    if not _env_on("XSA_AC_PERSISTENT", "1"):
        return False
    if int(state.win) != 1:
        return False
    if getattr(model, "_ac_qkv_bank", None) is None:
        return False
    if any(sk.draft_pool > 0 or sk.draft_local > 0 for sk in state.slots):
        return False
    return True


def _next_pow2(n: int) -> int:
    p = 1
    while p < int(n):
        p *= 2
    return max(p, 1)


if _HAS_TRITON:

    @triton.jit
    def _load_d(PTR, D, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        return tl.load(PTR + offs, mask=offs < D, other=0.0).to(tl.float32)

    @triton.jit
    def _store_d(PTR, vec, D, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        tl.store(PTR + offs, vec.to(PTR.dtype.element_ty), mask=offs < D)

    @triton.jit
    def _rms_vec(vec, D, eps):
        return vec * tl.rsqrt(tl.sum(vec * vec, axis=0) / D + eps)

    @triton.jit
    def _tanh(x):
        # Triton has no tl.tanh; 2*sigmoid(2x)-1 matches torch.tanh in fp32.
        return 2.0 / (1.0 + tl.exp(-2.0 * x)) - 1.0

    @triton.jit
    def _attn_heads(
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

    _LAUNCH_KERN = _attn_heads
else:
    _LAUNCH_KERN = None


@dataclass
class _PersistWS:
    x: torch.Tensor
    x0: torch.Tensor
    xin: torch.Tensor
    attn_in: torch.Tensor
    scr: torch.Tensor
    q: torch.Tensor
    kbuf: torch.Tensor
    vbuf: torch.Tensor
    y: torch.Tensor
    hid: torch.Tensor
    logit_buf: torch.Tensor
    lane0: torch.Tensor
    lane1: torch.Tensor
    skips: torch.Tensor
    k_pack: torch.Tensor  # [n_slots, n_kv, max_len, hd]
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
    rope_half: int
    gate_n: int
    n_slots: int
    max_len: int
    d: int
    hd: int
    n_heads: int
    n_kv: int
    vocab: int
    soft_pos: float
    soft_neg: float
    smear_lam: float
    final_mode: int
    asym_cap: bool


def _itinerary(model: "GPT") -> tuple[torch.Tensor, int, bool]:
    enc, dec = model._iter_indices()
    n_enc = len(enc)
    n_skip = int(model.num_skip_weights)
    psl = int(model.parallel_start_layer)
    rows: list[list[int]] = []
    par_seen = False
    has_par = False
    for layer in enc:
        blk = model.blocks[layer]
        rows.append(
            [
                int(layer),
                _KIND_ENC,
                -1,
                len(rows),
                -1,
                int(blk.attn.use_xsa),
                int(blk.attn.sparse_attn_gate),
            ]
        )
    for skip_idx, layer in enumerate(dec):
        blk = model.blocks[layer]
        src = (n_enc - 1 - skip_idx) if skip_idx < n_skip else -1
        if layer >= psl and psl > 0:
            kind = _KIND_PAR0 if not par_seen else _KIND_PAR
            par_seen = True
            has_par = True
        else:
            kind = _KIND_DEC
        rows.append(
            [
                int(layer),
                kind,
                src,
                -1,
                skip_idx if src >= 0 else -1,
                int(blk.attn.use_xsa),
                int(blk.attn.sparse_attn_gate),
            ]
        )
    itin = torch.tensor(rows, dtype=torch.int32)
    return itin, n_enc, has_par


def _stack_weights(model: "GPT", device: torch.device, dtype: torch.dtype) -> dict:
    blocks = list(model.blocks)
    mix = torch.stack(
        [
            (
                b._ac_resid_mix
                if getattr(b, "_ac_resid_mix", None) is not None
                else b.resid_mix.to(dtype)
            )
            for b in blocks
        ]
    ).to(device=device, dtype=dtype)
    a_sc = torch.stack(
        [
            (
                b._ac_attn_scale
                if getattr(b, "_ac_attn_scale", None) is not None
                else b.attn_scale.to(dtype)
            )
            for b in blocks
        ]
    ).to(device=device, dtype=dtype)
    m_sc = torch.stack(
        [
            (
                b._ac_mlp_scale
                if getattr(b, "_ac_mlp_scale", None) is not None
                else b.mlp_scale.to(dtype)
            )
            for b in blocks
        ]
    ).to(device=device, dtype=dtype)
    q_gain = torch.stack(
        [
            (
                b.attn._ac_q_gain
                if getattr(b.attn, "_ac_q_gain", None) is not None
                else b.attn.q_gain.to(dtype)
            )
            for b in blocks
        ]
    ).to(device=device, dtype=dtype)
    gate_n = int(blocks[0].attn.gate_window)
    n_heads = int(blocks[0].attn.num_heads)
    gates = []
    for b in blocks:
        if b.attn.sparse_attn_gate:
            gw = getattr(b.attn, "_ac_gate_w", None)
            if gw is None:
                gw = b.attn.attn_gate_w.to(dtype)
            gates.append(gw * float(b.attn.sparse_attn_gate_scale))
        else:
            gates.append(torch.zeros(n_heads, gate_n, dtype=dtype, device=device))
    gate_w = torch.stack(gates).to(device=device, dtype=dtype)
    ln = torch.tensor(
        [float(b.ln_scale_factor) for b in blocks], dtype=torch.float32, device=device
    )
    pr = (
        getattr(model, "_ac_parallel_resid", None)
        if getattr(model, "_ac_parallel_resid", None) is not None
        else model.parallel_resid_lambdas.to(dtype)
    ).to(device=device, dtype=dtype).contiguous()
    pp = (
        getattr(model, "_ac_parallel_post", None)
        if getattr(model, "_ac_parallel_post", None) is not None
        else model.parallel_post_lambdas.to(dtype)
    ).to(device=device, dtype=dtype).contiguous().view(len(blocks), 4)
    rot = blocks[0].attn.rotary
    cos = getattr(rot, "_ac_cos", None)
    sin = getattr(rot, "_ac_sin", None)
    if cos is None or sin is None:
        raise RuntimeError("persistent step needs warmed RoPE tables")
    rh = int(cos.shape[-1])
    cos = cos.reshape(-1, rh).contiguous()
    sin = sin.reshape(-1, rh).contiguous()
    sw = getattr(model, "_ac_skip_weights", None)
    if sw is None:
        sw = model.skip_weights.to(dtype)
    sg = getattr(model, "_ac_skip_gates", None)
    if model.skip_gates is not None and sg is None:
        sg = model.skip_gates.to(dtype)
    if sg is None:
        sg = torch.zeros_like(sw)
    smear_w = getattr(model, "_ac_smear_gate_w", None)
    if smear_w is None and model.smear_gate_enabled:
        smear_w = model.smear_gate.weight.to(dtype)
    if smear_w is None:
        smear_w = torch.zeros(1, gate_n, dtype=dtype, device=device)
    return {
        "mix": mix.contiguous(),
        "a_sc": a_sc.contiguous(),
        "m_sc": m_sc.contiguous(),
        "q_gain": q_gain.contiguous(),
        "gate_w": gate_w.contiguous(),
        "ln": ln,
        "pr": pr,
        "pp": pp,
        "cos": cos,
        "sin": sin,
        "skip_w": sw.contiguous(),
        "skip_g": sg.contiguous(),
        "smear_w": smear_w.reshape(-1).contiguous(),
        "rope_half": rh,
        "gate_n": gate_n,
    }


def _launch_meta(
    model: "GPT",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float, float, int]:
    """Host-side launch bindings. Never call this during CUDA graph capture."""
    tok_emb = getattr(model, "_ac_tok_emb", None)
    if tok_emb is None:
        tok_emb = model.tok_emb.weight
    if model.tie_embeddings:
        lm_w = tok_emb
    else:
        lm_w = getattr(model, "_ac_lm_head", None)
        if lm_w is None and model.lm_head is not None:
            lm_w = model.lm_head.weight
        if lm_w is None:
            raise RuntimeError("persistent step needs an LM head")
    out_w = model._ac_qo_bank[model.num_layers :]
    soft_pos = float(model.logit_softcap)
    soft_neg = soft_pos
    if model.asym_logit_enabled:
        psp = getattr(model, "_ac_softcap_pos", None)
        psn = getattr(model, "_ac_softcap_neg", None)
        soft_pos = float((psp if psp is not None else model.softcap_pos).item())
        soft_neg = float((psn if psn is not None else model.softcap_neg).item())
    smear_lam = 0.0
    if model.smear_gate_enabled:
        sl = getattr(model, "_ac_smear_lambda", None)
        smear_lam = float((sl if sl is not None else model.smear_lambda).item())
    final_mode = (
        2
        if model.parallel_final_lane == "mlp"
        else 1
        if model.parallel_final_lane == "attn"
        else 0
    )
    return tok_emb, lm_w, out_w, soft_pos, soft_neg, smear_lam, final_mode


def prepare_persistent(model: "GPT", state: "StaticState") -> _PersistWS | None:
    if not can_persistent(model, state):
        return None
    device = state.token.device
    dtype = getattr(model, "_ac_compute_dtype", None) or state.prev_raw.dtype
    d = int(model.cfg.model_dim)
    hd = int(model.blocks[0].attn.head_dim)
    n_kv = int(model.blocks[0].attn.num_kv_heads)
    n_slots = len(state.slots)
    max_len = int(state.max_len)
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
            k_pack[i].copy_(sk.k.reshape(max_len, n_kv, hd).permute(1, 0, 2).contiguous())
            v_pack[i].copy_(sk.v.reshape(max_len, n_kv, hd).permute(1, 0, 2).contiguous())
            sk.k = k_pack[i].permute(1, 0, 2).unsqueeze(0)
            sk.v = v_pack[i].permute(1, 0, 2).unsqueeze(0)
            _mark_static_address(sk.k)
            _mark_static_address(sk.v)

    def _z(*shape, dt=dtype):
        t = torch.zeros(*shape, device=device, dtype=dt)
        _mark_static_address(t)
        return t

    n_heads = int(model.blocks[0].attn.num_heads)
    tok_emb, lm_w, out_w, soft_pos, soft_neg, smear_lam, final_mode = _launch_meta(
        model
    )
    ws = _PersistWS(
        x=_z(d),
        x0=_z(d),
        xin=_z(d),
        attn_in=_z(d),
        scr=_z(max(qkv_dim, mlp_dim)),
        q=_z(n_heads * hd),
        kbuf=_z(n_kv * hd),
        vbuf=_z(n_kv * hd),
        y=_z(d),
        hid=_z(d),
        logit_buf=_z(int(model.cfg.vocab_size)),
        lane0=_z(d),
        lane1=_z(d),
        skips=_z(max(n_enc, 1), d),
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
        rope_half=packed["rope_half"],
        gate_n=packed["gate_n"],
        n_slots=n_slots,
        max_len=max_len,
        d=d,
        hd=hd,
        n_heads=n_heads,
        n_kv=n_kv,
        vocab=int(model.cfg.vocab_size),
        soft_pos=soft_pos,
        soft_neg=soft_neg,
        smear_lam=smear_lam,
        final_mode=final_mode,
        asym_cap=bool(model.asym_logit_enabled),
    )
    _mark_static_address(ws.k_pack)
    _mark_static_address(ws.v_pack)
    _mark_static_address(ws.itin)
    _mark_static_address(ws.out_w)
    _mark_static_address(ws.tok_emb)
    _mark_static_address(ws.lm_w)
    ws.itin_host = [tuple(int(v) for v in row) for row in itin.tolist()]
    from .ac_gemv import can_gemv_rows, gemv_vec, want_tiled_ffn

    if (
        want_tiled_ffn()
        and can_gemv_rows()
        and model._ac_mlp_up is not None
        and model._ac_mlp_down is not None
    ):
        try:
            gemv_vec(ws.y, ws.out_w[0], ws.hid)
            gemv_vec(ws.attn_in, model._ac_mlp_up[0], ws.scr[: ws.mlp_dim])
            gemv_vec(ws.scr[: ws.mlp_dim], model._ac_mlp_down[0], ws.hid)
            gemv_vec(ws.x, ws.lm_w, ws.logit_buf)
        except Exception as exc:
            ws._persist_gemv_failed = True  # type: ignore[attr-defined]
            cause = exc.args[-1] if exc.args else exc
            print(
                f"[AC incr] persist tiled GEMV unavailable ({cause!r}); "
                "using cuBLAS GEMV",
                file=sys.stderr,
                flush=True,
            )
        else:
            ws.hid.zero_()
            ws.scr.zero_()
            ws.logit_buf.zero_()
    if _env_on("XSA_AC_MEGA", "1"):
        from .mega_step import attach_mega_workspace

        attach_mega_workspace(ws, device)
    else:
        ws.use_mega = False
        ws.n_cta = 0
    return ws


def refresh_persistent_weights(model: "GPT") -> None:
    ws: _PersistWS | None = getattr(model, "_persist_ws", None)
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
    ws.soft_pos = soft_pos
    ws.soft_neg = soft_neg
    ws.smear_lam = smear_lam
    ws.final_mode = final_mode
    ws.asym_cap = bool(model.asym_logit_enabled)


def _dump_layer(ws: _PersistWS, slot: int, field: int, src: torch.Tensor) -> None:
    buf = getattr(ws, "dump_d", None)
    if buf is None:
        return
    buf[slot, field].copy_(src.reshape(-1)[: buf.shape[-1]])


def _rms(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.float().pow(2).mean() + eps).to(dtype=x.dtype)


def persist_embed(state: "StaticState", ws: "_PersistWS") -> None:
    """Token embed + smear + RMS into ``ws.x`` / ``ws.x0`` / ``cand_prev_raw``."""
    raw = F.embedding(state.token.view(-1), ws.tok_emb)[0]
    state.cand_prev_raw.view(-1).copy_(raw)
    if ws.has_smear:
        g = ws.smear_lam * torch.sigmoid(
            (raw[: ws.gate_n].float() * ws.smear_w[: ws.gate_n].float()).sum()
        )
        x = _rms(raw + g.to(ws.x.dtype) * state.prev_raw.view(-1), _RMS_EPS)
    else:
        x = _rms(raw, _RMS_EPS)
    ws.x.copy_(x)
    ws.x0.copy_(x)


def persist_attn_slot(state: "StaticState", ws: "_PersistWS", slot: int) -> None:
    """Persist ``_attn_heads`` over Q + k/v_pack. Current token is the tail."""
    layer, _kind, _ss, _sd, _sw, use_xsa, use_gate = ws.itin_host[slot]
    tail = ws.max_len - 1
    ws.vbuf.copy_(ws.v_pack[slot, :, tail].reshape(-1))
    ws.kbuf.copy_(ws.k_pack[slot, :, tail].reshape(-1))
    _attn_heads[(ws.n_heads,)](
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
        BLOCK_H=_next_pow2(ws.hd),
        BLOCK_L=128,
        BLOCK_G=_next_pow2(max(ws.gate_n, 1)),
        num_warps=4,
    )


class _Win1FFN:
    """1D persist buffers as ``[1, *]`` so decode runs the same FFN as encode."""

    __slots__ = (
        "x",
        "xin",
        "attn_in",
        "y",
        "hid",
        "scr",
        "lane0",
        "lane1",
        "skips",
        "ln",
        "out_w",
        "a_sc",
        "m_sc",
        "pr",
        "pp",
        "itin_host",
        "lm_w",
        "logit_buf",
        "has_par",
        "final_mode",
        "asym_cap",
        "soft_pos",
        "soft_neg",
    )

    def __init__(self, ws: "_PersistWS"):
        mlp = int(ws.mlp_dim)
        self.x = ws.x.view(1, -1)
        self.xin = ws.xin.view(1, -1)
        self.attn_in = ws.attn_in.view(1, -1)
        self.y = ws.y.view(1, -1)
        self.hid = ws.hid.view(1, -1)
        self.scr = ws.scr[:mlp].view(1, -1)
        self.lane0 = ws.lane0.view(1, -1)
        self.lane1 = ws.lane1.view(1, -1)
        self.skips = ws.skips.view(ws.skips.shape[0], 1, -1)
        self.ln = ws.ln
        self.out_w = ws.out_w
        self.a_sc = ws.a_sc
        self.m_sc = ws.m_sc
        self.pr = ws.pr
        self.pp = ws.pp
        self.itin_host = ws.itin_host
        self.lm_w = ws.lm_w
        buf = getattr(ws, "logit_buf", None)
        self.logit_buf = None if buf is None else buf.view(1, -1)
        self.has_par = ws.has_par
        self.final_mode = ws.final_mode
        self.asym_cap = ws.asym_cap
        self.soft_pos = ws.soft_pos
        self.soft_neg = ws.soft_neg


def ffn_slot_rows(model: "GPT", ws: object, slot: int) -> None:
    """Out-proj + MLP for ``ws.*`` shaped ``[win, *]``. Encode and decode share this."""
    from .ac_gemv import (
        ffn_stage_rows,
        gemv_rows,
        rms_rows_match,
        want_stage_fusion,
    )

    layer, kind, _skip_src, skip_dst, _skip_wi, _use_xsa, _use_gate = ws.itin_host[
        slot
    ]
    ln = ws.ln[layer].to(ws.x.dtype)
    par = kind == 2 or kind == 3
    up_bank = model._ac_mlp_up
    down_bank = model._ac_mlp_down
    win = int(ws.x.shape[0])
    if want_stage_fusion():
        ffn_stage_rows(
            y=ws.y,
            out_w=ws.out_w[layer],
            hid=ws.hid,
            a_sc=ws.a_sc[layer],
            xin=ws.xin,
            # Stage 1 still reads every column of Y while output CTAs run.
            # Keep scaled parallel attention in distinct scratch to avoid
            # an in-place cross-CTA read/write race.
            attn_s=ws.attn_in,
            src=ws.lane1 if par else ws.xin,
            up_w=up_bank[layer],
            ln=ln,
            scr=ws.scr,
            down_w=down_bank[layer],
            m_sc=ws.m_sc[layer],
            x=ws.x,
            lane0=ws.lane0,
            lane1=ws.lane1,
            pr=ws.pr[layer],
            pp=ws.pp[layer],
            skips=ws.skips,
            skip_dst=skip_dst,
            par=par,
            eps=_RMS_EPS,
        )
        return
    gemv_rows(ws.y, ws.out_w[layer], ws.hid)
    if par:
        ws.y.copy_(ws.hid * ws.a_sc[layer])
        src = ws.lane1
    else:
        ws.xin.copy_(ws.xin + ws.a_sc[layer] * ws.hid)
        src = ws.xin
    try:
        rms_rows_match(src, ws.attn_in, ln, _RMS_EPS)
    except Exception:
        for r in range(win):
            ws.attn_in[r].copy_(_rms(src[r], _RMS_EPS) * ln)
    gemv_rows(ws.attn_in, up_bank[layer], ws.scr)
    ws.scr.copy_(torch.where(ws.scr >= 0, ws.scr, ws.scr * 0.5).square())
    gemv_rows(ws.scr, down_bank[layer], ws.hid)
    if par:
        attn_s = ws.y
        mlp_s = ws.hid * ws.m_sc[layer]
        pr = ws.pr[layer]
        pp = ws.pp[layer]
        ws.lane0.copy_(pr[0] * ws.lane0 + pp[0] * attn_s + pp[2] * mlp_s)
        ws.lane1.copy_(pr[1] * ws.lane1 + pp[1] * attn_s + pp[3] * mlp_s)
    else:
        ws.x.copy_(ws.xin + ws.m_sc[layer] * ws.hid)
        if skip_dst >= 0:
            ws.skips[skip_dst].copy_(ws.x)


def logits_rows(state: "StaticState", ws: object) -> None:
    """Final RMS + LM head + softcap. ``ws.hid`` / ``ws.logit_buf`` are ``[win, *]``."""
    from .ac_gemv import (
        gemv_rows,
        logits_stage_rows,
        rms_rows_match,
        want_stage_fusion,
    )

    if ws.has_par:
        if ws.final_mode == 1:
            src = ws.lane0
        elif ws.final_mode == 2:
            src = ws.lane1
        else:
            src = 0.5 * (ws.lane0 + ws.lane1)
    else:
        src = ws.x
    win = int(src.shape[0])
    if want_stage_fusion():
        logits_stage_rows(
            src,
            ws.lm_w,
            ws.logit_buf,
            eps=_RMS_EPS,
        )
        # Keep the exact reference tanh implementation; the exp identity can
        # differ by one BF16 value and move an integer AC frequency.
        logits = ws.logit_buf
        if ws.asym_cap:
            logits = torch.where(
                logits >= 0,
                ws.soft_pos * torch.tanh(logits / ws.soft_pos),
                ws.soft_neg * torch.tanh(logits / ws.soft_neg),
            )
        else:
            logits = ws.soft_pos * torch.tanh(logits / ws.soft_pos)
        state.logits.view(win, -1).copy_(logits.float())
        return
    try:
        rms_rows_match(src, ws.hid, None, _RMS_EPS)
    except Exception:
        for r in range(win):
            ws.hid[r].copy_(_rms(src[r], _RMS_EPS))
    gemv_rows(ws.hid, ws.lm_w, ws.logit_buf)
    logits = ws.logit_buf
    if ws.asym_cap:
        logits = torch.where(
            logits >= 0,
            ws.soft_pos * torch.tanh(logits / ws.soft_pos),
            ws.soft_neg * torch.tanh(logits / ws.soft_neg),
        )
    else:
        logits = ws.soft_pos * torch.tanh(logits / ws.soft_pos)
    state.logits.view(logits.shape[0], -1).copy_(logits.float())


def persist_ffn_slot(model: "GPT", ws: "_PersistWS", slot: int) -> None:
    """Out-proj + leaky-square MLP + residual. ``ws.y`` must be complete."""
    from .ac_gemv import can_gemv_rows, want_tiled_ffn

    if (
        want_tiled_ffn()
        and can_gemv_rows()
        and hasattr(ws, "mlp_dim")
        and hasattr(ws, "scr")
        and hasattr(ws, "hid")
        and not getattr(ws, "_persist_gemv_failed", False)
    ):
        try:
            ffn_slot_rows(model, _Win1FFN(ws), slot)
            layer, kind, _skip_src, _sd, _sw, _ux, _ug = ws.itin_host[slot]
            if kind == 2 or kind == 3:
                _dump_layer(ws, slot, 2, ws.lane0)
                _dump_layer(ws, slot, 3, ws.lane1)
            else:
                _dump_layer(ws, slot, 2, ws.x)
            return
        except Exception as exc:
            ws._persist_gemv_failed = True  # type: ignore[attr-defined]
            if not getattr(ws, "_persist_gemv_logged", False):
                ws._persist_gemv_logged = True  # type: ignore[attr-defined]
                cause = exc.args[-1] if exc.args else exc
                print(
                    f"[AC incr] persist tiled GEMV unavailable ({cause!r}); "
                    "using cuBLAS GEMV",
                    file=sys.stderr,
                    flush=True,
                )
    layer, kind, _skip_src, skip_dst, _skip_wi, _use_xsa, _use_gate = ws.itin_host[
        slot
    ]
    ln = ws.ln[layer].to(ws.x.dtype)
    par = kind == 2 or kind == 3
    up_bank = model._ac_mlp_up
    down_bank = model._ac_mlp_down
    attn_out = F.linear(ws.y, ws.out_w[layer])
    _dump_layer(ws, slot, 4, attn_out)
    if par:
        ws.y.copy_(attn_out * ws.a_sc[layer])
        ws.attn_in.copy_(_rms(ws.lane1, _RMS_EPS) * ln)
    else:
        xout = ws.xin + ws.a_sc[layer] * attn_out
        ws.xin.copy_(xout)
        ws.attn_in.copy_(_rms(xout, _RMS_EPS) * ln)
    hid = F.linear(ws.attn_in, up_bank[layer])
    hid = torch.where(hid >= 0, hid, hid * 0.5).square()
    mlp = F.linear(hid, down_bank[layer])
    if par:
        attn_s = ws.y
        mlp_s = mlp * ws.m_sc[layer]
        pr = ws.pr[layer]
        pp = ws.pp[layer]
        ws.lane0.copy_(pr[0] * ws.lane0 + pp[0] * attn_s + pp[2] * mlp_s)
        ws.lane1.copy_(pr[1] * ws.lane1 + pp[1] * attn_s + pp[3] * mlp_s)
        _dump_layer(ws, slot, 2, ws.lane0)
        _dump_layer(ws, slot, 3, ws.lane1)
    else:
        xout = ws.xin + ws.m_sc[layer] * mlp
        ws.x.copy_(xout)
        if skip_dst >= 0:
            ws.skips[skip_dst].copy_(xout)
        _dump_layer(ws, slot, 2, ws.x)


def persist_logits(state: "StaticState", ws: "_PersistWS") -> None:
    """Final RMS + LM head + softcap into ``state.logits``."""
    from .ac_gemv import can_gemv_rows, want_tiled_ffn

    if (
        want_tiled_ffn()
        and can_gemv_rows()
        and getattr(ws, "logit_buf", None) is not None
        and not getattr(ws, "_persist_gemv_failed", False)
    ):
        try:
            logits_rows(state, _Win1FFN(ws))
            return
        except Exception as exc:
            ws._persist_gemv_failed = True  # type: ignore[attr-defined]
            if not getattr(ws, "_persist_gemv_logged", False):
                ws._persist_gemv_logged = True  # type: ignore[attr-defined]
                cause = exc.args[-1] if exc.args else exc
                print(
                    f"[AC incr] persist tiled GEMV unavailable ({cause!r}); "
                    "using cuBLAS GEMV",
                    file=sys.stderr,
                    flush=True,
                )
    if ws.has_par:
        if ws.final_mode == 1:
            x = ws.lane0
        elif ws.final_mode == 2:
            x = ws.lane1
        else:
            x = 0.5 * (ws.lane0 + ws.lane1)
    else:
        x = ws.x
    x = _rms(x, _RMS_EPS)
    ws.x.copy_(x)
    logits = F.linear(x, ws.lm_w)
    if ws.asym_cap:
        logits = torch.where(
            logits >= 0,
            ws.soft_pos * torch.tanh(logits / ws.soft_pos),
            ws.soft_neg * torch.tanh(logits / ws.soft_neg),
        )
    else:
        logits = ws.soft_pos * torch.tanh(logits / ws.soft_pos)
    state.logits.view(-1).copy_(logits.float())


def _rope(vec: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rh: int) -> torch.Tensor:
    x1 = vec[..., :rh]
    x2 = vec[..., rh : 2 * rh]
    rot = torch.cat((x1 * cos + x2 * sin, -x1 * sin + x2 * cos), dim=-1)
    if vec.shape[-1] > 2 * rh:
        return torch.cat((rot, vec[..., 2 * rh :]), dim=-1)
    return rot


def _persist_launch(model: "GPT", state: "StaticState", ws: _PersistWS) -> None:
    """Full W=1 stack. Megakernel first; cuBLAS itinerary if it cannot run."""
    if getattr(ws, "use_mega", False) and not getattr(ws, "_mega_failed", False):
        from .mega_step import try_mega_extend

        if try_mega_extend(model, state, ws):
            return
        ws._mega_failed = True
        print(
            "[AC incr] megakernel failed "
            f"({getattr(ws, '_mega_error', 'unknown')}); "
            "using cuBLAS persist itinerary",
            file=sys.stderr,
            flush=True,
        )
    _persist_launch_python(model, state, ws)


def _persist_launch_python(model: "GPT", state: "StaticState", ws: _PersistWS) -> None:
    """cuBLAS GEMMs + per-head Triton attn. Graph-safe fallback."""
    d = ws.d
    hd = ws.hd
    n_heads = ws.n_heads
    n_kv = ws.n_kv
    rh = ws.rope_half
    tail = ws.max_len - 1
    dtype = ws.x.dtype
    persist_embed(state, ws)
    qkv_bank = model._ac_qkv_bank
    for slot, row in enumerate(ws.itin_host):
        layer, kind, skip_src, skip_dst, skip_wi, use_xsa, use_gate = row
        ln = ws.ln[layer].to(dtype)
        par = kind == 2 or kind == 3
        x = ws.x
        if kind == 2:
            ws.lane0.copy_(x)
            ws.lane1.copy_(x)
        if par:
            x = ws.lane0
        if skip_src >= 0:
            scaled = ws.skip_w[skip_wi] * ws.skips[skip_src]
            if ws.has_skip_gate:
                gg = torch.sigmoid(ws.skip_g[skip_wi].float()).to(dtype)
                x = scaled * (1.0 - gg) + x * gg
            else:
                x = x + scaled
        if par:
            ws.lane0.copy_(x)
            x = ws.lane0
        else:
            ws.x.copy_(x)
            x = ws.x
        xin = ws.mix[layer, 0] * x + ws.mix[layer, 1] * ws.x0
        attn_in = _rms(xin, _RMS_EPS) * ln
        ws.xin.copy_(xin)
        ws.attn_in.copy_(attn_in)
        _dump_layer(ws, slot, 0, ws.attn_in)
        ws.scr[: ws.qkv_dim].copy_(F.linear(attn_in, qkv_bank[layer]))
        cos = ws.cos.index_select(0, state.pos.view(-1)).view(rh)
        sin = ws.sin.index_select(0, state.pos.view(-1)).view(rh)
        kv_dim = n_kv * hd
        qraw = ws.scr[:d].view(n_heads, hd).float()
        qn = (
            qraw
            * torch.rsqrt(qraw.pow(2).mean(-1, keepdim=True) + _RMS_EPS)
            * ws.q_gain[layer].float().view(n_heads, 1)
        ).to(dtype)
        ws.q.copy_(_rope(qn, cos, sin, rh).reshape(-1))
        kraw = ws.scr[d : d + kv_dim].view(n_kv, hd).float()
        vraw = ws.scr[d + kv_dim : d + 2 * kv_dim].view(n_kv, hd)
        kn = (
            kraw * torch.rsqrt(kraw.pow(2).mean(-1, keepdim=True) + _RMS_EPS)
        ).to(dtype)
        kn = _rope(kn, cos, sin, rh)
        ws.kbuf.copy_(kn.reshape(-1))
        ws.vbuf.copy_(vraw.reshape(-1))
        ws.k_pack[slot, :, tail].copy_(kn)
        ws.v_pack[slot, :, tail].copy_(vraw)
        dump_q = getattr(ws, "dump_q", None)
        if dump_q is not None:
            dump_q[slot].copy_(ws.q)
        persist_attn_slot(state, ws, slot)
        _dump_layer(ws, slot, 1, ws.y)
        persist_ffn_slot(model, ws, slot)
    persist_logits(state, ws)


def try_persistent_extend(model: "GPT", state: "StaticState") -> bool:
    """Run the W=1 persistent kernel into ``state``. False → use fused slots."""
    if not can_persistent(model, state) or getattr(model, "_persist_disabled", False):
        return False
    ws: _PersistWS | None = getattr(model, "_persist_ws", None)
    if (
        ws is None
        or ws.k_pack.shape[0] != len(state.slots)
        or ws.k_pack.dim() != 4
        or ws.k_pack.shape[2] != state.max_len
        or not hasattr(ws, "out_w")
        or not hasattr(ws, "itin_host")
    ):
        try:
            ws = prepare_persistent(model, state)
        except Exception:
            return False
        if ws is None:
            return False
        model._persist_ws = ws
    if not _HAS_TRITON:
        return False
    try:
        _persist_launch(model, state, ws)
    except Exception as exc:
        model._persist_disabled = True
        cause = (
            exc.args[-1]
            if type(exc).__name__ == "CompilationError" and exc.args
            else exc
        )
        print(
            f"[AC incr] persistent stack kernel failed ({cause!r}); "
            "using per-slot fused steps",
            file=sys.stderr,
            flush=True,
        )
        return False
    return True
