"""large GPT stack (XSA-all + SmearGate + SparseAttnGate + loops).

Long causal train/probe/prefill uses a Triton tile (``train_attn.py``) with
the same mask + fp32 softmax as ``_causal_attn_chunk``. W=1 decode stays on
the mega/persist kernels.
"""

from __future__ import annotations

import math
import os
from functools import partial
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

if TYPE_CHECKING:
    from .config import XsaTttConfig


def _mark_static_address(t: torch.Tensor) -> None:
    """Tell Dynamo this tensor's data_ptr is stable across compiled steps.

    CUDAGraph trees otherwise treat eager inputs as ephemeral, copy them into
    a staging buffer, and skip graphs when ``static_extend`` ``copy_``s into
    ``cand_prev_raw`` / ``logits`` / KV tails (36 mutations on this stack).
    """
    if t.device.type != "cuda":
        return
    mark = getattr(torch._dynamo, "mark_static_address", None)
    if mark is None:
        deco = getattr(torch._dynamo, "decorators", None)
        mark = getattr(deco, "mark_static_address", None) if deco is not None else None
    if mark is not None:
        mark(t)


def _cudagraph_mark_step_begin() -> None:
    fn = getattr(getattr(torch, "compiler", None), "cudagraph_mark_step_begin", None)
    if fn is not None:
        fn()


def _enable_compile_tf32() -> None:
    """Use TF32 tensor cores for compiled float32 matmuls.

    ``COMPRESSION_DETERMINISTIC=strict`` sets precision to ``highest`` (and
    ``allow_tf32=False``), which makes Inductor warn and skip tensor cores.
    Encode and decode both call this before ``torch.compile``, so lockstep
    is preserved; the bitstream is not interchangeable with a no-TF32 run.
    """
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:  # noqa: BLE001
        pass


def _as_x_dtype(weight: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Cast only when dtypes differ (AC step pre-casts banks to skip this)."""
    if weight.dtype == x.dtype:
        return weight
    return weight.to(x.dtype)


def _adopt_ac_view(
    owner: object, name: str, src: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """Keep a compute-dtype snapshot; recast in-place after replenish."""
    cast = src.detach().to(dtype=dtype)
    if cast.data_ptr() == src.data_ptr():
        cast = cast.clone()
    cur = getattr(owner, name, None)
    if (
        isinstance(cur, torch.Tensor)
        and cur.shape == cast.shape
        and cur.dtype == cast.dtype
        and cur.device == cast.device
    ):
        cur.copy_(cast)
        return cur
    setattr(owner, name, cast)
    _mark_static_address(cast)
    return cast


def _projection_linear(
    module: nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    role: str,
) -> torch.Tensor:
    """F.linear with optional NanoQuant projection-moment instrumentation."""
    y = F.linear(x, _as_x_dtype(weight, x))
    collector = getattr(module, "_projection_moment_collector", None)
    layer = int(getattr(module, "_projection_moment_layer", -1))
    if collector is not None and layer >= 0:
        collector.accumulate_input(layer, role, x)
        if y.requires_grad:
            y.register_hook(
                lambda grad, li=layer, r=role, c=collector: c.accumulate_output_grad(
                    li, r, grad
                )
            )
    return y


class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)


class Rotary(nn.Module):
    def __init__(
        self,
        dim: int,
        base: float = 1e4,
        train_seq_len: int = 2048,
        rope_dims: int = 0,
        yarn: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.base = base
        self.train_seq_len = train_seq_len
        self.yarn = yarn
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        inv_freq = 1.0 / base ** (
            torch.arange(0, self.rope_dims, 2, dtype=torch.float32) / self.rope_dims
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: torch.Tensor | None = None
        self._sin_cached: torch.Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached < seq_len
            or self._cos_cached.device != device
        ):
            rd = self.rope_dims
            if self.yarn and seq_len > self.train_seq_len:
                scale = seq_len / self.train_seq_len
                new_base = self.base * scale ** (rd / (rd - 2))
                inv_freq = 1.0 / new_base ** (
                    torch.arange(0, rd, 2, dtype=torch.float32, device=device) / rd
                )
            else:
                inv_freq = self.inv_freq.float().to(device)
            t = torch.arange(seq_len, device=device, dtype=torch.float32)
            freqs = torch.outer(t, inv_freq)
            # clone(): avoid inference_mode tensors sticking in the cache and
            # poisoning later autograd / no_grad forwards.
            self._cos_cached = freqs.cos()[None, :, None, :].clone()
            self._sin_cached = freqs.sin()[None, :, None, :].clone()
            self._seq_len_cached = seq_len
        assert self._cos_cached is not None and self._sin_cached is not None
        ac_cos = getattr(self, "_ac_cos", None)
        ac_sin = getattr(self, "_ac_sin", None)
        if (
            ac_cos is not None
            and ac_sin is not None
            and ac_cos.dtype == dtype
            and ac_cos.device == device
            and ac_cos.shape[1] >= seq_len
        ):
            return ac_cos[:, :seq_len], ac_sin[:, :seq_len]
        if self._cos_cached.dtype == dtype:
            return self._cos_cached[:, :seq_len], self._sin_cached[:, :seq_len]
        return (
            self._cos_cached[:, :seq_len].to(dtype=dtype),
            self._sin_cached[:, :seq_len].to(dtype=dtype),
        )


def apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rope_dims: int = 0
) -> torch.Tensor:
    if rope_dims > 0 and rope_dims < x.size(-1):
        x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]
        half = rope_dims // 2
        x1, x2 = x_rope[..., :half], x_rope[..., half:]
        x_rope = torch.cat((x1 * cos + x2 * sin, x1 * -sin + x2 * cos), dim=-1)
        return torch.cat((x_rope, x_pass), dim=-1)
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * -sin + x2 * cos), dim=-1)


class KVSlot:
    """Preallocated KV buffer for one attention slot (incremental AC path).

    Avoids the per-step ``torch.cat`` realloc (quadratic memory traffic over a
    16k-extend segment). Values are identical to the cat path — only storage
    differs — so encode/decode numerics are unaffected as long as both sides
    use the same path.
    """

    __slots__ = ("max_len", "k", "v", "length")

    def __init__(self, max_len: int):
        self.max_len = int(max_len)
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None
        self.length = 0

    def reset(self) -> None:
        self.length = 0

    def append(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Write new K/V rows in place; return views + was-empty flag."""
        if (
            self.k is None
            or self.k.dtype != k.dtype
            or self.k.device != k.device
            or self.k.shape[0] != k.shape[0]
            or self.k.shape[2:] != k.shape[2:]
        ):
            b, _, h, d = k.shape
            self.k = torch.empty(
                b, self.max_len, h, d, dtype=k.dtype, device=k.device
            )
            self.v = torch.empty_like(self.k)
            self.length = 0
        was = self.length
        new = was + k.size(1)
        if new > self.max_len:
            raise RuntimeError(
                f"KVSlot overflow: {new} > max_len {self.max_len}"
            )
        assert self.v is not None
        self.k[:, was:new] = k
        self.v[:, was:new] = v
        self.length = new
        return self.k[:, :new], self.v[:, :new], was == 0


class StaticKV:
    """Fixed-size KV buffer for one attention slot (CUDA-graph-capturable).

    Unlike :class:`KVSlot` (dynamic length views), every W-token window step
    reads the *full* ``max_len`` buffer under a validity mask derived from the
    shared device-resident ``pos`` counter, so shapes never change and the
    step can be captured once and replayed. Prefill (``step_mode`` False)
    writes rows ``[0, T)`` eagerly and is never captured.
    """

    __slots__ = (
        "max_len",
        "win",
        "pos",
        "arange",
        "arange_w",
        "is_tail",
        "tail_local",
        "tail_vis",
        "step_mode",
        "draft_local",
        "draft_pool",
        "draft_window",
        "draft_start",
        "draft_end",
        "k",
        "v",
    )

    def __init__(
        self,
        max_len: int,
        pos: torch.Tensor,
        arange: torch.Tensor,
        arange_w: torch.Tensor,
        is_tail: torch.Tensor,
        tail_local: torch.Tensor,
    ):
        self.max_len = int(max_len)
        self.win = int(arange_w.numel())
        self.pos = pos  # (1,) int64, shared across slots, owned by StaticState
        self.arange = arange  # (max_len,) int64, shared
        self.arange_w = arange_w  # (win,) int64, shared
        # Window K/V always live at the last W rows (constant addresses —
        # graph-safe copy_, no scatter/index_put). Prefix is [0, pos).
        self.is_tail = is_tail  # (max_len,) bool
        self.tail_local = tail_local  # (max_len,) int64
        # (W, L) tail visibility — static (does not depend on pos).
        self.tail_vis = is_tail.unsqueeze(0) & (
            tail_local.unsqueeze(0) <= arange_w.unsqueeze(1)
        )
        self.step_mode = False  # True during the capturable window step
        # Decode-only draft. ``draft_pool>0``: HCA-style mean-pool every
        # ``draft_pool`` prefix keys + last ``draft_window`` raw keys.
        # ``draft_local>0`` and pool==0: last-N slice (legacy). 0/0 = full
        # attention (verify / encode).
        self.draft_local = 0
        self.draft_pool = 0
        self.draft_window = 0
        self.draft_start = 0
        self.draft_end = 0
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None

    def ensure(self, like: torch.Tensor) -> None:
        """Allocate zeroed buffers matching ``like``'s (B, ·, Hkv, D)/dtype.

        Once a ``max_len`` buffer exists (including a view of packed K/V),
        keep it. Prefill may produce a different compute dtype than the AC
        workspace; ``copy_`` casts. Reallocating here detaches the slot from
        ``k_pack`` and leaves encode/decode megakernels on a stale prefix.
        """
        if (
            self.k is not None
            and self.v is not None
            and self.k.shape[0] == like.shape[0]
            and self.k.shape[1] == self.max_len
            and self.k.shape[2:] == like.shape[2:]
            and self.k.device == like.device
        ):
            return
        if (
            self.k is None
            or self.k.dtype != like.dtype
            or self.k.device != like.device
            or self.k.shape[0] != like.shape[0]
            or self.k.shape[2:] != like.shape[2:]
        ):
            b, _, h, d = like.shape
            # zeros (not empty): masked-out rows must stay finite so the
            # softmax mask alone excludes them.
            self.k = torch.zeros(
                b, self.max_len, h, d, dtype=like.dtype, device=like.device
            )
            self.v = torch.zeros_like(self.k)
            _mark_static_address(self.k)
            _mark_static_address(self.v)


class StaticState:
    """All device buffers for the capturable W-token window AC step."""

    __slots__ = (
        "token",
        "prev_raw",
        "cand_prev_raw",
        "pos",
        "arange",
        "arange_w",
        "logits",
        "slots",
        "max_len",
        "win",
    )

    def __init__(
        self,
        *,
        token: torch.Tensor,
        prev_raw: torch.Tensor,
        cand_prev_raw: torch.Tensor,
        pos: torch.Tensor,
        arange: torch.Tensor,
        arange_w: torch.Tensor,
        logits: torch.Tensor,
        slots: "list[StaticKV]",
        max_len: int,
        win: int,
    ):
        self.token = token
        self.prev_raw = prev_raw
        self.cand_prev_raw = cand_prev_raw
        self.pos = pos
        self.arange = arange
        self.arange_w = arange_w
        self.logits = logits
        self.slots = slots
        self.max_len = int(max_len)
        self.win = int(win)
        self.mark_static_addresses()

    def mark_static_addresses(self) -> None:
        """Pin every long-lived AC buffer for inductor CUDA graphs."""
        for t in (
            self.token,
            self.prev_raw,
            self.cand_prev_raw,
            self.pos,
            self.arange,
            self.arange_w,
            self.logits,
        ):
            _mark_static_address(t)
        for sk in self.slots:
            _mark_static_address(sk.pos)
            _mark_static_address(sk.arange)
            _mark_static_address(sk.arange_w)
            _mark_static_address(sk.is_tail)
            _mark_static_address(sk.tail_local)
            if sk.k is not None:
                _mark_static_address(sk.k)
            if sk.v is not None:
                _mark_static_address(sk.v)

    def set_step_mode(self, on: bool) -> None:
        for sk in self.slots:
            sk.step_mode = bool(on)

    def set_draft_range(
        self,
        local: int = 0,
        start: int = 0,
        end: int = 0,
        *,
        pool: int = 0,
        window: int = 0,
    ) -> None:
        """Host-side draft window (eager only; never captured)."""
        local_i = int(local)
        start_i = int(start)
        end_i = int(end)
        pool_i = int(pool)
        window_i = int(window)
        for sk in self.slots:
            sk.draft_local = local_i
            sk.draft_pool = pool_i
            sk.draft_window = window_i
            sk.draft_start = start_i
            sk.draft_end = end_i

    def commit_window(self, dest: int) -> None:
        """Copy the tail window into prefix rows ``[dest, dest+W)``.

        Host-side ``dest`` (Python int) — not captured. Speculative re-runs
        only overwrite the tail, so a rejected window never lands in prefix.
        """
        w = self.win
        for sk in self.slots:
            if sk.k is None:
                continue
            assert sk.v is not None
            sk.k[:, dest : dest + w].copy_(sk.k[:, -w:])
            sk.v[:, dest : dest + w].copy_(sk.v[:, -w:])


def _gqa_attend_prefix_and_tail(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    n_pre: int,
    sk: StaticKV,
) -> torch.Tensor:
    """GQA scores of q (1,W,H,D) against cat(prefix, tail) K/V (1, n_pre+W, …)."""
    _, n_win, n_heads, head_dim = q.shape
    n_kv = k.shape[2]
    group = n_heads // n_kv
    n_k = k.shape[1]
    qg = q.reshape(n_win, n_kv, group, head_dim).permute(1, 2, 0, 3)
    k_t = k.reshape(n_k, n_kv, head_dim).permute(1, 2, 0).unsqueeze(1)
    scale = 1.0 / math.sqrt(head_dim)
    att = torch.matmul(qg, k_t.to(dtype=qg.dtype)) * scale
    tail_vis = sk.arange_w[None, :] <= sk.arange_w[:, None]
    if n_pre > 0:
        pre_vis = torch.ones(n_win, n_pre, dtype=torch.bool, device=q.device)
        vis = torch.cat([pre_vis, tail_vis], dim=1)
    else:
        vis = tail_vis
    att.masked_fill_(~vis[None, None, :, :], float("-inf"))
    att = torch.softmax(att, dim=-1)
    v_t = v.reshape(n_k, n_kv, head_dim).permute(1, 0, 2).unsqueeze(1)
    y = torch.matmul(att, v_t.to(dtype=att.dtype))
    return y.permute(2, 0, 1, 3).reshape(1, n_win, n_heads, head_dim)


def _mean_pool_kv(t: torch.Tensor, n_blocks: int, pool: int) -> torch.Tensor:
    """(1, n_blocks*pool, H, D) → (1, n_blocks, H, D) mean over each block."""
    return t.reshape(1, n_blocks, pool, t.shape[2], t.shape[3]).mean(dim=2)


def _static_window_attention_hca(q: torch.Tensor, sk: StaticKV) -> torch.Tensor:
    """Decode draft: HCA-style pooled prefix + raw local window + causal tail.

    Mean-pools every ``draft_pool`` keys in ``[0, pos - window)``, keeps
    ``[pos - window, pos)`` uncompressed, and scores only that short list
    (plus the W-token tail). Covers the full prefix without a 32k matmul.
    Untrained stand-in for DeepSeek-V4 HCA (no learned compressor).
    """
    assert sk.k is not None and sk.v is not None
    pos = int(sk.draft_end)
    pool = max(1, int(sk.draft_pool))
    win_u = max(0, int(sk.draft_window))
    k_tail = sk.k[:, -sk.win :]
    v_tail = sk.v[:, -sk.win :]
    parts_k: list[torch.Tensor] = []
    parts_v: list[torch.Tensor] = []
    n_pre = 0
    if pos > 0:
        w_start = max(0, pos - win_u) if win_u > 0 else pos
        n_blocks = w_start // pool
        if n_blocks > 0:
            sl = n_blocks * pool
            parts_k.append(_mean_pool_kv(sk.k[:, :sl], n_blocks, pool))
            parts_v.append(_mean_pool_kv(sk.v[:, :sl], n_blocks, pool))
            n_pre += n_blocks
        raw0 = n_blocks * pool
        if raw0 < pos:
            parts_k.append(sk.k[:, raw0:pos])
            parts_v.append(sk.v[:, raw0:pos])
            n_pre += pos - raw0
    parts_k.append(k_tail)
    parts_v.append(v_tail)
    return _gqa_attend_prefix_and_tail(
        q, torch.cat(parts_k, dim=1), torch.cat(parts_v, dim=1), n_pre, sk
    )


def _static_window_attention_local(q: torch.Tensor, sk: StaticKV) -> torch.Tensor:
    """Decode draft: attend only to prefix ``[draft_start, draft_end)`` + tail.

    Legacy last-N slice (``XSA_AC_DRAFT_POOL=0``). Eager-only.
    """
    assert sk.k is not None and sk.v is not None
    start = int(sk.draft_start)
    end = int(sk.draft_end)
    k_pre = sk.k[:, start:end]
    v_pre = sk.v[:, start:end]
    k = torch.cat([k_pre, sk.k[:, -sk.win :]], dim=1)
    v = torch.cat([v_pre, sk.v[:, -sk.win :]], dim=1)
    return _gqa_attend_prefix_and_tail(q, k, v, k_pre.shape[1], sk)


def _static_window_attention(q: torch.Tensor, sk: StaticKV) -> torch.Tensor:
    """W-token attention: prefix ``[0, pos)`` plus causal tail window.

    Manual grouped-query attention: q (1,W,H,D) against k/v (1,max_len,Hkv,D)
    without materializing the GQA expansion. Query row i sees committed
    prefix keys (arange < pos) and tail keys with local index <= i. Each
    output row is bitwise independent of later window tokens — the property
    speculative decode relies on. Softmax stays in the working dtype (the
    fp32 cast cloned a W×L matrix per layer and dominated the W=64 profile).
    """
    if sk.draft_pool > 0:
        return _static_window_attention_hca(q, sk)
    if sk.draft_local > 0:
        return _static_window_attention_local(q, sk)
    assert sk.k is not None and sk.v is not None
    _, n_win, n_heads, head_dim = q.shape
    n_kv = sk.k.shape[2]
    group = n_heads // n_kv
    # (Hkv, group, W, D) x (Hkv, 1, D, L) -> (Hkv, group, W, L)
    qg = q.reshape(n_win, n_kv, group, head_dim).permute(1, 2, 0, 3)
    k_t = sk.k.reshape(sk.max_len, n_kv, head_dim).permute(1, 2, 0).unsqueeze(1)
    scale = 1.0 / math.sqrt(head_dim)
    if k_t.dtype != qg.dtype:
        k_t = k_t.to(dtype=qg.dtype)
    att = torch.matmul(qg, k_t) * scale
    prefix = sk.arange < sk.pos  # (L,)
    tail = getattr(sk, "tail_vis", None)
    if tail is None:
        tail = sk.is_tail & (sk.tail_local[None, :] <= sk.arange_w[:, None])
    att.masked_fill_(~(prefix | tail)[None, None], float("-inf"))
    att = torch.softmax(att, dim=-1)
    v_t = sk.v.reshape(sk.max_len, n_kv, head_dim).permute(1, 0, 2).unsqueeze(1)
    if v_t.dtype != att.dtype:
        v_t = v_t.to(dtype=att.dtype)
    y = torch.matmul(att, v_t)  # (Hkv, group, W, D)
    return y.permute(2, 0, 1, 3).reshape(1, n_win, n_heads, head_dim)


def _expand_kv(k: torch.Tensor, v: torch.Tensor, n_heads: int) -> tuple[torch.Tensor, torch.Tensor]:
    """k/v: (B, T, Hkv, D) → expand GQA to n_heads."""
    n_kv = k.size(2)
    if n_kv != n_heads:
        rep = n_heads // n_kv
        k = k.repeat_interleave(rep, dim=2)
        v = v.repeat_interleave(rep, dim=2)
    return k, v


def _sdpa_causal(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """q/k/v: (B, T, H, D) → y: (B, T, H, D). Expands GQA KV heads."""
    n_heads = q.size(2)
    k, v = _expand_kv(k, v, n_heads)
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)
    y = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True)
    return y.transpose(1, 2)


def _attn_chunk_size() -> int:
    """Query-chunk size for long causal attention (0 disables chunking).

    Must match between encode and decode (both read the same default/env).
    """
    try:
        return int(os.environ.get("XSA_ATTN_CHUNK", "2048"))
    except ValueError:
        return 2048


def _causal_attn_chunk(
    q_c: torch.Tensor, k_c: torch.Tensor, v_c: torch.Tensor, q0: int
) -> torch.Tensor:
    """Attention for query rows [q0, q0+C) against keys [0, q0+C).

    Explicit fp32 softmax with a causal mask on the diagonal block. Fixed
    shapes per chunk index → deterministic kernels, same on both codec sides.
    """
    head_dim = q_c.size(-1)
    scale = 1.0 / math.sqrt(head_dim)
    att = torch.matmul(q_c, k_c.transpose(-1, -2)) * scale  # (B,H,C,L)
    n_q = q_c.size(2)
    n_k = k_c.size(2)
    row = torch.arange(n_q, device=q_c.device) + q0
    col = torch.arange(n_k, device=q_c.device)
    att = att.masked_fill(
        col[None, None, None, :] > row[None, None, :, None], float("-inf")
    )
    att = torch.softmax(att.float(), dim=-1).to(q_c.dtype)
    return torch.matmul(att, v_c)


def _chunked_causal_sdpa(
    q_t: torch.Tensor, k_t: torch.Tensor, v_t: torch.Tensor, chunk: int
) -> torch.Tensor:
    """Causal attention in query chunks; O(chunk × T) live memory.

    Replaces math-SDPA for long training/probe sequences under strict
    determinism: full math-SDPA materializes the (H, T, T) fp32 attention
    matrix (32 GiB at T=32k) and its backward needs several such buffers —
    the retrain-boundary OOM. CUDA uses a Triton tile (``train_attn.py``)
    with the same causal mask and fp32 softmax; the Python loop remains the
    CPU / fallback path. Each eager chunk is gradient-checkpointed, so the
    fallback backward recomputes one (H, chunk, T) tile at a time.
    """
    from .train_attn import triton_causal_sdpa

    tiled = triton_causal_sdpa(q_t, k_t, v_t)
    if tiled is not None:
        return tiled
    n_q = q_t.size(2)
    outs = []
    for q0 in range(0, n_q, chunk):
        c = min(chunk, n_q - q0)
        q_c = q_t[:, :, q0 : q0 + c]
        k_c = k_t[:, :, : q0 + c]
        v_c = v_t[:, :, : q0 + c]
        if torch.is_grad_enabled() and (
            q_c.requires_grad or k_c.requires_grad or v_c.requires_grad
        ):
            out = checkpoint(
                _causal_attn_chunk, q_c, k_c, v_c, q0, use_reentrant=False
            )
        else:
            out = _causal_attn_chunk(q_c, k_c, v_c, q0)
        outs.append(out)
    return torch.cat(outs, dim=2)


def _sdpa_with_past(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, is_prefill: bool
) -> torch.Tensor:
    """Attention where K/V may be longer than Q (decode step).

    Same full-causal SDPA as training. Prefill (Tq==Tk): causal mask.
    Decode (Tq==1, Tk>=1): attend to all cached past (caller truncates to block).
    Long causal sequences route through the chunked path (bounded memory
    under the strict-mode math backend).
    """
    n_heads = q.size(2)
    k, v = _expand_kv(k, v, n_heads)
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)
    causal = bool(is_prefill and q.size(1) > 1)
    chunk = _attn_chunk_size()
    if causal and chunk > 0 and q.size(1) > chunk:
        y = _chunked_causal_sdpa(q_t, k_t, v_t, chunk)
    else:
        y = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=causal)
    return y.transpose(1, 2)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        train_seq_len: int,
        yarn: bool = False,
        gate_window: int = 12,
        sparse_attn_gate: bool = False,
        sparse_attn_gate_init_std: float = 0.0,
        sparse_attn_gate_scale: float = 1.0,
        attn_out_gate: bool = False,
        gated_attn: bool = False,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if int(attn_out_gate) + int(gated_attn) + int(sparse_attn_gate) > 1:
            raise ValueError("attn gates are mutually exclusive")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.q_gain = nn.Parameter(
            torch.full((num_heads,), qk_gain_init, dtype=torch.float32)
        )
        self.rope_dims = 0
        self.rotary = Rotary(
            self.head_dim, base=rope_base, train_seq_len=train_seq_len, yarn=yarn
        )
        self.use_xsa = False
        self.gate_window = gate_window
        self.attn_out_gate = attn_out_gate
        self.gated_attn = gated_attn
        self.sparse_attn_gate = sparse_attn_gate
        self.sparse_attn_gate_scale = sparse_attn_gate_scale
        if attn_out_gate:
            self.attn_gate_proj = CastedLinear(gate_window, num_heads, bias=False)
            self.attn_gate_proj._zero_init = True  # type: ignore[attr-defined]
        if gated_attn:
            W = torch.empty(num_heads, dim, dtype=torch.float32)
            nn.init.normal_(W, mean=0.0, std=0.01)
            self.attn_gate_w = nn.Parameter(W)
        if sparse_attn_gate:
            W = torch.empty(num_heads, gate_window, dtype=torch.float32)
            if sparse_attn_gate_init_std > 0:
                nn.init.normal_(W, mean=0.0, std=sparse_attn_gate_init_std)
            else:
                nn.init.zeros_(W)
            self.attn_gate_w = nn.Parameter(W)

    def _xsa_efficient(self, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Subtract KV-group value projection from attention output (XSA)."""
        B, T, H, D = y.shape
        Hkv = v.size(-2)
        group = H // Hkv
        y_g = y.reshape(B, T, Hkv, group, D)
        vn = F.normalize(v, dim=-1).unsqueeze(-2)
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return (y_g - proj).reshape(B, T, H, D)

    def forward(
        self,
        x: torch.Tensor,
        q_w: torch.Tensor,
        k_w: torch.Tensor,
        v_w: torch.Tensor,
        out_w: torch.Tensor,
        *,
        past_kv: "tuple[torch.Tensor, torch.Tensor] | KVSlot | None" = None,
        pos_offset: int = 0,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        bsz, seqlen, dim = x.shape
        q_raw = _projection_linear(self, x, q_w, "q")
        q = q_raw.reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = _projection_linear(self, x, k_w, "k").reshape(
            bsz, seqlen, self.num_kv_heads, self.head_dim
        )
        v = _projection_linear(self, x, v_w, "v").reshape(
            bsz, seqlen, self.num_kv_heads, self.head_dim
        )
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        static_step = isinstance(past_kv, StaticKV) and past_kv.step_mode
        if static_step:
            # Capture-safe RoPE: index the warmed table by device-side
            # window positions (no Python-int slicing on per-step values).
            sk: StaticKV = past_kv  # type: ignore[assignment]
            idx = sk.pos + sk.arange_w  # (W,) absolute buffer positions
            cos_tab, sin_tab = self.rotary(sk.max_len, x.device, q.dtype)
            cos = cos_tab.index_select(1, idx)
            sin = sin_tab.index_select(1, idx)
        else:
            total_len = pos_offset + seqlen
            cos_full, sin_full = self.rotary(total_len, x.device, q.dtype)
            cos = cos_full[:, pos_offset:total_len]
            sin = sin_full[:, pos_offset:total_len]
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        qg = getattr(self, "_ac_q_gain", None)
        if qg is None or qg.dtype != q.dtype:
            qg = self.q_gain.to(dtype=q.dtype)
        q = q * qg[None, None, :, None]
        if isinstance(past_kv, StaticKV):
            past_kv.ensure(k)
            assert past_kv.k is not None and past_kv.v is not None
            if static_step:
                # Fixed tail addresses: copy_ is graph-safe and does not
                # trip the deterministic index_put / radix-sort path.
                if k.dtype == past_kv.k.dtype:
                    past_kv.k[:, -past_kv.win :].copy_(k)
                    past_kv.v[:, -past_kv.win :].copy_(v)
                else:
                    past_kv.k[:, -past_kv.win :].copy_(k.to(dtype=past_kv.k.dtype))
                    past_kv.v[:, -past_kv.win :].copy_(v.to(dtype=past_kv.v.dtype))
                y = _static_window_attention(q, past_kv)
            else:
                # Eager prefill at a segment start: rows [0, T), plain causal
                # attention over the fresh tokens (numerics == legacy prefill).
                # Cast into the packed AC buffer — never let dtype mismatch
                # replace slot.k and detach it from k_pack.
                past_kv.k[:, :seqlen].copy_(k.to(dtype=past_kv.k.dtype))
                past_kv.v[:, :seqlen].copy_(v.to(dtype=past_kv.v.dtype))
                y = _sdpa_with_past(q, k, v, is_prefill=True)
        elif isinstance(past_kv, KVSlot):
            k_cat, v_cat, was_empty = past_kv.append(k, v)
            y = _sdpa_with_past(q, k_cat, v_cat, is_prefill=was_empty)
        elif past_kv is not None:
            pk, pv = past_kv
            k_cat = torch.cat([pk, k], dim=1)
            v_cat = torch.cat([pv, v], dim=1)
            y = _sdpa_with_past(q, k_cat, v_cat, is_prefill=False)
        else:
            y = _sdpa_with_past(q, k, v, is_prefill=True)
        # XSA uses the *new* V slice (matches full-forward when T_new==T).
        if self.use_xsa:
            y = self._xsa_efficient(y, v)
        if self.attn_out_gate:
            gate_in = x[..., : self.gate_window].contiguous()
            g = 2.0 * torch.sigmoid(self.attn_gate_proj(gate_in))
            y = y * g[..., None]
        if self.gated_attn:
            g = torch.sigmoid(F.linear(x.contiguous(), self.attn_gate_w.to(x.dtype)))
            y = y * g[..., None]
        if self.sparse_attn_gate:
            gate_in = x[..., : self.gate_window].contiguous()
            gw = getattr(self, "_ac_gate_w", None)
            if gw is None or gw.dtype != x.dtype:
                gw = self.attn_gate_w.to(x.dtype)
            g = torch.sigmoid(
                self.sparse_attn_gate_scale * F.linear(gate_in, gw)
            )
            y = y * g[..., None]
        y = y.reshape(bsz, seqlen, dim)
        out = _projection_linear(self, y, out_w, "o")
        if use_cache:
            if isinstance(past_kv, (KVSlot, StaticKV)):
                return out, past_kv
            k_ret = k if past_kv is None else k_cat
            v_ret = v if past_kv is None else v_cat
            return out, (k_ret, v_ret)
        return out


class MLP(nn.Module):
    def forward(
        self, x: torch.Tensor, up_w: torch.Tensor, down_w: torch.Tensor
    ) -> torch.Tensor:
        # LeakyReLU(0.5)^2 — large-profile motif
        hidden = F.leaky_relu(
            _projection_linear(self, x, up_w, "up"), negative_slope=0.5
        ).square()
        return _projection_linear(self, hidden, down_w, "down")


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: float,
        rope_base: float,
        qk_gain_init: float,
        train_seq_len: int,
        layer_idx: int = 0,
        ln_scale: bool = False,
        yarn: bool = False,
        gate_window: int = 12,
        sparse_attn_gate: bool = False,
        sparse_attn_gate_init_std: float = 0.0,
        sparse_attn_gate_scale: float = 1.0,
        attn_out_gate: bool = False,
        gated_attn: bool = False,
    ):
        super().__init__()
        del mlp_mult
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(
            dim,
            num_heads,
            num_kv_heads,
            rope_base,
            qk_gain_init,
            train_seq_len,
            yarn=yarn,
            gate_window=gate_window,
            sparse_attn_gate=sparse_attn_gate,
            sparse_attn_gate_init_std=sparse_attn_gate_init_std,
            sparse_attn_gate_scale=sparse_attn_gate_scale,
            attn_out_gate=attn_out_gate,
            gated_attn=gated_attn,
        )
        self.mlp = MLP()
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(
            torch.stack((torch.ones(dim), torch.zeros(dim))).float()
        )
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0

    def forward(
        self,
        x: torch.Tensor,
        x0: torch.Tensor,
        q_w: torch.Tensor,
        k_w: torch.Tensor,
        v_w: torch.Tensor,
        out_w: torch.Tensor,
        up_w: torch.Tensor,
        down_w: torch.Tensor,
        *,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        pos_offset: int = 0,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        mix = getattr(self, "_ac_resid_mix", None)
        if mix is None or mix.dtype != x.dtype:
            mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_ret = self.attn(
            self.attn_norm(x_in) * self.ln_scale_factor,
            q_w,
            k_w,
            v_w,
            out_w,
            past_kv=past_kv,
            pos_offset=pos_offset,
            use_cache=use_cache,
        )
        if use_cache:
            attn_out, present = attn_ret  # type: ignore[misc]
        else:
            attn_out = attn_ret  # type: ignore[assignment]
            present = None
        a_sc = getattr(self, "_ac_attn_scale", None)
        if a_sc is None or a_sc.dtype != x_in.dtype:
            a_sc = self.attn_scale.to(dtype=x_in.dtype)
        m_sc = getattr(self, "_ac_mlp_scale", None)
        if m_sc is None or m_sc.dtype != x_in.dtype:
            m_sc = self.mlp_scale.to(dtype=x_in.dtype)
        x_out = x_in + a_sc[None, None, :] * attn_out
        x_out = x_out + m_sc[None, None, :] * self.mlp(
            self.mlp_norm(x_out) * self.ln_scale_factor, up_w, down_w
        )
        if use_cache:
            assert present is not None
            return x_out, present
        return x_out


class GPT(nn.Module):
    """large GPT with weight banks, U-Net skips, parallel residuals, XSA."""

    def __init__(self, h: "XsaTttConfig"):
        super().__init__()
        if h.logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {h.logit_softcap}")
        self.cfg = h
        self.tie_embeddings = h.tie_embeddings
        self.tied_embed_init_std = h.tied_embed_init_std
        self.logit_softcap = h.logit_softcap
        self.asym_logit_enabled = h.asym_logit_rescale
        if self.asym_logit_enabled:
            self.softcap_pos = nn.Parameter(torch.tensor(h.logit_softcap))
            self.softcap_neg = nn.Parameter(torch.tensor(h.logit_softcap))
        self.tok_emb = nn.Embedding(h.vocab_size, h.model_dim)
        self.num_layers = h.num_layers
        head_dim = h.model_dim // h.num_heads
        kv_dim = h.num_kv_heads * head_dim
        hidden_dim = int(h.mlp_mult * h.model_dim)
        self.qo_bank = nn.Parameter(
            torch.empty(2 * h.num_layers, h.model_dim, h.model_dim)
        )
        self.kv_bank = nn.Parameter(torch.empty(2 * h.num_layers, kv_dim, h.model_dim))
        self.mlp_up_bank = nn.Parameter(
            torch.empty(h.num_layers, hidden_dim, h.model_dim)
        )
        self.mlp_down_bank = nn.Parameter(
            torch.empty(h.num_layers, h.model_dim, hidden_dim)
        )
        self.num_encoder_layers = h.num_layers // 2
        self.num_decoder_layers = h.num_layers - self.num_encoder_layers
        self.blocks = nn.ModuleList(
            [
                Block(
                    h.model_dim,
                    h.num_heads,
                    h.num_kv_heads,
                    h.mlp_mult,
                    h.rope_base,
                    h.qk_gain_init,
                    h.block_size,
                    layer_idx=i,
                    ln_scale=h.ln_scale,
                    yarn=h.rope_yarn,
                    gate_window=h.gate_window,
                    sparse_attn_gate=h.sparse_attn_gate_enabled,
                    sparse_attn_gate_init_std=h.sparse_attn_gate_init_std,
                    sparse_attn_gate_scale=h.sparse_attn_gate_scale,
                    attn_out_gate=h.attn_out_gate_enabled,
                    gated_attn=h.gated_attn_enabled,
                )
                for i in range(h.num_layers)
            ]
        )
        for layer_idx, block in enumerate(self.blocks):
            block.attn._projection_moment_layer = layer_idx
            block.mlp._projection_moment_layer = layer_idx
            block.attn._projection_moment_collector = None
            block.mlp._projection_moment_collector = None
        if h.rope_dims > 0:
            for block in self.blocks:
                block.attn.rope_dims = h.rope_dims
                block.attn.rotary = Rotary(
                    head_dim,
                    base=h.rope_base,
                    train_seq_len=h.block_size,
                    rope_dims=h.rope_dims,
                    yarn=h.rope_yarn,
                )
        self.final_norm = RMSNorm()
        self.lm_head = (
            None
            if h.tie_embeddings
            else CastedLinear(h.model_dim, h.vocab_size, bias=False)
        )
        if self.lm_head is not None:
            self.lm_head._zero_init = True  # type: ignore[attr-defined]
        if h.xsa_last_n > 0:
            for i in range(max(0, h.num_layers - h.xsa_last_n), h.num_layers):
                self.blocks[i].attn.use_xsa = True
        self.looping_active = False
        if h.num_loops > 0:
            loop_seg = list(range(h.loop_start, h.loop_end + 1))
            all_indices = list(range(h.loop_start))
            for _ in range(h.num_loops + 1):
                all_indices.extend(loop_seg)
            all_indices.extend(range(h.loop_end + 1, h.num_layers))
            num_enc = len(all_indices) // 2
            self.encoder_indices = all_indices[:num_enc]
            self.decoder_indices = all_indices[num_enc:]
        else:
            self.encoder_indices = list(range(self.num_encoder_layers))
            self.decoder_indices = list(
                range(self.num_encoder_layers, h.num_layers)
            )
        self.num_skip_weights = min(
            len(self.encoder_indices), len(self.decoder_indices)
        )
        self.skip_weights = nn.Parameter(
            torch.ones(self.num_skip_weights, h.model_dim, dtype=torch.float32)
        )
        self.skip_gates = (
            nn.Parameter(
                torch.zeros(self.num_skip_weights, h.model_dim, dtype=torch.float32)
            )
            if h.skip_gates_enabled
            else None
        )
        self.parallel_start_layer = h.parallel_start_layer
        self.parallel_final_lane = h.parallel_final_lane.lower()
        self.parallel_post_lambdas = nn.Parameter(
            torch.ones(h.num_layers, 2, 2, dtype=torch.float32)
        )
        self.parallel_resid_lambdas = nn.Parameter(
            torch.full((h.num_layers, 2), 1.1, dtype=torch.float32)
        )
        self.smear_gate_enabled = h.smear_gate_enabled
        self.smear_gate_bos_fix = h.smear_gate_bos_fix
        if self.smear_gate_enabled:
            self.smear_window = h.gate_window
            self.smear_gate = CastedLinear(self.smear_window, 1, bias=False)
            self.smear_gate._zero_init = True  # type: ignore[attr-defined]
            self.smear_lambda = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.gradient_checkpointing = bool(
            getattr(h, "gradient_checkpointing", True)
        )
        self._init_weights()

    def set_projection_moment_collector(self, collector) -> None:
        """Attach/detach a duck-typed projection moment collector."""
        for block in self.blocks:
            block.attn._projection_moment_collector = collector
            block.mlp._projection_moment_collector = collector

    def _run_block(
        self,
        i: int,
        x: torch.Tensor,
        x0: torch.Tensor,
    ) -> torch.Tensor:
        q_w, k_w, v_w, out_w, up_w, down_w = self._bank_weights(i)
        return self.blocks[i](x, x0, q_w, k_w, v_w, out_w, up_w, down_w)

    def _maybe_checkpoint_block(
        self, i: int, x: torch.Tensor, x0: torch.Tensor
    ) -> torch.Tensor:
        if self.training and self.gradient_checkpointing:
            # partial keeps layer index out of the checkpoint tensor args.
            return checkpoint(
                partial(self._run_block, i),
                x,
                x0,
                use_reentrant=False,
            )
        return self._run_block(i, x, x0)

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        n = self.num_layers
        proj_scale = 1.0 / math.sqrt(2 * n)
        for i in range(n):
            nn.init.orthogonal_(self.qo_bank.data[i], gain=1.0)
            nn.init.zeros_(self.qo_bank.data[n + i])
            self.qo_bank.data[n + i].mul_(proj_scale)
            nn.init.orthogonal_(self.kv_bank.data[i], gain=1.0)
            nn.init.orthogonal_(self.kv_bank.data[n + i], gain=1.0)
        for i in range(n):
            nn.init.orthogonal_(self.mlp_up_bank.data[i], gain=1.0)
            nn.init.zeros_(self.mlp_down_bank.data[i])
            self.mlp_down_bank.data[i].mul_(proj_scale)
        for _name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                elif (
                    module.weight.ndim == 2
                    and module.weight.shape[0] >= 64
                    and module.weight.shape[1] >= 64
                ):
                    nn.init.orthogonal_(module.weight, gain=1.0)

    def _bank_weights(self, i: int):
        n = self.num_layers
        qo = getattr(self, "_ac_qo_bank", None)
        kv = getattr(self, "_ac_kv_bank", None)
        up = getattr(self, "_ac_mlp_up", None)
        down = getattr(self, "_ac_mlp_down", None)
        if qo is None or not getattr(self, "_ac_step_active", False):
            qo, kv = self.qo_bank, self.kv_bank
            up, down = self.mlp_up_bank, self.mlp_down_bank
        return (
            qo[i],
            kv[i],
            kv[n + i],
            qo[n + i],
            up[i],
            down[i],
        )

    def prepare_ac_compute_weights(self, dtype: torch.dtype | None = None) -> torch.dtype:
        """Snapshot banks/scales in the AC compute dtype (no per-step ``.to``).

        Retrain writes fp32 parameters in-place — call again after each
        replenish so the next encode/decode steps see updated weights.
        """
        if dtype is None:
            param = next(self.parameters())
            if bool(self.cfg.use_bf16) and param.device.type == "cuda":
                dtype = torch.bfloat16
            else:
                dtype = param.dtype
        self._ac_compute_dtype = dtype
        _adopt_ac_view(self, "_ac_qo_bank", self.qo_bank, dtype)
        _adopt_ac_view(self, "_ac_kv_bank", self.kv_bank, dtype)
        n = self.num_layers
        qkv = torch.cat(
            (
                self._ac_qo_bank[:n],
                self._ac_kv_bank[:n],
                self._ac_kv_bank[n:],
            ),
            dim=1,
        )
        _adopt_ac_view(self, "_ac_qkv_bank", qkv, dtype)
        _adopt_ac_view(self, "_ac_mlp_up", self.mlp_up_bank, dtype)
        _adopt_ac_view(self, "_ac_mlp_down", self.mlp_down_bank, dtype)
        _adopt_ac_view(self, "_ac_skip_weights", self.skip_weights, dtype)
        if self.skip_gates is not None:
            _adopt_ac_view(self, "_ac_skip_gates", self.skip_gates, dtype)
        _adopt_ac_view(self, "_ac_parallel_post", self.parallel_post_lambdas, dtype)
        _adopt_ac_view(self, "_ac_parallel_resid", self.parallel_resid_lambdas, dtype)
        _adopt_ac_view(self, "_ac_tok_emb", self.tok_emb.weight, dtype)
        if self.smear_gate_enabled:
            _adopt_ac_view(self, "_ac_smear_lambda", self.smear_lambda, dtype)
            _adopt_ac_view(self, "_ac_smear_gate_w", self.smear_gate.weight, dtype)
        if not self.tie_embeddings and self.lm_head is not None:
            _adopt_ac_view(self, "_ac_lm_head", self.lm_head.weight, dtype)
        if self.asym_logit_enabled:
            _adopt_ac_view(self, "_ac_softcap_pos", self.softcap_pos, dtype)
            _adopt_ac_view(self, "_ac_softcap_neg", self.softcap_neg, dtype)
        for block in self.blocks:
            _adopt_ac_view(block, "_ac_resid_mix", block.resid_mix, dtype)
            _adopt_ac_view(block, "_ac_attn_scale", block.attn_scale, dtype)
            _adopt_ac_view(block, "_ac_mlp_scale", block.mlp_scale, dtype)
            _adopt_ac_view(block.attn, "_ac_q_gain", block.attn.q_gain, dtype)
            if getattr(block.attn, "sparse_attn_gate", False):
                _adopt_ac_view(
                    block.attn, "_ac_gate_w", block.attn.attn_gate_w, dtype
                )
            rot = block.attn.rotary
            if rot._cos_cached is not None and rot._sin_cached is not None:
                _adopt_ac_view(rot, "_ac_cos", rot._cos_cached, dtype)
                _adopt_ac_view(rot, "_ac_sin", rot._sin_cached, dtype)
        if getattr(self, "_persist_ws", None) is not None:
            from .persistent_step import refresh_persistent_weights

            refresh_persistent_weights(self)
        if getattr(self, "_encode_ws", None) is not None:
            from .mega_encode import refresh_encode_weights

            refresh_encode_weights(self)
        return dtype

    def _set_ac_step_active(self, on: bool) -> None:
        self._ac_step_active = bool(on)
        for block in self.blocks:
            block._ac_step_active = bool(on)
            block.attn._ac_step_active = bool(on)
            block.mlp._ac_step_active = bool(on)

    def _raw_token_emb(self, input_ids: torch.Tensor) -> torch.Tensor:
        w = getattr(self, "_ac_tok_emb", None)
        if getattr(self, "_ac_step_active", False) and w is not None:
            return F.embedding(input_ids, w)
        return self.tok_emb(input_ids)

    def _smear_gate_values(
        self, raw: torch.Tensor, gate_in: torch.Tensor
    ) -> torch.Tensor:
        sl = getattr(self, "_ac_smear_lambda", None)
        if sl is None or sl.dtype != raw.dtype:
            sl = self.smear_lambda.to(dtype=raw.dtype)
        gw = getattr(self, "_ac_smear_gate_w", None)
        if (
            getattr(self, "_ac_step_active", False)
            and gw is not None
            and gw.dtype == gate_in.dtype
        ):
            return sl * torch.sigmoid(F.linear(gate_in, gw))
        return sl * torch.sigmoid(self.smear_gate(gate_in))

    def _apply_decoder_skip(
        self, skip_idx: int, dest: torch.Tensor, skips: list[torch.Tensor]
    ) -> torch.Tensor:
        if skip_idx >= self.num_skip_weights or not skips:
            return dest
        skip = skips.pop()
        sw = getattr(self, "_ac_skip_weights", None)
        if sw is None or sw.dtype != dest.dtype:
            sw = self.skip_weights.to(dtype=dest.dtype)
        scaled = sw[skip_idx][None, None, :] * skip
        if self.skip_gates is None:
            return dest + scaled
        sg = getattr(self, "_ac_skip_gates", None)
        if sg is None or sg.dtype != dest.dtype:
            sg = self.skip_gates.to(dtype=dest.dtype)
        g = torch.sigmoid(sg[skip_idx])[None, None, :]
        return torch.lerp(scaled, dest, g)

    def _parallel_block(
        self,
        block_idx: int,
        lane0: torch.Tensor,
        lane1: torch.Tensor,
        x0: torch.Tensor,
        q_w,
        k_w,
        v_w,
        out_w,
        up_w,
        down_w,
        *,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        pos_offset: int = 0,
        use_cache: bool = False,
    ):
        block = self.blocks[block_idx]
        mix = getattr(block, "_ac_resid_mix", None)
        if mix is None or mix.dtype != lane0.dtype:
            mix = block.resid_mix.to(dtype=lane0.dtype)
        attn_read = mix[0][None, None, :] * lane0 + mix[1][None, None, :] * x0
        attn_ret = block.attn(
            block.attn_norm(attn_read) * block.ln_scale_factor,
            q_w,
            k_w,
            v_w,
            out_w,
            past_kv=past_kv,
            pos_offset=pos_offset,
            use_cache=use_cache,
        )
        if use_cache:
            attn_out, present = attn_ret  # type: ignore[misc]
        else:
            attn_out = attn_ret  # type: ignore[assignment]
            present = None
        a_sc = getattr(block, "_ac_attn_scale", None)
        if a_sc is None or a_sc.dtype != attn_out.dtype:
            a_sc = block.attn_scale.to(dtype=attn_out.dtype)
        m_sc = getattr(block, "_ac_mlp_scale", None)
        if m_sc is None or m_sc.dtype != lane1.dtype:
            m_sc = block.mlp_scale.to(dtype=lane1.dtype)
        attn_out = a_sc[None, None, :] * attn_out
        mlp_out = m_sc[None, None, :] * block.mlp(
            block.mlp_norm(lane1) * block.ln_scale_factor, up_w, down_w
        )
        pr = getattr(self, "_ac_parallel_resid", None)
        pp = getattr(self, "_ac_parallel_post", None)
        if pr is None or pr.dtype != lane0.dtype:
            pr = self.parallel_resid_lambdas.to(dtype=lane0.dtype)
        if pp is None or pp.dtype != lane0.dtype:
            pp = self.parallel_post_lambdas.to(dtype=lane0.dtype)
        attn_resid = pr[block_idx, 0]
        attn_post = pp[block_idx, 0]
        mlp_resid = pr[block_idx, 1]
        mlp_post = pp[block_idx, 1]
        lane0 = attn_resid * lane0 + attn_post[0] * attn_out + mlp_post[0] * mlp_out
        lane1 = mlp_resid * lane1 + attn_post[1] * attn_out + mlp_post[1] * mlp_out
        if use_cache:
            assert present is not None
            return lane0, lane1, present
        return lane0, lane1

    def _final_parallel_hidden(self, lane0: torch.Tensor, lane1: torch.Tensor):
        if self.parallel_final_lane == "mlp":
            return lane1
        if self.parallel_final_lane == "attn":
            return lane0
        return 0.5 * (lane0 + lane1)

    def _apply_smear(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        if not self.smear_gate_enabled:
            return x
        sl = self.smear_lambda.to(dtype=x.dtype)
        gate_in = x[:, 1:, : self.smear_window].contiguous()
        g = sl * torch.sigmoid(self.smear_gate(gate_in))
        if self.smear_gate_bos_fix:
            # Byte LM has no BOS id; keep identity at position 0 only.
            x = torch.cat([x[:, :1], x[:, 1:] + g * x[:, :-1]], dim=1)
        else:
            x = torch.cat([x[:, :1], x[:, 1:] + g * x[:, :-1]], dim=1)
        return x

    def _forward_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(input_ids)
        x = self._apply_smear(x, input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[torch.Tensor] = []
        enc_iter = (
            self.encoder_indices
            if self.looping_active
            else range(self.num_encoder_layers)
        )
        dec_iter = (
            self.decoder_indices
            if self.looping_active
            else range(
                self.num_encoder_layers,
                self.num_encoder_layers + self.num_decoder_layers,
            )
        )
        for i in enc_iter:
            x = self._maybe_checkpoint_block(i, x, x0)
            skips.append(x)
        psl = self.parallel_start_layer
        lane0 = None
        lane1 = None
        for skip_idx, i in enumerate(dec_iter):
            q_w, k_w, v_w, out_w, up_w, down_w = self._bank_weights(i)
            if i >= psl and psl > 0:
                if lane0 is None:
                    lane0 = x
                    lane1 = x
                if skip_idx < self.num_skip_weights and skips:
                    skip = skips.pop()
                    w = self.skip_weights[skip_idx].to(dtype=lane0.dtype)[None, None, :]
                    if self.skip_gates is not None:
                        g = torch.sigmoid(
                            self.skip_gates[skip_idx].to(dtype=lane0.dtype)
                        )[None, None, :]
                        lane0 = torch.lerp(w * skip, lane0, g)
                    else:
                        lane0 = lane0 + w * skip
                # Parallel residual path: still checkpoint the fused step.
                if self.training and self.gradient_checkpointing:

                    def _par(
                        l0: torch.Tensor,
                        l1: torch.Tensor,
                        x0_t: torch.Tensor,
                        *,
                        _i: int = i,
                        _qw=q_w,
                        _kw=k_w,
                        _vw=v_w,
                        _ow=out_w,
                        _uw=up_w,
                        _dw=down_w,
                    ):
                        return self._parallel_block(
                            _i, l0, l1, x0_t, _qw, _kw, _vw, _ow, _uw, _dw
                        )

                    lane0, lane1 = checkpoint(
                        _par, lane0, lane1, x0, use_reentrant=False
                    )
                else:
                    lane0, lane1 = self._parallel_block(
                        i, lane0, lane1, x0, q_w, k_w, v_w, out_w, up_w, down_w
                    )
            else:
                if skip_idx < self.num_skip_weights and skips:
                    scaled_skip = (
                        self.skip_weights[skip_idx].to(dtype=x.dtype)[None, None, :]
                        * skips.pop()
                    )
                    if self.skip_gates is not None:
                        g = torch.sigmoid(
                            self.skip_gates[skip_idx].to(dtype=x.dtype)
                        )[None, None, :]
                        x = torch.lerp(scaled_skip, x, g)
                    else:
                        x = x + scaled_skip
                x = self._maybe_checkpoint_block(i, x, x0)
        if lane0 is not None:
            x = self._final_parallel_hidden(lane0, lane1)
        return self.final_norm(x)

    def _project_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        if getattr(self, "_ac_step_active", False):
            if self.tie_embeddings:
                w = getattr(self, "_ac_tok_emb", None)
                if w is not None:
                    return F.linear(hidden, w)
            else:
                w = getattr(self, "_ac_lm_head", None)
                if w is not None:
                    return F.linear(hidden, w)
        if self.tie_embeddings:
            return F.linear(hidden, self.tok_emb.weight)
        assert self.lm_head is not None
        return self.lm_head(hidden)

    def _apply_asym_softcap(self, logits: torch.Tensor) -> torch.Tensor:
        sp = getattr(self, "_ac_softcap_pos", None)
        sn = getattr(self, "_ac_softcap_neg", None)
        if (
            not getattr(self, "_ac_step_active", False)
            or sp is None
            or sn is None
            or sp.dtype != logits.dtype
        ):
            sp = self.softcap_pos.to(logits.dtype)
            sn = self.softcap_neg.to(logits.dtype)
        return torch.where(
            logits >= 0,
            sp * torch.tanh(logits / sp),
            sn * torch.tanh(logits / sn),
        )

    def forward_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self._forward_hidden(input_ids)
        logits_proj = self._project_logits(hidden)
        if self.asym_logit_enabled:
            return self._apply_asym_softcap(logits_proj)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

    def forward(
        self, input_ids: torch.Tensor, target_ids: torch.Tensor
    ) -> torch.Tensor:
        logits = self.forward_logits(input_ids)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            target_ids.reshape(-1),
            reduction="mean",
        )

    def enable_looping(self, active: bool = True) -> None:
        self.looping_active = bool(active) and self.cfg.num_loops > 0

    def _iter_indices(self) -> tuple[list[int], list[int]]:
        if self.looping_active:
            return list(self.encoder_indices), list(self.decoder_indices)
        return (
            list(range(self.num_encoder_layers)),
            list(
                range(
                    self.num_encoder_layers,
                    self.num_encoder_layers + self.num_decoder_layers,
                )
            ),
        )

    def _embed_tokens(
        self,
        input_ids: torch.Tensor,
        *,
        prev_raw_emb: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed (+ smear) tokens. Returns (normed x, raw emb for next smear)."""
        raw = self._raw_token_emb(input_ids)
        if not self.smear_gate_enabled:
            x = F.rms_norm(raw, (raw.size(-1),))
            return x, raw[:, -1:, :].detach()
        bsz, tlen, _ = raw.shape
        if tlen == 1:
            if prev_raw_emb is None:
                x = raw
            else:
                gate_in = raw[:, :, : self.smear_window].contiguous()
                g = self._smear_gate_values(raw, gate_in)
                x = raw + g * prev_raw_emb.to(dtype=raw.dtype)
        else:
            gate_in = raw[:, 1:, : self.smear_window].contiguous()
            g = self._smear_gate_values(raw, gate_in)
            x = torch.cat([raw[:, :1], raw[:, 1:] + g * raw[:, :-1]], dim=1)
        x = F.rms_norm(x, (x.size(-1),))
        return x, raw[:, -1:, :].detach()

    def make_kv_slots(self, max_len: int) -> list[KVSlot]:
        """Preallocated per-slot KV buffers for the incremental AC path."""
        enc_iter, dec_iter = self._iter_indices()
        return [KVSlot(max_len) for _ in range(len(enc_iter) + len(dec_iter))]

    def _embed_tokens_window(
        self, input_ids: torch.Tensor, prev_raw_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed (+ smear) a mid-stream W-token window.

        Unlike :meth:`_embed_tokens` (whose multi-token branch leaves position
        0 unsmeared — sequence-start semantics), every window position smears
        with its predecessor's raw embedding; position 0 is seeded with
        ``prev_raw_emb`` (the last raw of the previous window / prefill, or
        zeros at stream start — the no-previous-token path, since the gated
        term vanishes). W=1 reduces exactly to the single-token extend embed.
        """
        raw = self._raw_token_emb(input_ids)
        if not self.smear_gate_enabled:
            x = F.rms_norm(raw, (raw.size(-1),))
            return x, raw[:, -1:, :].detach()
        prev = torch.cat([prev_raw_emb.to(dtype=raw.dtype), raw[:, :-1]], dim=1)
        gate_in = raw[:, :, : self.smear_window].contiguous()
        g = self._smear_gate_values(raw, gate_in)
        x = F.rms_norm(raw + g * prev, (raw.size(-1),))
        return x, raw[:, -1:, :].detach()

    def make_static_state(
        self, max_len: int, device: torch.device, win: int = 1
    ) -> StaticState:
        """Device buffers for the capturable (CUDA-graphable) W-token AC step."""
        enc_iter, dec_iter = self._iter_indices()
        n_slots = len(enc_iter) + len(dec_iter)
        win = int(win)
        max_len = int(max_len)
        emb_dtype = getattr(self, "_ac_compute_dtype", None) or torch.float32
        pos = torch.zeros(1, dtype=torch.long, device=device)
        arange = torch.arange(max_len, dtype=torch.long, device=device)
        arange_w = torch.arange(win, dtype=torch.long, device=device)
        is_tail = arange >= (max_len - win)
        tail_local = arange - (max_len - win)
        return StaticState(
            token=torch.zeros(1, win, dtype=torch.long, device=device),
            prev_raw=torch.zeros(
                1, 1, self.cfg.model_dim, dtype=emb_dtype, device=device
            ),
            cand_prev_raw=torch.zeros(
                1, 1, self.cfg.model_dim, dtype=emb_dtype, device=device
            ),
            pos=pos,
            arange=arange,
            arange_w=arange_w,
            logits=torch.zeros(
                1, win, self.cfg.vocab_size, dtype=torch.float32, device=device
            ),
            slots=[
                StaticKV(max_len, pos, arange, arange_w, is_tail, tail_local)
                for _ in range(n_slots)
            ],
            max_len=max_len,
            win=win,
        )

    @torch.no_grad()
    def static_extend(self, state: StaticState) -> None:
        """One W-token window AC step on static buffers (CUDA-graph capturable).

        Reads ``state.token`` (1, W) / ``state.prev_raw`` / ``state.pos``
        (window start); writes K/V rows at ``pos + [0, W)``, per-position
        next-byte logits into ``state.logits`` (1, W, V fp32) and the last
        position's raw embedding into ``state.cand_prev_raw`` (promoted to
        ``prev_raw`` by the engine only when the window is finalized, so
        speculative re-runs of the same window stay bit-identical). ``pos`` is
        managed by the caller. No Python-value-dependent shapes or syncs.
        """
        self._set_ac_step_active(True)
        try:
            self._static_extend_body(state, fused=False)
        finally:
            self._set_ac_step_active(False)

    @torch.no_grad()
    def static_extend_fused(self, state: StaticState) -> None:
        """Fused W-token AC step (persistent stack on W=1 CUDA; else fused slots).

        Same single-stream buffers and replenish trajectory as
        :meth:`static_extend`. Numerics are not bit-identical to eager —
        encode and decode must both use this path.
        """
        self._set_ac_step_active(True)
        try:
            self._static_extend_body(state, fused=True)
        finally:
            self._set_ac_step_active(False)

    def _static_extend_body(self, state: StaticState, *, fused: bool) -> None:
        enc_iter, dec_iter = self._iter_indices()
        use_fused = fused and state.token.device.type == "cuda"
        if use_fused:
            if int(state.win) > 1:
                from .mega_encode import try_mega_encode

                if try_mega_encode(self, state):
                    return
            from .persistent_step import try_persistent_extend

            if try_persistent_extend(self, state):
                return
        x, last_raw = self._embed_tokens_window(state.token, state.prev_raw)
        state.cand_prev_raw.copy_(last_raw)
        x0_step = x
        slot = 0
        skips: list[torch.Tensor] = []
        if use_fused:
            from .fused_step import fused_block_forward, fused_parallel_block
        for i in enc_iter:
            q_w, k_w, v_w, out_w, up_w, down_w = self._bank_weights(i)
            if use_fused:
                x = fused_block_forward(
                    self,
                    i,
                    x,
                    x0_step,
                    q_w,
                    k_w,
                    v_w,
                    out_w,
                    up_w,
                    down_w,
                    state.slots[slot],
                )
            else:
                x = self.blocks[i](  # type: ignore[assignment]
                    x,
                    x0_step,
                    q_w,
                    k_w,
                    v_w,
                    out_w,
                    up_w,
                    down_w,
                    past_kv=state.slots[slot],
                    pos_offset=0,
                    use_cache=False,
                )
            slot += 1
            skips.append(x)
        psl = self.parallel_start_layer
        lane0 = None
        lane1 = None
        for skip_idx, i in enumerate(dec_iter):
            q_w, k_w, v_w, out_w, up_w, down_w = self._bank_weights(i)
            if i >= psl and psl > 0:
                if lane0 is None:
                    lane0 = x
                    lane1 = x
                lane0 = self._apply_decoder_skip(skip_idx, lane0, skips)
                if use_fused:
                    lane0, lane1 = fused_parallel_block(
                        self,
                        i,
                        lane0,
                        lane1,
                        x0_step,
                        q_w,
                        k_w,
                        v_w,
                        out_w,
                        up_w,
                        down_w,
                        state.slots[slot],
                    )
                else:
                    lane0, lane1 = self._parallel_block(  # type: ignore[misc]
                        i,
                        lane0,
                        lane1,
                        x0_step,
                        q_w,
                        k_w,
                        v_w,
                        out_w,
                        up_w,
                        down_w,
                        past_kv=state.slots[slot],
                        pos_offset=0,
                        use_cache=False,
                    )
            else:
                x = self._apply_decoder_skip(skip_idx, x, skips)
                if use_fused:
                    x = fused_block_forward(
                        self,
                        i,
                        x,
                        x0_step,
                        q_w,
                        k_w,
                        v_w,
                        out_w,
                        up_w,
                        down_w,
                        state.slots[slot],
                    )
                else:
                    x = self.blocks[i](  # type: ignore[assignment]
                        x,
                        x0_step,
                        q_w,
                        k_w,
                        v_w,
                        out_w,
                        up_w,
                        down_w,
                        past_kv=state.slots[slot],
                        pos_offset=0,
                        use_cache=False,
                    )
            slot += 1
        if lane0 is not None:
            x = self._final_parallel_hidden(lane0, lane1)
        hidden = self.final_norm(x)
        logits = self._project_logits(hidden)
        if self.asym_logit_enabled:
            logits = self._apply_asym_softcap(logits)
        else:
            logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        state.logits.copy_(logits.float())

    def compile_static_extend(
        self,
        state: StaticState,
        *,
        mode: str | None = None,
        fullgraph: bool = False,
        dynamic: bool = False,
    ):
        """Return a ``torch.compile`` copy of :meth:`static_extend`.

        Closes over ``state`` (no-arg callable) so Dynamo sees the same
        static-address buffers every step — required for inductor CUDA
        graphs, which skip on mutated eager inputs. The original method
        stays eager (draft / prefill / A/B). Default mode is
        ``XSA_AC_COMPILE_MODE`` or ``reduce-overhead``. Compile both codec
        sides or neither — fused numerics are not bit-identical to eager.
        """
        if mode is None:
            mode = (
                os.environ.get("XSA_AC_COMPILE_MODE", "reduce-overhead").strip()
                or "reduce-overhead"
            )
        allowed = ("default", "reduce-overhead", "max-autotune")
        if mode not in allowed:
            raise ValueError(
                f"XSA_AC_COMPILE_MODE={mode!r} not in {allowed}"
            )
        state.mark_static_addresses()
        _enable_compile_tf32()
        eager = self.static_extend

        def _fn() -> None:
            eager(state)

        return torch.compile(
            _fn, mode=mode, fullgraph=fullgraph, dynamic=dynamic
        )

    @torch.no_grad()
    def prefill_cache(
        self,
        input_ids: torch.Tensor,
        *,
        kv_slots: "list[KVSlot] | list[StaticKV] | None" = None,
    ) -> "KVCache":
        """Run full prompt once; return KV cache + last-step logits helper state.

        When ``kv_slots`` (:class:`KVSlot` reset, or :class:`StaticKV` from a
        static state) is provided, K/V rows are written into the preallocated
        buffers and later extends are in-place — no per-step ``torch.cat``.
        """
        enc_iter, dec_iter = self._iter_indices()
        n_slots = len(enc_iter) + len(dec_iter)
        if kv_slots is not None:
            if len(kv_slots) != n_slots:
                raise ValueError(
                    f"kv_slots has {len(kv_slots)} slots, model needs {n_slots}"
                )
            if any(
                isinstance(s, KVSlot) and s.length != 0 for s in kv_slots
            ):
                raise ValueError("kv_slots must be reset() before prefill")
        x, last_raw = self._embed_tokens(input_ids)
        x0 = x
        pos_offset = 0
        seqlen = int(input_ids.size(1))
        kv_out: list[tuple[torch.Tensor, torch.Tensor] | KVSlot | None] = (
            [None] * n_slots
        )

        def _past_for(slot_idx: int):
            return kv_slots[slot_idx] if kv_slots is not None else None

        skips: list[torch.Tensor] = []
        slot = 0
        for i in enc_iter:
            q_w, k_w, v_w, out_w, up_w, down_w = self._bank_weights(i)
            x, present = self.blocks[i](  # type: ignore[misc]
                x,
                x0,
                q_w,
                k_w,
                v_w,
                out_w,
                up_w,
                down_w,
                past_kv=_past_for(slot),
                pos_offset=pos_offset,
                use_cache=True,
            )
            kv_out[slot] = present
            slot += 1
            skips.append(x)
        psl = self.parallel_start_layer
        lane0 = None
        lane1 = None
        for skip_idx, i in enumerate(dec_iter):
            q_w, k_w, v_w, out_w, up_w, down_w = self._bank_weights(i)
            if i >= psl and psl > 0:
                if lane0 is None:
                    lane0 = x
                    lane1 = x
                if skip_idx < self.num_skip_weights and skips:
                    skip = skips.pop()
                    w = self.skip_weights[skip_idx].to(dtype=lane0.dtype)[None, None, :]
                    if self.skip_gates is not None:
                        g = torch.sigmoid(
                            self.skip_gates[skip_idx].to(dtype=lane0.dtype)
                        )[None, None, :]
                        lane0 = torch.lerp(w * skip, lane0, g)
                    else:
                        lane0 = lane0 + w * skip
                lane0, lane1, present = self._parallel_block(  # type: ignore[misc]
                    i,
                    lane0,
                    lane1,
                    x0,
                    q_w,
                    k_w,
                    v_w,
                    out_w,
                    up_w,
                    down_w,
                    past_kv=_past_for(slot),
                    pos_offset=pos_offset,
                    use_cache=True,
                )
            else:
                if skip_idx < self.num_skip_weights and skips:
                    scaled_skip = (
                        self.skip_weights[skip_idx].to(dtype=x.dtype)[None, None, :]
                        * skips.pop()
                    )
                    if self.skip_gates is not None:
                        g = torch.sigmoid(
                            self.skip_gates[skip_idx].to(dtype=x.dtype)
                        )[None, None, :]
                        x = torch.lerp(scaled_skip, x, g)
                    else:
                        x = x + scaled_skip
                x, present = self.blocks[i](  # type: ignore[misc]
                    x,
                    x0,
                    q_w,
                    k_w,
                    v_w,
                    out_w,
                    up_w,
                    down_w,
                    past_kv=_past_for(slot),
                    pos_offset=pos_offset,
                    use_cache=True,
                )
            kv_out[slot] = present
            slot += 1
        if lane0 is not None:
            x = self._final_parallel_hidden(lane0, lane1)
        hidden = self.final_norm(x)
        return KVCache(
            kv_slots=kv_out,
            x0=x0,
            last_raw_emb=last_raw,
            length=seqlen,
            last_hidden=hidden[:, -1:, :],
        )

    @torch.no_grad()
    def extend_cache(
        self, cache: "KVCache", token_id: int | torch.Tensor
    ) -> tuple["KVCache", torch.Tensor]:
        """Append one token; return updated cache and logits for the *next* byte."""
        if isinstance(token_id, int):
            ids = torch.tensor([[token_id]], device=cache.x0.device, dtype=torch.long)
        else:
            ids = token_id.view(1, 1).to(device=cache.x0.device, dtype=torch.long)
        enc_iter, dec_iter = self._iter_indices()
        x, last_raw = self._embed_tokens(ids, prev_raw_emb=cache.last_raw_emb)
        # resid_mix reads x0 at the current position only, so the cache keeps
        # just the newest slice (constant memory across a 16k-extend segment).
        x0_step = x
        pos_offset = cache.length
        kv_out: list[tuple[torch.Tensor, torch.Tensor] | KVSlot | None] = []
        skips: list[torch.Tensor] = []
        slot = 0
        for i in enc_iter:
            q_w, k_w, v_w, out_w, up_w, down_w = self._bank_weights(i)
            past = cache.kv_slots[slot]
            x, present = self.blocks[i](  # type: ignore[misc]
                x,
                x0_step,
                q_w,
                k_w,
                v_w,
                out_w,
                up_w,
                down_w,
                past_kv=past,
                pos_offset=pos_offset,
                use_cache=True,
            )
            kv_out.append(present)
            slot += 1
            skips.append(x)
        psl = self.parallel_start_layer
        lane0 = None
        lane1 = None
        for skip_idx, i in enumerate(dec_iter):
            q_w, k_w, v_w, out_w, up_w, down_w = self._bank_weights(i)
            past = cache.kv_slots[slot]
            if i >= psl and psl > 0:
                if lane0 is None:
                    lane0 = x
                    lane1 = x
                if skip_idx < self.num_skip_weights and skips:
                    skip = skips.pop()
                    w = self.skip_weights[skip_idx].to(dtype=lane0.dtype)[None, None, :]
                    if self.skip_gates is not None:
                        g = torch.sigmoid(
                            self.skip_gates[skip_idx].to(dtype=lane0.dtype)
                        )[None, None, :]
                        lane0 = torch.lerp(w * skip, lane0, g)
                    else:
                        lane0 = lane0 + w * skip
                lane0, lane1, present = self._parallel_block(  # type: ignore[misc]
                    i,
                    lane0,
                    lane1,
                    x0_step,
                    q_w,
                    k_w,
                    v_w,
                    out_w,
                    up_w,
                    down_w,
                    past_kv=past,
                    pos_offset=pos_offset,
                    use_cache=True,
                )
            else:
                if skip_idx < self.num_skip_weights and skips:
                    scaled_skip = (
                        self.skip_weights[skip_idx].to(dtype=x.dtype)[None, None, :]
                        * skips.pop()
                    )
                    if self.skip_gates is not None:
                        g = torch.sigmoid(
                            self.skip_gates[skip_idx].to(dtype=x.dtype)
                        )[None, None, :]
                        x = torch.lerp(scaled_skip, x, g)
                    else:
                        x = x + scaled_skip
                x, present = self.blocks[i](  # type: ignore[misc]
                    x,
                    x0_step,
                    q_w,
                    k_w,
                    v_w,
                    out_w,
                    up_w,
                    down_w,
                    past_kv=past,
                    pos_offset=pos_offset,
                    use_cache=True,
                )
            kv_out.append(present)
            slot += 1
        if lane0 is not None:
            x = self._final_parallel_hidden(lane0, lane1)
        hidden = self.final_norm(x)
        # Caller re-prefills at segment boundaries (context window reset).
        new_cache = KVCache(
            kv_slots=kv_out,
            x0=x,
            last_raw_emb=last_raw,
            length=cache.length + 1,
            last_hidden=hidden,
        )
        logits = self._project_logits(hidden)
        if self.asym_logit_enabled:
            logits = self._apply_asym_softcap(logits)
        else:
            logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        return new_cache, logits

    @torch.no_grad()
    def logits_from_cache(self, cache: "KVCache") -> torch.Tensor:
        """Logits at the last cached position (predict next byte)."""
        hidden = cache.last_hidden
        logits = self._project_logits(hidden)
        if self.asym_logit_enabled:
            return self._apply_asym_softcap(logits)
        return self.logit_softcap * torch.tanh(logits / self.logit_softcap)

class KVCache:
    """Incremental decode state (one KV pair or preallocated slot per attn slot).

    ``x0`` holds the init-features slice of the *newest* position only (that is
    all resid_mix reads during single-token extends) plus serves as the device
    anchor for new token ids.
    """

    __slots__ = ("kv_slots", "x0", "last_raw_emb", "length", "last_hidden")

    def __init__(
        self,
        *,
        kv_slots: list[tuple[torch.Tensor, torch.Tensor] | KVSlot | None],
        x0: torch.Tensor,
        last_raw_emb: torch.Tensor,
        length: int,
        last_hidden: torch.Tensor,
    ):
        self.kv_slots = kv_slots
        self.x0 = x0
        self.last_raw_emb = last_raw_emb
        self.length = int(length)
        self.last_hidden = last_hidden


def build_model(cfg: "XsaTttConfig", *, device: torch.device | None = None) -> GPT:
    model = GPT(cfg)
    # Compress / online path: depth recurrence on.
    model.enable_looping(True)
    if device is not None:
        model = model.to(device)
    return model
