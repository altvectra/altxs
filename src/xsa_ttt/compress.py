"""Arithmetic coding with NNCP-style online LoRA retrain (large arch).

Lockstep (encode ≡ decode):
  1. Before predicting byte ``i``, if ``i > 0`` and ``i % every == 0``:
     AdamW step(s) on the last ``block_size`` window of the already-agreed prefix.
  2. Emit / consume ``P(x_i | prefix)`` from the (adapted) model.

Default ``retrain_mode=full`` updates all params (NNCP-style). ``retrain_mode=lora``
freezes the base GPT and updates LoRA adapters.
``retrain_mode=replenish`` updates all params with the *pretrain* loss shape
(CE on the just-coded chunk only, previous block as context) — continuous
replenishment training for quant-damaged checkpoints.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[1]
_MODEL = _REPO / "model"
for p in (_REPO, _MODEL):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from arithmetic_coder_lm import (  # noqa: E402
    decode_with_probs,
    encode_decode_segment,
    encode_with_probs,
)

from .config import XsaTttConfig, should_online_retrain_at
from .model import GPT
from .ttt_lora import (
    TTTLoRA,
    _forward_logits_lora,
    ce_with_lora,
    forward_logits_with_lora,
    make_ttt_optimizer,
)

def uniform_probs(vocab_size: int, *, dtype: np.dtype = np.float32) -> np.ndarray:
    v = max(1, int(vocab_size))
    return np.full(v, 1.0 / v, dtype=dtype)


def _amp_ctx(device: torch.device, use_bf16: bool):
    if use_bf16 and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _softmax_np(
    logits: torch.Tensor,
    *,
    vocab_size: int | None = None,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Softmax → numpy for AC (row or batch of rows). Default float32."""
    p = (
        torch.softmax(logits.detach().float(), dim=-1)
        .cpu()
        .numpy()
        .astype(dtype, copy=False)
    )
    v = int(vocab_size) if vocab_size is not None else int(p.shape[-1])
    if p.ndim == 1:
        s = float(p.sum())
        if s <= 0 or not math.isfinite(s):
            return uniform_probs(v, dtype=dtype)
        p /= s
        return p
    p /= np.maximum(p.sum(axis=-1, keepdims=True), 1e-300)
    return p


def _sym_dtype(vocab_size: int) -> np.dtype:
    return np.dtype(np.uint8 if int(vocab_size) <= 256 else np.uint16)


def _as_symbol_array(data: bytes | np.ndarray, *, vocab_size: int) -> np.ndarray:
    if isinstance(data, (bytes, bytearray)):
        if vocab_size > 256:
            raise ValueError("bytes input requires vocab_size <= 256")
        return np.frombuffer(data, dtype=np.uint8)
    return np.asarray(data, dtype=_sym_dtype(vocab_size))


def _symbols_for_hash(arr: np.ndarray) -> bytes:
    """Stable byte view of the symbol stream for SHA."""
    if arr.dtype == np.uint8 or arr.dtype == np.dtype("u1"):
        return (
            arr.tobytes() if not isinstance(arr, np.memmap) else bytes(arr)
        )
    return np.asarray(arr, dtype="<u2").tobytes()


def _seg_decoded_to_bytes(
    seg_decoded: bytes | bytearray | list[int] | np.ndarray,
    *,
    vocab_size: int,
) -> bytes:
    """Serialize one decoded segment the same way SHA hashes the stream."""
    if vocab_size <= 256:
        if isinstance(seg_decoded, (bytes, bytearray)):
            return bytes(seg_decoded)
        return bytes(np.asarray(seg_decoded, dtype=np.uint8).reshape(-1))
    return np.asarray(seg_decoded, dtype="<u2").reshape(-1).tobytes()


def _chunk_ctx_start(chunk_start: int, block_size: int) -> tuple[int, int]:
    """Return ``(p0, ctx_start)`` for chunked TF matching AC encode."""
    p0 = 1 if chunk_start == 0 else int(chunk_start)
    return p0, max(0, p0 - int(block_size))


def chunk_tf_train_loss(
    model: GPT,
    arr: np.ndarray,
    *,
    chunk_start: int,
    chunk_end: int,
    block_size: int,
    device: torch.device,
    use_bf16: bool = False,
    lora: TTTLoRA | None = None,
) -> torch.Tensor:
    """One chunked TF forward; CE on bytes ``[chunk_start, chunk_end)`` only.

    Same ``ctx_start`` / sequence as ``_chunk_tf_probs`` (AC encode), so pretrain
    matches the compress path. Gradients flow into ``model``. When a (frozen
    healer) ``lora`` is attached, the forward goes through the adapter path so
    the trained base weights match the serving distribution exactly.
    """
    p0, ctx_start = _chunk_ctx_start(chunk_start, block_size)
    if p0 >= chunk_end or chunk_end - ctx_start < 2:
        return torch.zeros((), device=device, requires_grad=True)
    ids = torch.as_tensor(
        np.asarray(arr[ctx_start:chunk_end], dtype=np.int64), device=device
    ).unsqueeze(0)
    x = ids[:, :-1]
    y = ids[:, 1:]
    # y[j] = arr[ctx_start + j + 1]; want j such that target index >= p0
    first = p0 - ctx_start - 1
    if first < 0:
        first = 0
    with _amp_ctx(device, use_bf16):
        if lora is not None:
            logits = _forward_logits_lora(model, x, lora)
        else:
            logits = model.forward_logits(x)
    logits_f = logits[:, first:, :].float()
    targets = y[:, first:]
    if targets.numel() == 0:
        return torch.zeros((), device=device, requires_grad=True)
    return F.cross_entropy(
        logits_f.reshape(-1, logits_f.size(-1)),
        targets.reshape(-1),
        reduction="mean",
    )


@torch.no_grad()
def _chunk_tf_probs(
    model: GPT,
    arr: np.ndarray,
    *,
    start: int,
    end: int,
    block_size: int,
    device: torch.device,
    lora: TTTLoRA | None,
    use_bf16: bool,
    dtype: np.dtype = np.float32,
    row_batch: int = 1024,
) -> np.ndarray:
    """Causal TF probs for bytes ``[start, end)`` — one GPU forward.

    ``logits[p]`` uses ``arr[ctx_start:p]`` with
    ``ctx_start = max(0, start - block_size)``. Encode and decode must share
    this exact gather (decode materializes the same chunk via KV extend).

    Softmax runs on GPU in float32; host rows are ``dtype`` (default float32).
    Logit→prob conversion is batched by ``row_batch`` to limit peak VRAM.
    """
    n = end - start
    if n <= 0:
        # Infer vocab from embedding when empty (should be rare).
        v = int(getattr(getattr(model, "cfg", None), "vocab_size", 256))
        return np.empty((0, v), dtype=dtype)
    p0, ctx_start = _chunk_ctx_start(start, block_size)
    ids = torch.as_tensor(
        np.asarray(arr[ctx_start:end], dtype=np.int64), device=device
    ).unsqueeze(0)
    with _amp_ctx(device, use_bf16):
        logits = forward_logits_with_lora(model, ids, lora)
    vocab_size = int(logits.size(-1))
    out = np.empty((n, vocab_size), dtype=dtype)
    row = 0
    if start == 0:
        out[0] = uniform_probs(vocab_size, dtype=dtype)
        row = 1
    if p0 >= end:
        return out
    ks = torch.arange(p0, end, device=device, dtype=torch.long) - ctx_start - 1
    gathered = logits[0].index_select(0, ks)
    n_g = int(gathered.size(0))
    batch = max(1, int(row_batch))
    for b0 in range(0, n_g, batch):
        b1 = min(n_g, b0 + batch)
        out[row + b0 : row + b1] = _softmax_np(
            gathered[b0:b1], vocab_size=vocab_size, dtype=dtype
        )
    return out


@torch.no_grad()
def _encode_segment_tf(
    model: GPT,
    arr: np.ndarray,
    seg: bytes | np.ndarray,
    *,
    start: int,
    end: int,
    block_size: int,
    device: torch.device,
    lora: TTTLoRA | None,
    use_bf16: bool,
    alphabet_size: int,
    row_batch: int = 1024,
) -> bytes:
    """Encode ``seg`` from one TF forward; softmax host batches only (no full NxV)."""
    n = end - start
    if n <= 0:
        return b""
    if len(seg) != n:
        raise ValueError(f"seg length {len(seg)} != end-start {n}")
    p0, ctx_start = _chunk_ctx_start(start, block_size)
    ids = torch.as_tensor(
        np.asarray(arr[ctx_start:end], dtype=np.int64), device=device
    ).unsqueeze(0)
    with _amp_ctx(device, use_bf16):
        logits = forward_logits_with_lora(model, ids, lora)
    vocab_size = int(logits.size(-1))
    if vocab_size != int(alphabet_size):
        raise ValueError(
            f"model vocab {vocab_size} != alphabet_size {alphabet_size}"
        )
    gathered = None
    if p0 < end:
        ks = torch.arange(p0, end, device=device, dtype=torch.long) - ctx_start - 1
        gathered = logits[0].index_select(0, ks)

    uni = uniform_probs(vocab_size, dtype=np.float32)
    batch = max(1, int(row_batch))
    cache_b0 = -1
    cache: np.ndarray | None = None

    def probs_fn(i: int, _prefix):
        nonlocal cache_b0, cache
        if start == 0 and i == 0:
            return uni
        # Row in ``gathered`` for segment-local index ``i``.
        gi = i - (1 if start == 0 else 0)
        if gathered is None:
            return uni
        b0 = (gi // batch) * batch
        if cache is None or b0 != cache_b0:
            b1 = min(int(gathered.size(0)), b0 + batch)
            cache = _softmax_np(
                gathered[b0:b1], vocab_size=vocab_size, dtype=np.float32
            )
            cache_b0 = b0
        return cache[gi - b0]

    return encode_with_probs(
        seg, probs_fn, desc=None, alphabet_size=alphabet_size
    )


def _chunk_bounds(i: int, *, every: int, n_total: int) -> tuple[int, int]:
    """Chunk containing byte ``i``, aligned to retrain boundaries."""
    if every <= 0:
        return 0, n_total
    start = (i // every) * every
    end = min(n_total, start + every)
    return start, max(end, i + 1)


def _window_xy(
    arr: np.ndarray,
    *,
    end: int,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Teacher-forced (x, y) from ``arr[:end]`` — last ``block_size`` positions."""
    if end < 2:
        return None
    t = min(int(block_size), end - 1)
    start = end - t - 1
    # x = arr[start:start+t], y = arr[start+1:start+1+t]
    x_np = np.asarray(arr[start : start + t], dtype=np.int64)
    y_np = np.asarray(arr[start + 1 : start + 1 + t], dtype=np.int64)
    if x_np.shape[0] < 2:
        return None
    x = torch.tensor(x_np, dtype=torch.long, device=device).unsqueeze(0)
    y = torch.tensor(y_np, dtype=torch.long, device=device).unsqueeze(0)
    return x, y


def _online_steps_at(position: int, cfg: XsaTttConfig, every: int) -> int:
    bootstrap_at = int(getattr(cfg, "ttt_bootstrap_symbols", 0))
    bootstrap_steps = int(getattr(cfg, "ttt_bootstrap_steps", 0))
    if position > 0 and bootstrap_steps > 0 and position == bootstrap_at:
        return bootstrap_steps
    if not should_online_retrain_at(position, every=every):
        return 0
    steps = int(cfg.online_retrain_steps)
    # Escalating replay bursts: each fires once, at the first retrain
    # boundary at/after its position (deterministic on both codec sides).
    spec = str(getattr(cfg, "ttt_burst_schedule", "") or "")
    if spec:
        from xsa_ttt.config import parse_burst_schedule

        for pos, burst_steps in parse_burst_schedule(spec):
            if position - every < pos <= position:
                steps = max(steps, int(burst_steps))
    return steps


def _replay_window_end(
    *, end: int, step_index: int, seed: int, block_size: int
) -> int:
    """Deterministic older-window end for NNCP-style replay retrain.

    Step 0 always trains the newest window ending at ``end``; steps >= 1 draw
    a window end uniformly from the decoded prefix. Depends only on
    ``(seed, end, step_index)``, both sides of the codec compute the same
    windows, so encode and decode stay lockstep.
    """
    lo = int(min(end, int(block_size) + 1))
    if step_index <= 0 or lo >= end:
        return int(end)
    mix = (int(seed) * 1_000_003 + int(end) * 31) ^ (
        int(step_index) * 0x9E3779B1
    )
    rng = np.random.default_rng(mix & 0x7FFF_FFFF_FFFF)
    return int(rng.integers(lo, int(end) + 1))


def _scheduled_replenish_lr(
    cfg: XsaTttConfig,
    end: int,
    *,
    n_total: int | None = None,
) -> float:
    """Effective replenish LR at stream position ``end`` (anneal + warmup).

    Mirrors the schedule applied inside ``online_retrain`` so live AC
    readouts can print the LR that governs the current boundary without
    reading optimizer state. Deterministic in ``(cfg, end, n_total)``.
    """
    lr0 = float(cfg.ttt_lora_lr)
    anneal_mode = str(getattr(cfg, "ttt_replenish_anneal", "exp")).lower()
    half_mb = float(getattr(cfg, "ttt_replenish_lr_half_mb", 0.0))
    factor: float | None = None
    if anneal_mode == "linear" and n_total:
        total = int(getattr(cfg, "ttt_replenish_total_bytes", 0)) or int(n_total)
        factor = max(0.0, 1.0 - float(end) / float(total))
    elif half_mb > 0:
        factor = 2.0 ** (-float(end) / (half_mb * 1e6))
    lr_eff = lr0
    if factor is not None:
        lr_eff = max(
            float(getattr(cfg, "ttt_replenish_lr_min", 4e-5)),
            lr0 * factor,
        )
    warmup = max(0, int(getattr(cfg, "ttt_replenish_warmup_steps", 0)))
    if warmup > 0:
        every_w = max(1, int(cfg.online_retrain_every))
        boundary_i = max(1, int(end) // every_w)
        if boundary_i <= warmup:
            return 0.0
    return float(lr_eff)


def _replenish_anchor_state(
    model: GPT, cfg: XsaTttConfig, device: torch.device
) -> dict[str, Any] | None:
    """Lazy per-stream precision-aware anchor (cached on the model).

    Loads per-weight stiffness masks (``mixed_bit_delta anchor``) and
    snapshots the shipped weights at the first replenish boundary. Both
    codec sides build identical state from identical inputs, so anchored
    trajectories stay lockstep. Weights without a mask (or shape-mismatched)
    are left free.
    """
    path = str(getattr(cfg, "ttt_replenish_anchor", "") or "")
    if not path:
        return None
    cached = getattr(model, "_replenish_anchor", None)
    if cached is not None:
        return cached
    from safetensors.torch import load_file as _st_load_file

    masks = _st_load_file(path, device=str(device))
    rate = float(getattr(cfg, "ttt_replenish_anchor_rate", 0.05))
    params = dict(model.named_parameters())
    entries: list[tuple[torch.nn.Parameter, torch.Tensor, torch.Tensor]] = []
    for name, lam in masks.items():
        p = params.get(name)
        if p is None or tuple(p.shape) != tuple(lam.shape):
            continue
        pull = (lam.to(dtype=torch.float32) * rate).to(dtype=p.dtype)
        entries.append((p, p.detach().clone(), pull))
    state: dict[str, Any] = {"entries": entries, "rate": rate, "path": path}
    model._replenish_anchor = state
    n_par = sum(int(p.numel()) for p, _, _ in entries)
    print(
        f"[xsa_ttt] replenish anchor: tensors={len(entries)} "
        f"params={n_par:,} rate={rate:g} masks={path}",
        flush=True,
    )
    return state


def _apply_replenish_anchor(state: dict[str, Any] | None) -> None:
    """Pull anchored weights toward their shipped values: w -= pull*(w-w0)."""
    if not state:
        return
    with torch.no_grad():
        for p, w0, pull in state["entries"]:
            p.data.sub_((p.data - w0) * pull)


def _xm_candidates(
    cfg: XsaTttConfig, end: int, seed: int, k: int
) -> list[tuple[int, int, float]]:
    """K deterministic candidate steps for a Forward-XM replenish boundary.

    Each candidate is ``(chunk_start, chunk_end, lr_scale)``. All candidates
    train on the previous chunk (the one before the just-coded boundary);
    exploration is over update scale only:
    0.5x / 1x / 2x, plus 4x while ``end`` is still in the first
    ``ttt_replenish_xm_4x_until_bytes`` (default 100 MiB). Falls back to
    the fresh chunk only when no previous chunk exists yet.
    Depends only on ``(cfg, end, k)`` — both codec sides enumerate
    identical lists. ``seed`` is unused (kept for call-site stability).
    """
    del seed  # window is fixed to previous chunk; no stochastic draws
    every = max(1, int(cfg.online_retrain_every))
    fresh_end = int(end)
    prev_end = fresh_end - every
    if prev_end >= 2:
        c_end = prev_end
        c_start = max(0, prev_end - every)
    else:
        c_end = fresh_end
        c_start = max(0, fresh_end - every)
    # Update-scale ladder: half / schedule / 2x. Negative scale inverted
    # Adam's moments (step at +1x, then W0 + s·ΔW) and 0.5x already
    # covers a smaller step. 4x is offered only in the head.
    scales = [0.5, 1.0, 2.0]
    n = max(2, int(k))
    chosen = scales[: min(n, len(scales))]
    until = int(getattr(cfg, "ttt_replenish_xm_4x_until_bytes", 0) or 0)
    if until > 0 and int(end) < until and 4.0 not in chosen:
        chosen.append(4.0)
    return [(c_start, c_end, float(s)) for s in chosen]


def _xm_probe_windows(
    end: int, *, every: int, n_probe: int, block_size: int
) -> list[tuple[int, int, int]]:
    """Held-out probe windows as ``(ctx_start, chunk_end, p0)``.

    Walks backward from ``end`` (all bytes < ``end``). The XM train chunk
    — the previous chunk ``[end-2e, end-e)`` — is skipped so probe CE is
    not the train loss. ``n_probe=1`` is the just-coded (fresh) chunk;
    ``n_probe=2`` is fresh plus the chunk *before* train. If there is no
    previous chunk yet, nothing is skipped (train is the fresh window).
    """
    every = max(1, int(every))
    fresh_end = int(end)
    prev_end = fresh_end - every
    if prev_end >= 2:
        train_end = prev_end
        train_start = max(0, prev_end - every)
    else:
        train_end = train_start = -1

    out: list[tuple[int, int, int]] = []
    c_end = fresh_end
    # +1 skip for the train chunk, plus a few extra if tiny leading chunks
    # fail the length check.
    for _ in range(max(1, n_probe) + 8):
        if len(out) >= max(1, n_probe) or c_end < 2:
            break
        c_start = max(0, c_end - every)
        if c_start == train_start and c_end == train_end:
            c_end = c_start
            continue
        p0, ctx_start = _chunk_ctx_start(c_start, block_size)
        if p0 < c_end and c_end - ctx_start >= 2:
            out.append((ctx_start, c_end, p0))
        c_end = c_start
    return out


def _xm_probe_ce(
    model: GPT,
    arr: np.ndarray,
    *,
    end: int,
    cfg: XsaTttConfig,
    device: torch.device,
    lora: TTTLoRA | None,
    use_bf16: bool,
) -> float:
    """Mean CE over held-out probe chunks (skips the XM train window).

    ``n_probe=1``: just-coded chunk. ``n_probe=2``: that plus the chunk
    before train. Every scored byte is < ``end`` (already agreed by both
    codec sides), so decode computes the identical score — scoring on
    future bytes would be illegal.

    Batches all probe windows into one ``(B, T)`` forward (right-padded;
    causal attention ignores pads for real positions) so the GPU sees one
    fat kernel instead of N thin ones. Numerically matches the mean of
    per-chunk ``chunk_tf_train_loss`` (same first/target slices).
    """
    every = max(1, int(cfg.online_retrain_every))
    n_probe = max(1, int(getattr(cfg, "ttt_replenish_xm_probe_chunks", 1)))
    block_size = int(cfg.block_size)
    windows = _xm_probe_windows(
        end, every=every, n_probe=n_probe, block_size=block_size
    )
    if not windows:
        return 0.0
    prev_ckpt = bool(getattr(model, "gradient_checkpointing", False))
    model.gradient_checkpointing = False  # no backward: plain forward
    try:
        with torch.no_grad():
            lengths = [ce - cs for cs, ce, _ in windows]
            max_len = max(lengths)
            bsz = len(windows)
            ids = torch.zeros(bsz, max_len, dtype=torch.long, device=device)
            for i, (ctx_start, chunk_end, _) in enumerate(windows):
                seq = np.asarray(arr[ctx_start:chunk_end], dtype=np.int64)
                ids[i, : seq.shape[0]] = torch.as_tensor(seq, device=device)
            x = ids[:, :-1]
            y = ids[:, 1:]
            with _amp_ctx(device, use_bf16):
                if lora is not None:
                    logits = _forward_logits_lora(model, x, lora)
                else:
                    logits = model.forward_logits(x)
            total = 0.0
            for i, (ctx_start, chunk_end, p0) in enumerate(windows):
                seq_len = chunk_end - ctx_start
                first = max(0, p0 - ctx_start - 1)
                # Right-pad: only score real token positions [first, seq_len-1).
                logits_i = logits[i, first : seq_len - 1].float()
                targets_i = y[i, first : seq_len - 1]
                if targets_i.numel() == 0:
                    continue
                total += float(
                    F.cross_entropy(logits_i, targets_i, reduction="mean").item()
                )
            return total / max(1, bsz)
    finally:
        model.gradient_checkpointing = prev_ckpt


def _xm_param_buffers(
    model: GPT, params: list[torch.nn.Parameter]
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Reuse per-stream W0/ΔW buffers so XM does not allocate 2× params/boundary."""
    w0 = getattr(model, "_xm_w0", None)
    delta = getattr(model, "_xm_delta", None)
    ok = (
        isinstance(w0, list)
        and isinstance(delta, list)
        and len(w0) == len(params)
        and len(delta) == len(params)
        and all(
            a.shape == p.shape and a.dtype == p.dtype and a.device == p.device
            for a, p in zip(w0, params)
        )
    )
    if not ok:
        w0 = [torch.empty_like(p) for p in params]
        delta = [torch.empty_like(p) for p in params]
        model._xm_w0 = w0
        model._xm_delta = delta
    return w0, delta


def _xm_replenish_boundary(
    model: GPT,
    arr: np.ndarray,
    *,
    end: int,
    cfg: XsaTttConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    lora: TTTLoRA | None,
    seed: int,
    lr_eff: float,
    anchor: dict[str, Any] | None,
    use_bf16: bool,
) -> float:
    """Forward-XM over update candidates (arXiv:2607.27372, adapted).

    Fast path (shared previous-chunk window): one AdamW step at 1x to get
    ΔW and updated moments, score each update scale by materializing
    ``W0 + s·ΔW`` (AdamW-homogeneous in lr), keep the best materialized
    weights. Moments from the 1x step are LR-invariant, so no optimizer
    deepcopy and no second backward. Probes are no-grad forwards only.
    Deterministic in ``(cfg, end)`` + shared prefix → lockstep.
    """
    k = int(getattr(cfg, "ttt_replenish_xm_k", 0))
    cands = _xm_candidates(cfg, end, seed, k)
    params = [p for g in optimizer.param_groups for p in g["params"]]
    w0, delta = _xm_param_buffers(model, params)
    with torch.no_grad():
        for dst, p in zip(w0, params):
            dst.copy_(p)

    # Shared window (all candidates use previous chunk; update scale differs).
    c_start, c_end, _ = cands[0]
    if c_end < 2 or any(c[:2] != (c_start, c_end) for c in cands):
        opt_snap = copy.deepcopy(optimizer.state_dict())
        return _xm_replenish_boundary_serial(
            model,
            arr,
            end=end,
            cfg=cfg,
            device=device,
            optimizer=optimizer,
            lora=lora,
            lr_eff=lr_eff,
            anchor=anchor,
            use_bf16=use_bf16,
            cands=cands,
            params=params,
            w0=[w.detach().clone() for w in w0],
            opt_snap=opt_snap,
        )

    for g in optimizer.param_groups:
        g["lr"] = float(lr_eff)
    optimizer.zero_grad(set_to_none=True)
    loss = chunk_tf_train_loss(
        model,
        arr,
        chunk_start=c_start,
        chunk_end=c_end,
        block_size=int(cfg.block_size),
        device=device,
        use_bf16=use_bf16,
        lora=lora,
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    optimizer.step()
    last = float(loss.detach().float().item())
    del loss
    # ΔW at scale=1 (pre-anchor). Anchor is reapplied after each materialize.
    with torch.no_grad():
        for d, p, w in zip(delta, params, w0):
            d.copy_(p - w)

    def _materialize(scale: float) -> None:
        s = float(scale)
        with torch.no_grad():
            for p, w, d in zip(params, w0, delta):
                p.copy_(w + s * d)
        _apply_replenish_anchor(anchor)

    scores: list[float] = []
    for _, _, scale in cands:
        _materialize(scale)
        scores.append(
            _xm_probe_ce(
                model,
                arr,
                end=end,
                cfg=cfg,
                device=device,
                lora=lora,
                use_bf16=use_bf16,
            )
        )
    best = min(range(len(scores)), key=lambda i: (scores[i], i))
    scale = float(cands[best][2])
    _materialize(scale)
    for g in optimizer.param_groups:
        g["lr"] = float(lr_eff) * scale
    model._xm_last = {
        "end": int(end),
        "chosen": int(best),
        "scores": scores,
        "candidates": cands,
    }
    return last


def _xm_replenish_boundary_serial(
    model: GPT,
    arr: np.ndarray,
    *,
    end: int,
    cfg: XsaTttConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    lora: TTTLoRA | None,
    lr_eff: float,
    anchor: dict[str, Any] | None,
    use_bf16: bool,
    cands: list[tuple[int, int, float]],
    params: list[torch.nn.Parameter],
    w0: list[torch.Tensor],
    opt_snap: dict[str, Any],
) -> float:
    """Serial XM fallback when candidates do not share a training window."""

    def _apply(c_start: int, c_end: int, scale: float) -> float:
        for g in optimizer.param_groups:
            g["lr"] = float(lr_eff) * float(scale)
        optimizer.zero_grad(set_to_none=True)
        loss = chunk_tf_train_loss(
            model,
            arr,
            chunk_start=c_start,
            chunk_end=c_end,
            block_size=int(cfg.block_size),
            device=device,
            use_bf16=use_bf16,
            lora=lora,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        _apply_replenish_anchor(anchor)
        return float(loss.detach().float().item())

    def _restore() -> None:
        with torch.no_grad():
            for p, w in zip(params, w0):
                p.copy_(w)
        optimizer.load_state_dict(opt_snap)

    scores: list[float] = []
    for c_start, c_end, scale in cands:
        if c_end < 2:
            scores.append(float("inf"))
            continue
        _apply(c_start, c_end, scale)
        scores.append(
            _xm_probe_ce(
                model,
                arr,
                end=end,
                cfg=cfg,
                device=device,
                lora=lora,
                use_bf16=use_bf16,
            )
        )
        _restore()
    best = min(range(len(scores)), key=lambda i: (scores[i], i))
    c_start, c_end, scale = cands[best]
    last = _apply(c_start, c_end, scale) if c_end >= 2 else float("nan")
    model._xm_last = {
        "end": int(end),
        "chosen": int(best),
        "scores": scores,
        "candidates": cands,
    }
    return last


# Update-scale labels for live XM readouts (all use the previous chunk).
_XM_LABELS = {
    0: "0.5x",
    1: "1x",
    2: "2x",
    3: "4x",
}


def _xm_share_str(counts: dict[int, int]) -> str:
    """Selection share by update scale, e.g. ``1x:62% 0.5x:21% 2x:17%``."""
    total = sum(counts.values())
    if total <= 0:
        return ""
    return " ".join(
        f"{_XM_LABELS.get(i, f's{i}')}:{100.0 * counts[i] / total:.0f}%"
        for i in sorted(counts)
    )


def online_retrain(
    model: GPT,
    arr: np.ndarray,
    *,
    end: int,
    cfg: XsaTttConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    lora: TTTLoRA | None,
    seed: int,
    steps_override: int | None = None,
    burst_state: dict | None = None,
    n_total: int | None = None,
) -> float:
    """NNCP retrain on decoded prefix ``arr[:end]``. Returns last CE (nats).

    With ``cfg.online_retrain_replay`` and ``steps > 1``, later steps train on
    deterministic older prefix windows (NNCP-style replay) instead of
    repeating the newest window.

    ``retrain_mode="replenish"`` swaps the window loss for the pretrain
    shape: CE on one ``online_retrain_every`` chunk only, with the preceding
    ``block_size`` bytes as context — the exact stream-pretrain step. The
    final step always trains the just-coded chunk (primes the next one).
    With ``steps > 1``: replay on (default) draws the earlier steps as
    deterministic uniform chunks from the whole decoded prefix (spaced
    multi-pass, like pretrain epochs); replay off walks back over the last
    ``steps`` chunks (massed). Uses the steady-state optimizer; set pretrain
    hypers via ``XSA_RETRAIN_LR`` / ``XSA_TTT_BETA1/2`` /
    ``XSA_TTT_WEIGHT_DECAY``; toggle replay via ``XSA_RETRAIN_REPLAY``.

    With ``cfg.ttt_replenish_xm_k >= 2`` (replenish mode, non-warmup
    boundaries) the whole boundary is handled by Forward-XM instead: K
    candidate single-step updates are explored from a snapshot, scored on a
    fixed prefix probe, and only the best is kept (``_xm_replenish_boundary``).

    Burst steps (``steps`` above the steady-state schedule, from
    ``cfg.ttt_burst_schedule``) are *replenishment*, not one-shot polish:
    they run on a pretrain-style AdamW (``ttt_burst_*`` hypers) and walk the
    prefix sequentially oldest -> newest (multi-epoch wrap), ending on the
    newest window. The steady-state optimizer is left untouched. Pass a
    mutable ``burst_state`` dict to persist the burst optimizer (its Adam
    second-moment preconditioner) across bursts — deterministic on both codec
    sides since bursts fire at identical boundaries.
    """
    steps = max(
        0,
        int(cfg.online_retrain_steps if steps_override is None else steps_override),
    )
    if steps == 0 or end < 2:
        return float("nan")
    torch.manual_seed(int(seed) + int(end))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed) + int(end))

    xy = _window_xy(arr, end=end, block_size=cfg.block_size, device=device)
    if xy is None:
        return float("nan")
    x, y = xy
    replay = bool(getattr(cfg, "online_retrain_replay", True)) and steps > 1
    last = float("nan")
    mode = (cfg.retrain_mode or "full").lower()
    # ``replenish``: pretrain-shape steps — CE on the chunk just coded only,
    # with the preceding block as context (identical to a stream-pretrain
    # step at this position). Replay/burst paths do not apply.
    is_replenish = mode == "replenish"
    lr_eff = 0.0

    if is_replenish:
        # Dose ramp: allow one extra pass per ``ramp`` decoded chunks, so
        # early boundaries never re-epoch a tiny prefix (the pretrain
        # analogue: visit density grows with wall-clock, it doesn't start
        # at maximum). ramp=0 disables. Deterministic in ``end``.
        ramp = max(0, int(getattr(cfg, "ttt_replenish_ramp", 0)))
        if ramp > 0:
            every_r = max(1, int(cfg.online_retrain_every))
            allowed = 1 + max(0, int(end) // every_r - 1) // ramp
            if allowed < steps:
                steps = allowed
        # Anneal: once the wound is healed, keep lowering the noise floor.
        # "linear": lr = lr0 * (1 - pos/total), anchored to
        # ttt_replenish_total_bytes (or n_total); "exp": lr = lr0 *
        # 2^(-pos/half). Pass count tapers by the same factor. Optional
        # warmup then overrides lr→0 for the first N boundaries.
        # Deterministic in ``end`` so encode and decode stay lockstep.
        anneal_mode = str(getattr(cfg, "ttt_replenish_anneal", "exp")).lower()
        half_mb = float(getattr(cfg, "ttt_replenish_lr_half_mb", 0.0))
        factor: float | None = None
        if anneal_mode == "linear" and n_total:
            # Probes can anchor the anneal to the full corpus length so the
            # head runs the full-run schedule prefix instead of cooling to
            # the floor within the probe (XSA_REPLENISH_TOTAL_BYTES).
            total = (
                int(getattr(cfg, "ttt_replenish_total_bytes", 0)) or int(n_total)
            )
            factor = max(0.0, 1.0 - float(end) / float(total))
        elif half_mb > 0:
            factor = 2.0 ** (-float(end) / (half_mb * 1e6))
        if factor is not None:
            steps_min = max(
                1, int(getattr(cfg, "ttt_replenish_steps_min", 1))
            )
            taper = max(steps_min, int(math.ceil(steps * factor)))
            if taper < steps:
                steps = taper
        # Always (re)set the schedule LR so a prior warmup's lr=0 cannot stick.
        lr_eff = _scheduled_replenish_lr(cfg, int(end), n_total=n_total)
        for group in optimizer.param_groups:
            group["lr"] = lr_eff
        replay = (
            bool(getattr(cfg, "online_retrain_replay", True)) and steps > 1
        )

    is_burst = (
        mode != "lora"
        and not is_replenish
        and steps > max(1, int(cfg.online_retrain_steps))
        and bool(str(getattr(cfg, "ttt_burst_schedule", "") or ""))
    )
    burst_optimizer: torch.optim.Optimizer | None = None
    burst_boundaries: list[int] = []
    if is_burst:
        every = max(1, int(cfg.online_retrain_every))
        burst_boundaries = list(range(every, int(end) + 1, every)) or [int(end)]
        if burst_state is not None:
            burst_optimizer = burst_state.get("optimizer")
        if burst_optimizer is None:
            burst_optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(getattr(cfg, "ttt_burst_lr", 3e-4)),
                betas=(
                    float(getattr(cfg, "ttt_burst_beta1", 0.9)),
                    float(getattr(cfg, "ttt_burst_beta2", 0.999)),
                ),
                weight_decay=float(getattr(cfg, "ttt_burst_weight_decay", 0.01)),
            )
            if burst_state is not None:
                burst_state["optimizer"] = burst_optimizer
    was_training = model.training
    prev_ckpt = bool(getattr(model, "gradient_checkpointing", True))
    # Full 16k retrain under math-SDP needs activation checkpointing + bf16.
    if mode != "lora" and int(cfg.block_size) >= 8192:
        model.gradient_checkpointing = True
    model.train()
    if lora is not None:
        lora.train()
    use_bf16 = bool(getattr(cfg, "use_bf16", False))
    anchor = _replenish_anchor_state(model, cfg, device) if is_replenish else None
    xm_k = (
        int(getattr(cfg, "ttt_replenish_xm_k", 0)) if is_replenish else 0
    )
    if xm_k >= 2 and lr_eff > 0.0:
        # Forward-XM boundary: explore K candidate updates, keep the one
        # with the best fixed-prefix probe CE. Warmup boundaries (lr=0)
        # fall through to the standard path so Adam moments still build.
        try:
            with torch.enable_grad():
                last = _xm_replenish_boundary(
                    model,
                    arr,
                    end=int(end),
                    cfg=cfg,
                    device=device,
                    optimizer=optimizer,
                    lora=lora,
                    seed=int(seed),
                    lr_eff=lr_eff,
                    anchor=anchor,
                    use_bf16=use_bf16,
                )
        finally:
            model.gradient_checkpointing = prev_ckpt
            model.eval()
            if lora is not None:
                lora.eval()
            if was_training:
                model.train()
        return last
    accum = is_replenish and steps > 1 and bool(
        getattr(cfg, "ttt_replenish_accum", False)
    )
    replay_lr_scale = float(
        getattr(cfg, "ttt_replenish_replay_lr_scale", 1.0)
    )
    base_lrs = [float(g["lr"]) for g in optimizer.param_groups]
    if accum:
        optimizer.zero_grad(set_to_none=True)
    try:
        with torch.enable_grad():
            cur_end = int(end)
            for step_index in range(steps):
                if is_replenish:
                    every_r = max(1, int(cfg.online_retrain_every))
                    if replay and step_index < steps - 1:
                        # Spaced replay: earlier steps train a deterministic
                        # chunk drawn from the decoded prefix. Uniform over
                        # the whole stream reproduces multi-pass pretrain;
                        # recent_mb > 0 restricts draws to the trailing
                        # window (NNCP-style local adaptation to the
                        # current content region).
                        n_chunks = max(1, int(end) // every_r)
                        lo_chunk = 1
                        recent_mb = float(
                            getattr(
                                cfg, "ttt_replenish_replay_recent_mb", 0.0
                            )
                        )
                        if recent_mb > 0:
                            span = max(
                                1, int(recent_mb * 1e6) // every_r
                            )
                            lo_chunk = max(1, n_chunks - span + 1)
                        mix = (
                            int(seed) * 1_000_003 + int(end) * 31
                        ) ^ (int(step_index) * 0x9E3779B1)
                        rng = np.random.default_rng(mix & 0x7FFF_FFFF_FFFF)
                        c_end = (
                            int(rng.integers(lo_chunk, n_chunks + 1)) * every_r
                        )
                    else:
                        # Final step (or replay off): walk-back ending on the
                        # just-coded chunk so the model is primed for the
                        # next one. Deterministic: depends on (end, every).
                        c_end = int(end) - (steps - 1 - step_index) * every_r
                    if c_end < 2:
                        continue
                    c_start = max(0, c_end - every_r)
                    if not accum:
                        optimizer.zero_grad(set_to_none=True)
                    loss = chunk_tf_train_loss(
                        model,
                        arr,
                        chunk_start=c_start,
                        chunk_end=c_end,
                        block_size=int(cfg.block_size),
                        device=device,
                        use_bf16=use_bf16,
                        lora=lora,
                    )
                    if accum:
                        # NNCP-shape: average the gradient over all windows
                        # of this boundary, single optimizer step at the end.
                        (loss / float(steps)).backward()
                        if step_index == steps - 1:
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), cfg.grad_clip
                            )
                            optimizer.step()
                            optimizer.zero_grad(set_to_none=True)
                            _apply_replenish_anchor(anchor)
                        last = float(loss.detach().float().item())
                        del loss
                        continue
                    if replay_lr_scale != 1.0:
                        # Replay windows step at a scaled LR; the final
                        # (fresh-chunk) step restores the annealed base LR.
                        fresh = step_index == steps - 1
                        for g, lr0 in zip(optimizer.param_groups, base_lrs):
                            g["lr"] = lr0 if fresh else lr0 * replay_lr_scale
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.grad_clip
                    )
                    optimizer.step()
                    _apply_replenish_anchor(anchor)
                    last = float(loss.detach().float().item())
                    del loss
                    continue
                if is_burst:
                    # Sequential epoch walk oldest -> newest (wraps); the
                    # final step returns to the newest window so the model
                    # is primed for the next chunk.
                    if step_index == steps - 1:
                        end_k = int(end)
                    else:
                        end_k = burst_boundaries[
                            step_index % len(burst_boundaries)
                        ]
                    if end_k != cur_end:
                        xy_k = _window_xy(
                            arr,
                            end=end_k,
                            block_size=cfg.block_size,
                            device=device,
                        )
                        if xy_k is not None:
                            x, y = xy_k
                            cur_end = end_k
                elif replay and step_index > 0:
                    end_k = _replay_window_end(
                        end=int(end),
                        step_index=step_index,
                        seed=int(seed),
                        block_size=int(cfg.block_size),
                    )
                    xy_k = _window_xy(
                        arr, end=end_k, block_size=cfg.block_size, device=device
                    )
                    if xy_k is not None:
                        x, y = xy_k
                opt = burst_optimizer if is_burst else optimizer
                opt.zero_grad(set_to_none=True)
                if lora is not None:
                    # mode=lora: adapters train on a frozen base. mode=full/
                    # replenish with a shipped healer LoRA: the base trains
                    # through the adapter forward so train == serve.
                    with _amp_ctx(device, use_bf16):
                        loss = ce_with_lora(model, x, y, lora)
                else:
                    with _amp_ctx(device, use_bf16):
                        loss = model(x, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    (
                        lora.parameters()
                        if mode == "lora" and lora is not None
                        else model.parameters()
                    ),
                    cfg.grad_clip,
                )
                opt.step()
                last = float(loss.detach().float().item())
                del loss
    finally:
        model.gradient_checkpointing = prev_ckpt
        model.eval()
        if lora is not None:
            lora.eval()
        if was_training:
            model.train()
    return last


@torch.no_grad()
def predict_next_probs(
    model: GPT,
    prefix: bytes | bytearray | memoryview | np.ndarray | list,
    *,
    block_size: int,
    device: torch.device,
    lora: TTTLoRA | None = None,
    use_bf16: bool = False,
    vocab_size: int = 256,
) -> np.ndarray:
    """P(next symbol | prefix) as ``vocab_size`` floats."""
    v = int(vocab_size)
    if isinstance(prefix, np.ndarray):
        n = int(prefix.shape[0])
        ctx = prefix[max(0, n - block_size) : n]
        ids = torch.tensor(np.asarray(ctx, dtype=np.int64), device=device).unsqueeze(0)
    elif isinstance(prefix, list):
        n = len(prefix)
        ctx = prefix[max(0, n - block_size) : n]
        if not ctx:
            return uniform_probs(v)
        ids = torch.tensor(ctx, dtype=torch.long, device=device).unsqueeze(0)
    else:
        n = len(prefix)
        ctx = bytes(prefix[max(0, n - block_size) : n])
        if not ctx:
            return uniform_probs(v)
        ids = torch.tensor(list(ctx), dtype=torch.long, device=device).unsqueeze(0)
    if ids.numel() == 0:
        return uniform_probs(v)
    with _amp_ctx(device, use_bf16):
        logits = forward_logits_with_lora(model, ids, lora)
    return _softmax_np(logits[0, -1], vocab_size=v)


def make_probs_fn(
    model: GPT,
    arr: np.ndarray,
    *,
    cfg: XsaTttConfig,
    device: torch.device,
    online_retrain_enabled: bool = True,
    lora: TTTLoRA | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    stats: dict[str, Any] | None = None,
    mode: Literal["encode", "decode"] = "encode",
    prob_cache: dict[tuple[int, int], np.ndarray] | None = None,
):
    """Build AC ``probs_fn(i, prefix)`` with optional NNCP retrain.

    Encode and decode share one chunked TF definition. Encode writes each
    chunk's rows into ``prob_cache`` (in-memory); decode reads them back so
    roundtrip does not re-forward. Without a cache, decode falls back to the
    padded chunk TF (same shapes → lockstep, but slower).
    """
    every = int(cfg.online_retrain_every)
    block = int(cfg.block_size)
    seed = int(cfg.seed)
    use_bf16 = bool(cfg.use_bf16)
    vocab_size = int(cfg.vocab_size)
    n_total = int(arr.shape[0])
    st = stats if stats is not None else {}
    st.setdefault("retrain_count", 0)
    st.setdefault("retrain_ce_sum", 0.0)
    st["infer_mode"] = (
        "chunked_tf+memcache" if prob_cache is not None else "chunked_tf_lockstep"
    )
    st["vocab_size"] = vocab_size

    chunk_probs: np.ndarray | None = None
    chunk_start = -1
    chunk_end = -1
    probs_buf = np.empty(vocab_size, dtype=np.float32)
    decode_buf = np.zeros(n_total, dtype=_sym_dtype(vocab_size))
    # Decode with a full encode cache can skip GPU entirely.
    cache_only = mode == "decode" and prob_cache is not None
    uni = uniform_probs(vocab_size, dtype=np.float32)

    def _prefix_as_arr(prefix, upto: int) -> np.ndarray:
        if isinstance(prefix, np.ndarray):
            return np.asarray(prefix[:upto], dtype=_sym_dtype(vocab_size))
        if isinstance(prefix, list):
            return np.asarray(prefix[:upto], dtype=_sym_dtype(vocab_size))
        if vocab_size > 256:
            raise TypeError("token AC prefix must be ndarray or list")
        return np.frombuffer(bytes(prefix[:upto]), dtype=np.uint8).copy()

    # Burst optimizer persists across bursts within one stream (encode and
    # decode each build their own probs_fn, so both sides warm it identically).
    burst_state: dict = {}

    def _maybe_retrain(i: int, prefix) -> None:
        nonlocal chunk_probs, chunk_start, chunk_end
        scheduled_steps = _online_steps_at(i, cfg, every)
        if cache_only:
            return
        if not (
            online_retrain_enabled
            and optimizer is not None
            and i > 0
            and scheduled_steps > 0
        ):
            return
        local = _prefix_as_arr(prefix, i) if len(prefix) >= i else arr[:i]
        ce = online_retrain(
            model,
            local,
            end=i,
            cfg=cfg,
            device=device,
            optimizer=optimizer,
            lora=lora,
            seed=seed,
            steps_override=scheduled_steps,
            burst_state=burst_state,
            n_total=n_total,
        )
        st["retrain_count"] += 1
        if math.isfinite(ce):
            st["retrain_ce_sum"] += ce
        chunk_probs = None
        chunk_start = -1
        chunk_end = -1

    def _run_chunk(data: np.ndarray, start: int, end: int) -> np.ndarray:
        return _chunk_tf_probs(
            model,
            data,
            start=start,
            end=end,
            block_size=block,
            device=device,
            lora=lora,
            use_bf16=use_bf16,
        )

    def _store_cache(start: int, end: int, rows: np.ndarray) -> None:
        if prob_cache is not None:
            # Copy so later in-place edits cannot corrupt the shared store.
            prob_cache[(start, end)] = np.array(rows, dtype=np.float32, copy=True)

    def _log_chunk_bpb(start: int, end: int, rows: np.ndarray) -> None:
        # Free per-chunk rate curve (encode-side only, no lockstep impact):
        # the exact bits the AC pays for this chunk, logged for analysis.
        syms = np.asarray(arr[start:end], dtype=np.int64)
        if syms.size == 0 or rows.shape[0] < syms.size:
            return
        p = rows[np.arange(syms.size), syms].astype(np.float64)
        bits = float(-np.log2(np.maximum(p, 1e-12)).sum())
        rows_log = st.setdefault("chunk_bpb_rows", [])
        rows_log.append({"pos": int(end), "bpb": bits / syms.size})
        st["bpb_bits_sum"] = st.get("bpb_bits_sum", 0.0) + bits
        st["bpb_sym_sum"] = st.get("bpb_sym_sum", 0) + int(syms.size)
        # Live rate readout ~every MiB so long encodes are observable.
        win = max(1, (1 << 20) // max(1, every))
        if len(rows_log) % win == 0:
            recent = rows_log[-win:]
            recent_bpb = sum(r["bpb"] for r in recent) / len(recent)
            cum_bpb = st["bpb_bits_sum"] / max(1, st["bpb_sym_sum"])
            lr_now = _scheduled_replenish_lr(cfg, int(end), n_total=n_total)
            msg = (
                f"[ac] pos={end / 1e6:.1f}MB "
                f"cum_bpb={cum_bpb:.4f} last{win}ch={recent_bpb:.4f} "
                f"lr={lr_now:.3e}"
            )
            try:
                from tqdm import tqdm

                tqdm.write(msg, file=sys.stderr)
            except Exception:
                print(msg, file=sys.stderr, flush=True)

    def probs_fn(i: int, prefix) -> np.ndarray:
        nonlocal chunk_probs, chunk_start, chunk_end
        _maybe_retrain(i, prefix)
        if i == 0:
            probs_buf[:] = uni
            return probs_buf

        # Use chunked TF for base and LoRA. (Older code forced per-token
        # predict_next_probs whenever lora!=None — ~100× slower.) Retrain
        # boundaries already clear chunk_probs in _maybe_retrain.
        start, end = _chunk_bounds(i, every=every, n_total=n_total)

        # Decode (or encode replay): prefer in-memory chunk rows from encode.
        if prob_cache is not None and mode == "decode":
            cached = prob_cache.get((start, end))
            if cached is not None:
                probs_buf[:] = cached[i - start]
                return probs_buf

        if mode == "encode":
            if chunk_probs is None or chunk_start != start or chunk_end != end:
                chunk_probs = _run_chunk(arr, start, end)
                chunk_start = start
                chunk_end = end
                _store_cache(start, end, chunk_probs)
                _log_chunk_bpb(start, end, chunk_probs)
            assert chunk_probs is not None
            probs_buf[:] = chunk_probs[i - start]
            return probs_buf

        # Slow fallback: padded chunk TF (same shapes as encode, no shared cache).
        decode_buf[start:end] = 0
        if i > 0:
            decode_buf[:i] = _prefix_as_arr(prefix, i)
        rows = _run_chunk(decode_buf, start, end)
        probs_buf[:] = rows[i - start]
        return probs_buf

    return probs_fn


def compress_bytes(
    model: GPT,
    data: bytes | np.ndarray,
    *,
    cfg: XsaTttConfig,
    device: torch.device,
    online_retrain_enabled: bool = True,
    progress: bool = True,
    prob_cache: dict[tuple[int, int], np.ndarray] | None = None,
) -> dict[str, Any]:
    """AC-encode ``data``; return payload + metrics.

    When ``prob_cache`` is omitted, a fresh in-memory cache is created and
    returned as ``result['prob_cache']`` for a fast lockstep decode.
    """
    vocab_size = int(cfg.vocab_size)
    arr = _as_symbol_array(data, vocab_size=vocab_size)
    if vocab_size <= 256:
        symbols: bytes | np.ndarray = (
            bytes(data)
            if isinstance(data, (bytes, bytearray))
            else (
                arr.tobytes() if not isinstance(arr, np.memmap) else bytes(arr)
            )
        )
    else:
        symbols = arr

    lora, optimizer = _init_online_state(model, cfg, device, online_retrain_enabled)
    if prob_cache is None:
        prob_cache = {}

    model.eval()
    stats: dict[str, Any] = {}
    probs_fn = make_probs_fn(
        model,
        arr,
        cfg=cfg,
        device=device,
        online_retrain_enabled=online_retrain_enabled,
        lora=lora,
        optimizer=optimizer,
        stats=stats,
        mode="encode",
        prob_cache=prob_cache,
    )
    desc = "AC encode" if progress else None
    payload = encode_with_probs(
        symbols, probs_fn, desc=desc, alphabet_size=vocab_size
    )
    n = int(arr.shape[0])
    payload_bits = len(payload) * 8
    bpb = payload_bits / max(1, n)
    return {
        "payload": payload,
        "n_bytes": n,  # symbol count (tokens if vocab > 256)
        "n_symbols": n,
        "payload_bytes": len(payload),
        "bpb": bpb,  # bits per symbol; see annotate_source_bpb()
        "bits_per_symbol": bpb,
        "vocab_size": vocab_size,
        "retrain_count": int(stats.get("retrain_count", 0)),
        "retrain_ce_mean": (
            float(stats["retrain_ce_sum"]) / max(1, int(stats["retrain_count"]))
            if stats.get("retrain_count")
            else float("nan")
        ),
        "infer_mode": str(stats.get("infer_mode", "chunked_tf+memcache")),
        "sha256": hashlib.sha256(_symbols_for_hash(arr)).hexdigest(),
        "chunk_bpb_rows": stats.get("chunk_bpb_rows") or [],
        "prob_cache": prob_cache,
    }


def annotate_source_bpb(
    meta: dict[str, Any],
    *,
    n_symbols_total: int | None = None,
    source_bytes: int | None = None,
) -> dict[str, Any]:
    """Add source-byte metrics so BPE curves compare fairly to V=256.

    ``meta['bpb']`` stays bits/symbol (legacy). New fields:
      - ``source_bpb``: bits per original ``payload_sim`` / byte-residual byte
      - ``est_full_payload_bytes``: probe rate × full symbol count
    """
    bits = float(meta.get("bits_per_symbol", meta.get("bpb", float("nan"))))
    meta["bits_per_symbol"] = bits
    n_sym = int(meta.get("n_symbols", meta.get("n_bytes", meta.get("ac_bytes", 0))))
    meta["n_symbols"] = n_sym
    vocab = int(meta.get("vocab_size", 256))

    if source_bytes is not None and int(source_bytes) > 0:
        meta["source_bytes"] = int(source_bytes)
    if n_symbols_total is not None and int(n_symbols_total) > 0:
        meta["n_symbols_total"] = int(n_symbols_total)

    src = meta.get("source_bytes")
    n_tot = meta.get("n_symbols_total")
    if src and n_tot and int(n_tot) > 0:
        bps = float(src) / float(n_tot)
        meta["bytes_per_symbol"] = bps
        meta["source_bpb"] = bits / bps
        meta["est_full_payload_bytes"] = int(round(float(n_tot) * bits / 8.0))
    elif vocab <= 256:
        # Byte LM: symbol == source byte.
        meta["bytes_per_symbol"] = 1.0
        meta["source_bpb"] = bits
        if n_tot:
            meta["est_full_payload_bytes"] = int(round(float(n_tot) * bits / 8.0))
        elif n_sym:
            meta["est_full_payload_bytes"] = int(meta.get("payload_bytes", 0))
    return meta


def _seg_roundtrip_mismatch(
    seg: bytes | np.ndarray,
    seg_decoded: bytes | list[int],
    *,
    vocab_size: int,
    start: int,
) -> int | None:
    """Return absolute first mismatch index, or None if equal."""
    if vocab_size <= 256:
        if seg_decoded == seg:
            return None
        for j, (a, b) in enumerate(zip(seg_decoded, seg)):
            if int(a) != int(b):
                return start + j
        return start
    src = np.asarray(seg, dtype=np.int64).reshape(-1)
    dec = np.asarray(seg_decoded, dtype=np.int64).reshape(-1)
    if src.shape == dec.shape and np.array_equal(src, dec):
        return None
    n = min(src.shape[0], dec.shape[0])
    for j in range(n):
        if int(src[j]) != int(dec[j]):
            return start + j
    return start


def compress_full_sha_lockstep(
    model: GPT,
    data: bytes | np.ndarray,
    *,
    cfg: XsaTttConfig,
    device: torch.device,
    online_retrain_enabled: bool = True,
    progress: bool = True,
    verify_decode: bool = False,
    verify_end: bool = False,
    row_batch: int = 1024,
    decoded_path: str | Path | None = None,
) -> dict[str, Any]:
    """Full-corpus AC with per-segment streams and optional SHA roundtrip.

    Default is **encode-only** (payload + bpb + source SHA). Set
    ``verify_decode=True`` (or env ``AC_VERIFY_DECODE=1``) for per-segment
    decode+SHA. ``verify_end=True`` replays a decode pass after encode.

    When verifying and ``decoded_path`` is set, append each decoded segment to
    that file as it is produced (flushed), then report the path in the meta.
    """
    import struct

    vocab_size = int(cfg.vocab_size)
    arr = _as_symbol_array(data, vocab_size=vocab_size)
    if vocab_size <= 256:
        symbols: bytes | np.ndarray = (
            bytes(data)
            if isinstance(data, (bytes, bytearray))
            else (
                arr.tobytes() if not isinstance(arr, np.memmap) else bytes(arr)
            )
        )
    else:
        symbols = arr

    n = int(arr.shape[0])
    every = max(1, int(cfg.online_retrain_every))
    block = int(cfg.block_size)
    seed = int(cfg.seed)
    use_bf16 = bool(cfg.use_bf16)
    do_seg_verify = bool(verify_decode)
    do_end_verify = bool(verify_end) and not do_seg_verify
    lora, optimizer = _init_online_state(model, cfg, device, online_retrain_enabled)
    burst_state: dict = {}

    model.eval()
    src_sha = hashlib.sha256(_symbols_for_hash(arr)).hexdigest()
    init_state = None
    if do_end_verify:
        init_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }
    dec_hasher = hashlib.sha256() if do_seg_verify else None
    out_parts = bytearray()
    seg_payloads: list[bytes] = []
    seg_rate_rows: list[dict[str, Any]] = []
    payload_bytes = 0
    retrain_count = 0
    retrain_ce_sum = 0.0
    xm_counts: dict[int, int] = {}
    first_mismatch: int | None = None
    decoded_out: Path | None = (
        Path(decoded_path) if decoded_path and (do_seg_verify or do_end_verify) else None
    )
    decoded_bytes_written = 0
    dec_fp = None

    def _write_decoded(seg_decoded) -> None:
        nonlocal decoded_bytes_written
        if dec_fp is None:
            return
        blob = _seg_decoded_to_bytes(seg_decoded, vocab_size=vocab_size)
        dec_fp.write(blob)
        dec_fp.flush()
        decoded_bytes_written += len(blob)

    mode_tag = (
        "encode+decode"
        if do_seg_verify
        else ("encode+end-verify" if do_end_verify else "encode-only")
    )
    indices = range(0, n, every)
    if progress:
        from tqdm import tqdm

        print(
            f"[AC full-SHA] starting ({n:,} sym, segment={every}, V={vocab_size}, "
            f"{mode_tag})…",
            file=sys.stderr,
            flush=True,
        )
        if decoded_out is not None:
            print(
                f"[AC full-SHA] writing decoded stream → {decoded_out}",
                file=sys.stderr,
                flush=True,
            )
        indices = tqdm(
            indices,
            total=(n + every - 1) // every,
            desc="AC full-SHA",
            unit="seg",
            file=sys.stderr,
            dynamic_ncols=True,
            mininterval=1.0,
            disable=False,
        )

    try:
        if decoded_out is not None and do_seg_verify:
            decoded_out.parent.mkdir(parents=True, exist_ok=True)
            dec_fp = open(decoded_out, "wb")

        for start in indices:
            scheduled_steps = _online_steps_at(start, cfg, every)
            if (
                online_retrain_enabled
                and optimizer is not None
                and start > 0
                and scheduled_steps > 0
            ):
                ce = online_retrain(
                    model,
                    arr,
                    end=start,
                    cfg=cfg,
                    device=device,
                    optimizer=optimizer,
                    lora=lora,
                    seed=seed,
                    steps_override=scheduled_steps,
                    burst_state=burst_state,
                    n_total=n,
                )
                retrain_count += 1
                if math.isfinite(ce):
                    retrain_ce_sum += ce
                model.eval()
                # Forward-XM diagnostics: running share of which candidate
                # type won each boundary, shown live on the progress bar.
                xm_last = getattr(model, "_xm_last", None)
                if (
                    xm_last is not None
                    and int(xm_last.get("end", -1)) == int(start)
                ):
                    chosen = int(xm_last.get("chosen", 0))
                    xm_counts[chosen] = xm_counts.get(chosen, 0) + 1
                    if progress and hasattr(indices, "set_postfix_str"):
                        indices.set_postfix_str(
                            _xm_share_str(xm_counts), refresh=False
                        )

            end = min(n, start + every)
            seg = symbols[start:end]
            if do_seg_verify:
                rows = _chunk_tf_probs(
                    model,
                    arr,
                    start=start,
                    end=end,
                    block_size=block,
                    device=device,
                    lora=lora,
                    use_bf16=use_bf16,
                    dtype=np.float32,
                    row_batch=row_batch,
                )
                seg_payload, seg_decoded = encode_decode_segment(
                    seg, rows, alphabet_size=vocab_size
                )
                if first_mismatch is None:
                    first_mismatch = _seg_roundtrip_mismatch(
                        seg, seg_decoded, vocab_size=vocab_size, start=start
                    )
                assert dec_hasher is not None
                dec_blob = _seg_decoded_to_bytes(
                    seg_decoded, vocab_size=vocab_size
                )
                dec_hasher.update(dec_blob)
                _write_decoded(seg_decoded)
            else:
                seg_payload = _encode_segment_tf(
                    model,
                    arr,
                    seg,
                    start=start,
                    end=end,
                    block_size=block,
                    device=device,
                    lora=lora,
                    use_bf16=use_bf16,
                    alphabet_size=vocab_size,
                    row_batch=row_batch,
                )
            if do_end_verify:
                seg_payloads.append(seg_payload)
            out_parts.extend(struct.pack("<Q", len(seg_payload)))
            out_parts.extend(seg_payload)
            payload_bytes += len(seg_payload)
            # Exact paid rate per segment (includes AC overhead) + live
            # readout ~every MiB so multi-hour encodes are observable.
            seg_rate_rows.append(
                {"pos": int(end), "bpb": len(seg_payload) * 8.0 / max(1, end - start)}
            )
            win = max(1, (1 << 20) // every)
            if progress and len(seg_rate_rows) % win == 0:
                recent = seg_rate_rows[-win:]
                recent_bpb = sum(r["bpb"] for r in recent) / len(recent)
                cum_bpb = payload_bytes * 8.0 / max(1, end)
                lr_now = _scheduled_replenish_lr(cfg, int(end), n_total=n)
                xm_note = (
                    f" xm[{_xm_share_str(xm_counts)}]" if xm_counts else ""
                )
                from tqdm import tqdm as _tqdm

                _tqdm.write(
                    f"[ac] pos={end / 1e6:.1f}MB cum_bpb={cum_bpb:.4f} "
                    f"last{win}seg={recent_bpb:.4f} lr={lr_now:.3e}"
                    + xm_note,
                    file=sys.stderr,
                )

        dec_sha: str | None = None
        sha_ok: bool | None = None
        if do_seg_verify:
            assert dec_hasher is not None
            dec_sha = dec_hasher.hexdigest()
            sha_ok = dec_sha == src_sha and first_mismatch is None
        elif do_end_verify:
            assert init_state is not None
            model.load_state_dict(init_state, strict=True)
            model.to(device)
            model.eval()
            lora, optimizer = _init_online_state(
                model, cfg, device, online_retrain_enabled
            )
            # Decoder-side burst optimizer starts cold, like a real decode.
            burst_state = {}
            dec_hasher = hashlib.sha256()
            first_mismatch = None
            if decoded_out is not None:
                if dec_fp is not None:
                    dec_fp.close()
                    dec_fp = None
                decoded_out.parent.mkdir(parents=True, exist_ok=True)
                dec_fp = open(decoded_out, "wb")
                decoded_bytes_written = 0
            if progress:
                from tqdm import tqdm

                print(
                    f"[AC full-SHA] end-verify decode ({len(seg_payloads)} segments)…",
                    file=sys.stderr,
                    flush=True,
                )
                if decoded_out is not None:
                    print(
                        f"[AC full-SHA] writing decoded stream → {decoded_out}",
                        file=sys.stderr,
                        flush=True,
                    )
                verify_iter = tqdm(
                    enumerate(seg_payloads),
                    total=len(seg_payloads),
                    desc="AC end-verify",
                    unit="seg",
                    file=sys.stderr,
                    dynamic_ncols=True,
                    mininterval=1.0,
                    disable=False,
                )
            else:
                verify_iter = enumerate(seg_payloads)

            for seg_i, seg_payload in verify_iter:
                start = seg_i * every
                scheduled_steps = _online_steps_at(start, cfg, every)
                if (
                    online_retrain_enabled
                    and optimizer is not None
                    and start > 0
                    and scheduled_steps > 0
                ):
                    ce = online_retrain(
                        model,
                        arr,
                        end=start,
                        cfg=cfg,
                        device=device,
                        optimizer=optimizer,
                        lora=lora,
                        seed=seed,
                        steps_override=scheduled_steps,
                        burst_state=burst_state,
                        n_total=n,
                    )
                    if math.isfinite(ce):
                        pass  # retrain already counted on encode pass
                    model.eval()
                end = min(n, start + every)
                seg = symbols[start:end]
                rows = _chunk_tf_probs(
                    model,
                    arr,
                    start=start,
                    end=end,
                    block_size=block,
                    device=device,
                    lora=lora,
                    use_bf16=use_bf16,
                    dtype=np.float32,
                    row_batch=row_batch,
                )

                def probs_fn(i: int, _prefix, _rows=rows):
                    return _rows[i]

                seg_decoded = decode_with_probs(
                    seg_payload,
                    end - start,
                    probs_fn,
                    desc=None,
                    alphabet_size=vocab_size,
                )
                if first_mismatch is None:
                    first_mismatch = _seg_roundtrip_mismatch(
                        seg, seg_decoded, vocab_size=vocab_size, start=start
                    )
                dec_blob = _seg_decoded_to_bytes(
                    seg_decoded, vocab_size=vocab_size
                )
                dec_hasher.update(dec_blob)
                _write_decoded(seg_decoded)
            dec_sha = dec_hasher.hexdigest()
            sha_ok = dec_sha == src_sha and first_mismatch is None
    finally:
        if dec_fp is not None:
            dec_fp.close()
            dec_fp = None

    bpb = (payload_bytes * 8) / max(1, n)
    return {
        "payload": bytes(out_parts),
        "n_bytes": n,
        "n_symbols": n,
        "payload_bytes": payload_bytes,
        "bpb": bpb,
        "bits_per_symbol": bpb,
        "vocab_size": int(cfg.vocab_size),
        "retrain_count": retrain_count,
        "retrain_ce_mean": (
            retrain_ce_sum / max(1, retrain_count) if retrain_count else float("nan")
        ),
        "infer_mode": "segmented_tf_sha",
        "chunk_bpb_rows": seg_rate_rows,
        "sha256": src_sha,
        "decoded_sha256": dec_sha,
        "sha256_ok": sha_ok,
        "roundtrip_ok": sha_ok,
        "first_mismatch": first_mismatch,
        "segment_bytes": every,
        "verify_decode": do_seg_verify,
        "verify_end": do_end_verify,
        "decoded_path": str(decoded_out) if decoded_out is not None else None,
        "decoded_bytes": decoded_bytes_written if decoded_out is not None else None,
    }


def decompress_payload(
    model: GPT,
    payload: bytes,
    n_bytes: int,
    *,
    cfg: XsaTttConfig,
    device: torch.device,
    online_retrain_enabled: bool = True,
    progress: bool = True,
    # Dummy array for encode-side shape; decode uses live prefix.
    arr: np.ndarray | None = None,
    prob_cache: dict[tuple[int, int], np.ndarray] | None = None,
) -> bytes | list[int]:
    vocab_size = int(cfg.vocab_size)
    if arr is None:
        arr = np.zeros(n_bytes, dtype=_sym_dtype(vocab_size))
    # With a full encode cache, decode is memory-only (skip LoRA/optim init).
    if prob_cache is not None:
        lora, optimizer = None, None
        online = False
    else:
        lora, optimizer = _init_online_state(
            model, cfg, device, online_retrain_enabled
        )
        online = online_retrain_enabled
    model.eval()
    probs_fn = make_probs_fn(
        model,
        arr,
        cfg=cfg,
        device=device,
        online_retrain_enabled=online,
        lora=lora,
        optimizer=optimizer,
        mode="decode",
        prob_cache=prob_cache,
    )
    desc = "AC decode" if progress else None
    return decode_with_probs(
        payload,
        n_bytes,
        probs_fn,
        desc=desc,
        alphabet_size=vocab_size,
    )


def _init_online_state(
    model: GPT,
    cfg: XsaTttConfig,
    device: torch.device,
    online_retrain_enabled: bool,
) -> tuple[TTTLoRA | None, torch.optim.Optimizer | None]:
    """Shared encode/decode LoRA + optimizer init (seeded for lockstep).

    If ``cfg.lora_path`` / ``XSA_LORA_PATH`` is set, load distilled adapters onto
    a frozen Init(seed) base. Online AdamW is only created when
    ``online_retrain_enabled`` is true (further TTT on top of distill).
    """
    import os

    from .ttt_lora import load_lora_adapters

    lora_path_s = getattr(cfg, "lora_path", None) or os.environ.get("XSA_LORA_PATH")
    lora_path = Path(lora_path_s) if lora_path_s else None
    mode = (cfg.retrain_mode or "full").lower()

    # Distilled LoRA on weightless init: always attach adapters when path set.
    if lora_path is not None:
        for p in model.parameters():
            p.requires_grad_(False)
        lora = load_lora_adapters(model, cfg, lora_path, device=device)
        if online_retrain_enabled and mode == "lora":
            return lora, make_ttt_optimizer(lora, cfg)
        if online_retrain_enabled:
            # Healer-LoRA arm: shipped adapters stay frozen while full-model
            # (full/replenish) TTT trains the base weights underneath them.
            for p in lora.parameters():
                p.requires_grad_(False)
            for p in model.parameters():
                p.requires_grad_(True)
            opt = torch.optim.AdamW(
                model.parameters(),
                lr=float(cfg.ttt_lora_lr),
                betas=(float(cfg.ttt_beta1), float(cfg.ttt_beta2)),
                weight_decay=float(cfg.ttt_weight_decay),
            )
            return lora, opt
        # Frozen distilled adapters (typical LTCB arm: |LoRA| + AC).
        return lora, None

    if not online_retrain_enabled:
        return None, None
    # Identical RNG for encode and decode so LoRA A matrices match.
    torch.manual_seed(int(cfg.seed) ^ 0xA5A51070)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(cfg.seed) ^ 0xA5A51070)
    if mode == "lora":
        for p in model.parameters():
            p.requires_grad_(False)
        lora = TTTLoRA(model, cfg).to(device)
        return lora, make_ttt_optimizer(lora, cfg)
    for p in model.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.ttt_lora_lr),
        betas=(float(cfg.ttt_beta1), float(cfg.ttt_beta2)),
        weight_decay=float(cfg.ttt_weight_decay),
    )
    return None, opt


def measure_teacher_forced_bpb(
    model: GPT,
    arr: np.ndarray,
    *,
    cfg: XsaTttConfig,
    device: torch.device,
    max_bytes: int | None = None,
    online_retrain_enabled: bool = True,
    progress: bool = True,
) -> dict[str, Any]:
    """Fast ideal-bpb probe (no AC bitstream) with optional online retrain."""
    n = int(arr.shape[0] if max_bytes is None else min(arr.shape[0], max_bytes))
    every = int(cfg.online_retrain_every)
    block = int(cfg.block_size)
    lora, optimizer = _init_online_state(model, cfg, device, online_retrain_enabled)
    burst_state: dict = {}

    model.eval()
    total_nll = 0.0
    count = 0
    retrain_count = 0
    pbar = None
    if progress:
        from tqdm import tqdm

        print(f"[TF probe] starting ({n:,} B)…", file=sys.stderr, flush=True)
        pbar = tqdm(
            total=n,
            desc="TF probe",
            unit="B",
            leave=True,
            file=sys.stderr,
            dynamic_ncols=True,
            mininterval=0.5,
            disable=False,
        )
    pos = 0
    try:
        while pos < n:
            scheduled_steps = _online_steps_at(pos, cfg, every)
            if (
                online_retrain_enabled
                and optimizer is not None
                and pos > 0
                and scheduled_steps > 0
            ):
                online_retrain(
                    model,
                    arr,
                    end=pos,
                    cfg=cfg,
                    device=device,
                    optimizer=optimizer,
                    lora=lora,
                    seed=cfg.seed,
                    steps_override=scheduled_steps,
                    burst_state=burst_state,
                    n_total=n,
                )
                retrain_count += 1
            end = min(n, pos + every if every > 0 else n)
            # Context: last block bytes before ``pos``, then predict pos..end-1
            ctx_start = max(0, pos - block)
            ctx = np.asarray(arr[ctx_start:end], dtype=np.int64)
            if ctx.shape[0] < 2:
                break
            ids = torch.tensor(ctx[:-1], dtype=torch.long, device=device).unsqueeze(0)
            targets = torch.tensor(ctx[1:], dtype=torch.long, device=device).unsqueeze(0)
            # Only accumulate NLL for positions in [pos, end)
            with torch.no_grad():
                logits = forward_logits_with_lora(model, ids, lora)
                log_probs = F.log_softmax(logits.float(), dim=-1)
                # Align: ids[j] predicts targets[j] = arr[ctx_start+j+1]
                # We want positions where ctx_start+j+1 is in [pos, end)
                first_pred = pos - ctx_start  # index into targets for arr[pos]
                if first_pred < 0:
                    first_pred = 0
                sl = log_probs[0, first_pred : first_pred + (end - pos)]
                tg = targets[0, first_pred : first_pred + (end - pos)]
                if sl.numel() == 0:
                    if pbar is not None:
                        pbar.update(end - pos)
                    pos = end
                    continue
                nll = -sl.gather(-1, tg.unsqueeze(-1)).squeeze(-1)
                chunk_nll = float(nll.sum().item())
                chunk_n = int(nll.numel())
                total_nll += chunk_nll
                count += chunk_n
                if pbar is not None:
                    live_bpb = (total_nll / max(1, count)) / math.log(2)
                    pbar.set_postfix(
                        bpb=f"{live_bpb:.3f}",
                        retrains=retrain_count,
                        refresh=False,
                    )
            if pbar is not None:
                pbar.update(end - pos)
            pos = end
    finally:
        if pbar is not None:
            pbar.close()

    nats = total_nll / max(1, count)
    bpb = nats / math.log(2)
    return {
        "n_bytes": count,
        "val_loss_nats": nats,
        "bpb": bpb,
        "retrain_count": retrain_count,
    }


def save_checkpoint(
    path: Path,
    model: GPT,
    cfg: XsaTttConfig,
    *,
    meta: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "cfg": cfg.__dict__,
        "meta": meta or {},
    }
    torch.save(payload, path)
    side = path.with_suffix(".json")
    side.write_text(
        json.dumps({"cfg": cfg.__dict__, "meta": meta or {}}, indent=2) + "\n"
    )


def load_checkpoint(
    path: Path, *, device: torch.device, cfg: XsaTttConfig | None = None
) -> tuple[GPT, XsaTttConfig]:
    from .config import XsaTttConfig as Cfg
    from .model import build_model

    blob = torch.load(path, map_location=device, weights_only=False)
    loaded_cfg = Cfg(**blob["cfg"]) if cfg is None else cfg
    model = build_model(loaded_cfg, device=device)
    model.load_state_dict(blob["model"])
    model.eval()
    return model, loaded_cfg
