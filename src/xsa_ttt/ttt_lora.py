"""LoRA adapters for NNCP-scheduled online TTT (large BatchedTTTLoRA, bsz=1)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from .model import apply_rotary_emb, _sdpa_causal

if TYPE_CHECKING:
    from .config import XsaTttConfig
    from .model import GPT


class LinearLoRA(nn.Module):
    """Single-batch LoRA with alpha/rank or rsLoRA (α/√r) scaling."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        *,
        alpha: float = 144.0,
        warm_start_a: bool = True,
        use_rslora: bool = False,
    ):
        super().__init__()
        self._bound = 1.0 / math.sqrt(in_features)
        if use_rslora:
            self._scale = float(alpha) / math.sqrt(float(rank))
        else:
            self._scale = float(alpha) / float(rank)
        self._warm_start_a = warm_start_a
        self.A = nn.Parameter(
            torch.empty(rank, in_features).uniform_(-self._bound, self._bound)
        )
        self.B = nn.Parameter(torch.zeros(out_features, rank))

    def reset(self) -> None:
        with torch.no_grad():
            if not self._warm_start_a:
                self.A.uniform_(-self._bound, self._bound)
            self.B.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in) → (..., out)
        return (x @ self.A.T @ self.B.T) * self._scale


def _lora_kwargs(cfg: "XsaTttConfig") -> dict[str, Any]:
    return {
        "rank": int(cfg.ttt_lora_rank),
        "alpha": float(cfg.ttt_lora_alpha),
        "warm_start_a": bool(cfg.ttt_warm_start_a),
        "use_rslora": bool(getattr(cfg, "ttt_use_rslora", False)),
    }


class TTTLoRA(nn.Module):
    """Per-slot Q/V/(K)/O/MLP + lm_head LoRA matching large-profile TTT_K/O/MLP flags."""

    def __init__(self, model: "GPT", cfg: "XsaTttConfig"):
        super().__init__()
        dim = model.qo_bank.shape[-1]
        vocab = model.tok_emb.num_embeddings
        if getattr(model, "looping_active", False):
            num_slots = len(model.encoder_indices) + len(model.decoder_indices)
        else:
            num_slots = len(model.blocks)
        kv_dim = model.blocks[0].attn.num_kv_heads * (
            dim // model.blocks[0].attn.num_heads
        )
        kw = _lora_kwargs(cfg)
        self.num_slots = int(num_slots)
        self.lm_head_lora = LinearLoRA(dim, vocab, **kw)
        self.q_loras = nn.ModuleList(
            [LinearLoRA(dim, dim, **kw) for _ in range(num_slots)]
        )
        self.v_loras = nn.ModuleList(
            [LinearLoRA(dim, kv_dim, **kw) for _ in range(num_slots)]
        )
        self.k_loras = (
            nn.ModuleList([LinearLoRA(dim, kv_dim, **kw) for _ in range(num_slots)])
            if cfg.ttt_k_lora
            else None
        )
        self.mlp_loras = (
            nn.ModuleList([LinearLoRA(dim, dim, **kw) for _ in range(num_slots)])
            if cfg.ttt_mlp_lora
            else None
        )
        self.o_loras = (
            nn.ModuleList([LinearLoRA(dim, dim, **kw) for _ in range(num_slots)])
            if cfg.ttt_o_lora
            else None
        )
        # False (default): O-LoRA(n_preattn). True: O-LoRA(attn_out_pre_proj
        # input y) so ΔW_O = scale·B@A is exact for weight-gap factorization.
        self.o_lora_on_y = bool(getattr(cfg, "ttt_o_lora_on_y", False))

    def reset(self) -> None:
        self.lm_head_lora.reset()
        for group in (
            self.q_loras,
            self.v_loras,
            self.k_loras,
            self.mlp_loras,
            self.o_loras,
        ):
            if group is None:
                continue
            for lora in group:
                lora.reset()


def _block_with_lora(
    model: "GPT",
    block,
    x: torch.Tensor,
    x0: torch.Tensor,
    lora: TTTLoRA,
    slot: int,
    q_w,
    k_w,
    v_w,
    out_w,
    up_w,
    down_w,
) -> torch.Tensor:
    mix = block.resid_mix.to(dtype=x.dtype)
    x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
    n = block.attn_norm(x_in) * block.ln_scale_factor
    attn = block.attn
    bsz, seqlen, dim = n.shape
    q_raw = F.linear(n, q_w.to(n.dtype)) + lora.q_loras[slot](n)
    q = q_raw.reshape(bsz, seqlen, attn.num_heads, attn.head_dim)
    k = F.linear(n, k_w.to(n.dtype))
    if lora.k_loras is not None:
        k = k + lora.k_loras[slot](n)
    k = k.reshape(bsz, seqlen, attn.num_kv_heads, attn.head_dim)
    v = (F.linear(n, v_w.to(n.dtype)) + lora.v_loras[slot](n)).reshape(
        bsz, seqlen, attn.num_kv_heads, attn.head_dim
    )
    q = F.rms_norm(q, (q.size(-1),))
    k = F.rms_norm(k, (k.size(-1),))
    cos, sin = attn.rotary(seqlen, n.device, q.dtype)
    q = apply_rotary_emb(q, cos, sin, attn.rope_dims)
    k = apply_rotary_emb(k, cos, sin, attn.rope_dims)
    q = q * attn.q_gain.to(dtype=q.dtype)[None, None, :, None]
    y = _sdpa_causal(q, k, v)
    if attn.use_xsa:
        y = attn._xsa_efficient(y, v)
    if attn.sparse_attn_gate:
        gate_in = n[..., : attn.gate_window].contiguous()
        g = torch.sigmoid(
            attn.sparse_attn_gate_scale
            * F.linear(gate_in, attn.attn_gate_w.to(n.dtype))
        )
        y = y * g[..., None]
    y = y.reshape(bsz, seqlen, dim)
    attn_out = F.linear(y, out_w.to(n.dtype))
    if lora.o_loras is not None:
        o_in = y if getattr(lora, "o_lora_on_y", False) else n
        attn_out = attn_out + lora.o_loras[slot](o_in)
    x_out = x_in + block.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
    mlp_n = block.mlp_norm(x_out) * block.ln_scale_factor
    mlp_out = block.mlp(mlp_n, up_w, down_w)
    if lora.mlp_loras is not None:
        mlp_out = mlp_out + lora.mlp_loras[slot](mlp_n)
    x_out = x_out + block.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * mlp_out
    return x_out


def _parallel_block_with_lora(
    model: "GPT",
    block_idx: int,
    lane0: torch.Tensor,
    lane1: torch.Tensor,
    x0: torch.Tensor,
    lora: TTTLoRA,
    slot: int,
    q_w,
    k_w,
    v_w,
    out_w,
    up_w,
    down_w,
):
    block = model.blocks[block_idx]
    mix = block.resid_mix.to(dtype=lane0.dtype)
    attn_read = mix[0][None, None, :] * lane0 + mix[1][None, None, :] * x0
    n = block.attn_norm(attn_read) * block.ln_scale_factor
    attn = block.attn
    bsz, seqlen, dim = n.shape
    q_raw = F.linear(n, q_w.to(n.dtype)) + lora.q_loras[slot](n)
    q = q_raw.reshape(bsz, seqlen, attn.num_heads, attn.head_dim)
    k = F.linear(n, k_w.to(n.dtype))
    if lora.k_loras is not None:
        k = k + lora.k_loras[slot](n)
    k = k.reshape(bsz, seqlen, attn.num_kv_heads, attn.head_dim)
    v = (F.linear(n, v_w.to(n.dtype)) + lora.v_loras[slot](n)).reshape(
        bsz, seqlen, attn.num_kv_heads, attn.head_dim
    )
    q = F.rms_norm(q, (q.size(-1),))
    k = F.rms_norm(k, (k.size(-1),))
    cos, sin = attn.rotary(seqlen, n.device, q.dtype)
    q = apply_rotary_emb(q, cos, sin, attn.rope_dims)
    k = apply_rotary_emb(k, cos, sin, attn.rope_dims)
    q = q * attn.q_gain.to(dtype=q.dtype)[None, None, :, None]
    y = _sdpa_causal(q, k, v)
    if attn.use_xsa:
        y = attn._xsa_efficient(y, v)
    if attn.sparse_attn_gate:
        gate_in = n[..., : attn.gate_window].contiguous()
        g = torch.sigmoid(
            attn.sparse_attn_gate_scale
            * F.linear(gate_in, attn.attn_gate_w.to(n.dtype))
        )
        y = y * g[..., None]
    y = y.reshape(bsz, seqlen, dim)
    attn_out = F.linear(y, out_w.to(n.dtype))
    if lora.o_loras is not None:
        o_in = y if getattr(lora, "o_lora_on_y", False) else n
        attn_out = attn_out + lora.o_loras[slot](o_in)
    attn_out = block.attn_scale.to(dtype=attn_out.dtype)[None, None, :] * attn_out
    mlp_n = block.mlp_norm(lane1) * block.ln_scale_factor
    mlp_out = block.mlp(mlp_n, up_w, down_w)
    if lora.mlp_loras is not None:
        mlp_out = mlp_out + lora.mlp_loras[slot](mlp_n)
    mlp_out = block.mlp_scale.to(dtype=lane1.dtype)[None, None, :] * mlp_out
    attn_resid = model.parallel_resid_lambdas[block_idx, 0].to(dtype=lane0.dtype)
    attn_post = model.parallel_post_lambdas[block_idx, 0].to(dtype=lane0.dtype)
    mlp_resid = model.parallel_resid_lambdas[block_idx, 1].to(dtype=lane0.dtype)
    mlp_post = model.parallel_post_lambdas[block_idx, 1].to(dtype=lane0.dtype)
    lane0 = attn_resid * lane0 + attn_post[0] * attn_out + mlp_post[0] * mlp_out
    lane1 = mlp_resid * lane1 + attn_post[1] * attn_out + mlp_post[1] * mlp_out
    return lane0, lane1


@torch.no_grad()
def forward_logits_with_lora(
    model: "GPT", input_ids: torch.Tensor, lora: TTTLoRA | None
) -> torch.Tensor:
    """Eval logits; if ``lora`` is None, falls back to base ``forward_logits``."""
    if lora is None:
        return model.forward_logits(input_ids)
    return _forward_logits_lora(model, input_ids, lora)


def _forward_logits_lora(
    model: "GPT", input_ids: torch.Tensor, lora: TTTLoRA
) -> torch.Tensor:
    x = model.tok_emb(input_ids)
    x = model._apply_smear(x, input_ids)
    x = F.rms_norm(x, (x.size(-1),))
    x0 = x
    skips: list[torch.Tensor] = []
    enc_iter = (
        model.encoder_indices
        if model.looping_active
        else list(range(model.num_encoder_layers))
    )
    dec_iter = (
        model.decoder_indices
        if model.looping_active
        else list(
            range(
                model.num_encoder_layers,
                model.num_encoder_layers + model.num_decoder_layers,
            )
        )
    )
    slot = 0
    for i in enc_iter:
        q_w, k_w, v_w, out_w, up_w, down_w = model._bank_weights(i)
        x = _block_with_lora(
            model, model.blocks[i], x, x0, lora, slot, q_w, k_w, v_w, out_w, up_w, down_w
        )
        slot += 1
        skips.append(x)
    psl = model.parallel_start_layer
    lane0 = None
    lane1 = None
    for skip_idx, i in enumerate(dec_iter):
        q_w, k_w, v_w, out_w, up_w, down_w = model._bank_weights(i)
        if i >= psl and psl > 0:
            if lane0 is None:
                lane0 = x
                lane1 = x
            if skip_idx < model.num_skip_weights and skips:
                skip = skips.pop()
                w = model.skip_weights[skip_idx].to(dtype=lane0.dtype)[None, None, :]
                if model.skip_gates is not None:
                    g = torch.sigmoid(
                        model.skip_gates[skip_idx].to(dtype=lane0.dtype)
                    )[None, None, :]
                    lane0 = torch.lerp(w * skip, lane0, g)
                else:
                    lane0 = lane0 + w * skip
            lane0, lane1 = _parallel_block_with_lora(
                model, i, lane0, lane1, x0, lora, slot, q_w, k_w, v_w, out_w, up_w, down_w
            )
        else:
            if skip_idx < model.num_skip_weights and skips:
                scaled_skip = (
                    model.skip_weights[skip_idx].to(dtype=x.dtype)[None, None, :]
                    * skips.pop()
                )
                if model.skip_gates is not None:
                    g = torch.sigmoid(
                        model.skip_gates[skip_idx].to(dtype=x.dtype)
                    )[None, None, :]
                    x = torch.lerp(scaled_skip, x, g)
                else:
                    x = x + scaled_skip
            x = _block_with_lora(
                model,
                model.blocks[i],
                x,
                x0,
                lora,
                slot,
                q_w,
                k_w,
                v_w,
                out_w,
                up_w,
                down_w,
            )
        slot += 1
    if lane0 is not None:
        x = model._final_parallel_hidden(lane0, lane1)
    x = model.final_norm(x)
    if model.tie_embeddings:
        logits = F.linear(x, model.tok_emb.weight)
    else:
        assert model.lm_head is not None
        logits = model.lm_head(x)
    logits = logits + lora.lm_head_lora(x)
    if model.asym_logit_enabled:
        return model._apply_asym_softcap(logits)
    return model.logit_softcap * torch.tanh(logits / model.logit_softcap)


def ce_with_lora(
    model: "GPT",
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    lora: TTTLoRA,
) -> torch.Tensor:
    """Teacher-forced CE through LoRA path (grads flow into LoRA only if base frozen)."""
    logits = _forward_logits_lora(model, input_ids, lora)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(),
        target_ids.reshape(-1),
        reduction="mean",
    )


def make_ttt_optimizer(lora: TTTLoRA, cfg: "XsaTttConfig") -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        lora.parameters(),
        lr=float(cfg.ttt_lora_lr),
        betas=(float(cfg.ttt_beta1), float(cfg.ttt_beta2)),
        weight_decay=float(cfg.ttt_weight_decay),
    )


def count_lora_parameters(lora: TTTLoRA) -> int:
    return int(sum(p.numel() for p in lora.parameters()))


def save_lora_adapters(
    lora: TTTLoRA,
    path: Path | str,
    *,
    meta: dict[str, Any] | None = None,
    dtype: torch.dtype | None = None,
) -> Path:
    """Write LoRA A/B weights (+ sibling ``.json`` meta for AC rebuild).

    ``dtype=torch.float16`` halves the shipped asset; ``load_lora_adapters``
    casts back to the module dtype on load.
    """
    path = Path(path)
    if path.suffix != ".safetensors":
        path = path.with_suffix(".safetensors")
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        k: v.detach().to("cpu", dtype=dtype).contiguous().clone()
        for k, v in lora.state_dict().items()
    }
    save_file(state, str(path))
    if meta is not None:
        path.with_suffix(".json").write_text(
            json.dumps(meta, indent=2, default=str) + "\n", encoding="utf8"
        )
    print(
        f"[lora] wrote {path} params={count_lora_parameters(lora):,} "
        f"bytes={path.stat().st_size:,}",
        flush=True,
    )
    return path


def load_lora_adapters(
    model: "GPT",
    cfg: "XsaTttConfig",
    path: Path | str,
    *,
    device: torch.device,
) -> TTTLoRA:
    """Rebuild ``TTTLoRA`` tree from ``cfg`` and load adapter weights."""
    path = Path(path)
    if path.suffix != ".safetensors":
        path = path.with_suffix(".safetensors")
    meta_path = path.with_suffix(".json")
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf8"))
        lora_cfg = meta.get("lora") or meta.get("cfg") or {}
        for key in (
            "ttt_lora_rank",
            "ttt_lora_alpha",
            "ttt_use_rslora",
            "ttt_k_lora",
            "ttt_o_lora",
            "ttt_mlp_lora",
            "ttt_warm_start_a",
            "ttt_o_lora_on_y",
        ):
            if key in lora_cfg:
                setattr(cfg, key, lora_cfg[key])
        if "seed" in meta and getattr(cfg, "seed", None) is not None:
            # Soft check: warn on mismatch but still load.
            if int(meta["seed"]) != int(cfg.seed):
                print(
                    f"[lora] warn: adapter seed={meta['seed']} "
                    f"!= cfg.seed={cfg.seed}",
                    flush=True,
                )
    lora = TTTLoRA(model, cfg).to(device)
    if bool(getattr(cfg, "ttt_o_lora_on_y", False)):
        lora.o_lora_on_y = True
    state = load_file(str(path), device=str(device))
    missing, unexpected = lora.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"LoRA load mismatch missing={missing} unexpected={unexpected}"
        )
    lora.eval()
    print(
        f"[lora] loaded {path} params={count_lora_parameters(lora):,} "
        f"o_on_y={int(getattr(lora, 'o_lora_on_y', False))}",
        flush=True,
    )
    return lora
