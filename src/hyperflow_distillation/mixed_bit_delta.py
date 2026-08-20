#!/usr/bin/env python3
"""Rebuild the dense student from the shipped mixed-bit ΔW codec.

Public contract (this is the product):

    student = Init(seed) + dequantized ΔW

The Release asset ``mixed_da_bpw*.safetensors`` + ``.json`` is the ΔW codec.
Seed and architecture come from that sidecar. There is no teacher checkpoint
and no pack/calibration CLI in this tree.

    python -m hyperflow_distillation.mixed_bit_delta decode \\
      --codec weights/mixed_da_bpw3.15_upb1.8.safetensors \\
      --out weights/student_dense.safetensors
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from hyperflow_distillation.train_hyperflow import apply_delta_to
from hyperflow_distillation.weight_space import MODELED
from xsa_ttt.checkpoint import save_checkpoint_safetensors
from xsa_ttt.config import XsaTttConfig
from xsa_ttt.device import enable_deterministic, resolve_device
from xsa_ttt.model import build_model

FORMAT_VERSION = 5
_COMPACT_MAGIC = b"MBZ1"


def _tensor_key(prefix: str, name: str, suffix: str = "") -> str:
    return f"{prefix}/{name}" + (f"/{suffix}" if suffix else "")


def _expand_compact_zlib(
    state: dict[str, torch.Tensor],
    meta: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Expand zlib-compacted packed streams back into a flat tensor dict."""
    comp = np.ascontiguousarray(
        state["compact/packed_zlib"].detach().cpu().numpy(), dtype=np.uint8
    ).reshape(-1)
    payload = zlib.decompress(comp.tobytes())
    if payload[:4] != _COMPACT_MAGIC:
        raise ValueError("bad compact magic")
    (index_len,) = struct.unpack_from("<I", payload, 4)
    index = json.loads(payload[8 : 8 + index_len].decode("utf8"))
    off = 8 + index_len
    chunks: dict[tuple[str, int], bytes] = {}
    for ent in index:
        length = int(ent["length"])
        blob = payload[off : off + length]
        off += length
        chunks[(str(ent["name"]), int(ent["bits"]))] = blob
    if off != len(payload):
        raise ValueError(
            f"compact payload trailing garbage: off={off} len={len(payload)}"
        )
    out: dict[str, torch.Tensor] = {
        k: v for k, v in state.items() if k != "compact/packed_zlib"
    }
    omit_bits = bool(meta.get("compact", {}).get("omit_bits", False))
    for name, layout in meta["layouts"].items():
        n_ch = int(layout["channels"])
        if omit_bits:
            bits = np.full(n_ch, 4, dtype=np.uint8)
            out[_tensor_key("bits", name)] = torch.from_numpy(bits)
        for b in (2, 3, 4):
            key = _tensor_key("packed", name, f"b{b}")
            blob = chunks.get((name, b))
            if blob is None:
                out[key] = torch.zeros(0, dtype=torch.uint8)
            else:
                out[key] = torch.from_numpy(
                    np.frombuffer(blob, dtype=np.uint8).copy()
                )
    return out


def unpack_codes(packed: np.ndarray, bits: int, count: int) -> np.ndarray:
    """Unpack unsigned fixed-width codes (LSB first)."""
    bits = int(bits)
    count = int(count)
    if bits not in (2, 3, 4):
        raise ValueError(f"bits must be 2/3/4, got {bits}")
    p = np.asarray(packed, dtype=np.uint8).reshape(-1)
    nbytes = 3 if bits == 3 else 1
    if p.size % nbytes:
        raise ValueError(f"packed byte count {p.size} invalid for {bits}-bit")
    byte_rows = p.reshape(-1, nbytes).astype(np.uint32)
    words = np.zeros(byte_rows.shape[0], dtype=np.uint32)
    for i in range(nbytes):
        words |= byte_rows[:, i] << (8 * i)
    per_word = 8 if bits == 3 else (8 // bits)
    shifts = np.arange(per_word, dtype=np.uint32) * bits
    out = (words[:, None] >> shifts[None, :]) & ((1 << bits) - 1)
    return out.astype(np.uint8).reshape(-1)[:count]


def _from_channels(
    name: str,
    channels: np.ndarray,
    shape: list[int],
    n_layers: int,
) -> torch.Tensor:
    """Packed functional channels → model tensor."""
    x = np.asarray(channels, dtype=np.float32)
    if name == "qo_bank":
        d = int(shape[-1])
        split = n_layers * d
        q = x[:split].reshape(n_layers, d, d)
        o = x[split:].reshape(n_layers, d, d).transpose(0, 2, 1)
        out = np.concatenate([q, o], axis=0)
    elif name == "kv_bank":
        kv, d = int(shape[1]), int(shape[2])
        split = n_layers * kv
        k = x[:split].reshape(n_layers, kv, d)
        v = x[split:].reshape(n_layers, kv, d)
        out = np.concatenate([k, v], axis=0)
    elif name == "mlp_up_bank":
        out = x.reshape(shape)
    elif name == "mlp_down_bank":
        out = x.reshape(shape[0], shape[2], shape[1]).transpose(0, 2, 1)
    elif name == "tok_emb.weight":
        out = x.reshape(shape)
    else:
        raise KeyError(name)
    return torch.from_numpy(np.ascontiguousarray(out))


def _symmetric_levels(scales: np.ndarray, bits: int) -> np.ndarray:
    """Expand per-channel fp16 scales to symmetric centroids ``(N, 2^b)``."""
    nlevels = 1 << int(bits)
    center = (nlevels - 1) / 2.0
    s = np.asarray(scales, dtype=np.float16).astype(np.float32).reshape(-1)
    return (np.arange(nlevels, dtype=np.float32)[None, :] - center) * s[:, None]


def _fp4_e2m1_table() -> np.ndarray:
    """OCP-style FP4 E2M1 decode table: code ``SEEM`` → float32 (16 values)."""
    out = np.zeros(16, dtype=np.float32)
    for code in range(16):
        sign = -1.0 if (code >> 3) & 1 else 1.0
        exp = (code >> 1) & 0b11
        mant = code & 1
        if exp == 0:
            out[code] = sign * 0.5 * float(mant)
        else:
            out[code] = sign * (1.0 + 0.5 * float(mant)) * float(2.0 ** (exp - 1))
    return out


_FP4_E2M1 = _fp4_e2m1_table()


def dequantize_channels(
    channel_bits: np.ndarray,
    codebooks: dict[int, np.ndarray],
    packed: dict[int, np.ndarray],
    *,
    channel_width: int,
    codebook_mode: str = "lloyd",
    scales: dict[int, np.ndarray] | None = None,
) -> np.ndarray:
    """Decode packed streams to functional channels."""
    mode = str(codebook_mode).lower().strip()
    bits = np.asarray(channel_bits, dtype=np.uint8).reshape(-1)
    if mode == "fp4":
        bits = bits.copy()
        bits[bits > 0] = 4
    out = np.zeros((bits.size, int(channel_width)), dtype=np.float32)
    for b in (2, 3, 4):
        idx = np.flatnonzero(bits == b)
        count = int(idx.size * channel_width)
        if count == 0:
            continue
        codes = unpack_codes(packed[b], b, count).reshape(idx.size, channel_width)
        cb = np.asarray(codebooks[b], dtype=np.float16).astype(np.float32)
        nlevels = 1 << b
        if mode in ("fp4", "hybrid_fp4_symmetric") and b == 4:
            if cb.shape != (idx.size, 1):
                raise ValueError(
                    f"fp4 scale shape {cb.shape} != {(idx.size, 1)}"
                )
            sc = cb.reshape(-1)
            out[idx] = _FP4_E2M1[codes.astype(np.int64)] * sc[:, None]
        elif mode == "lloyd" or (cb.ndim == 2 and cb.shape == (idx.size, nlevels)):
            if cb.shape != (idx.size, nlevels):
                raise ValueError(
                    f"{b}-bit codebook shape {cb.shape} != {(idx.size, nlevels)}"
                )
            out[idx] = np.take_along_axis(cb, codes.astype(np.int64), axis=1)
        elif mode in ("symmetric", "hybrid_fp4_symmetric") or (
            cb.ndim == 2 and cb.shape == (idx.size, 1)
        ):
            levels = _symmetric_levels(cb, b)
            out[idx] = np.take_along_axis(levels, codes.astype(np.int64), axis=1)
        elif mode == "shared" or (cb.ndim == 2 and cb.shape == (1, nlevels)):
            if scales is None or b not in scales:
                raise ValueError(f"{b}-bit shared mode requires scales")
            sc = np.asarray(scales[b], dtype=np.float16).astype(np.float32).reshape(-1)
            if sc.shape[0] != idx.size:
                raise ValueError(
                    f"{b}-bit scale shape {sc.shape} != ({idx.size},)"
                )
            shared = cb.reshape(1, nlevels)
            picked = shared[0, codes.astype(np.int64)]
            out[idx] = picked * sc[:, None]
        else:
            raise ValueError(
                f"{b}-bit codebook shape {cb.shape} incompatible with mode={mode}"
            )
    return out


def load_codec(
    codec_path: Path | str,
    *,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, torch.Tensor]]:
    """Decode codec into a deterministic Init(seed)+ΔŴ model."""
    codec_path = Path(codec_path)
    if codec_path.suffix != ".safetensors":
        codec_path = codec_path.with_suffix(".safetensors")
    meta_path = codec_path.with_suffix(".json")
    meta = json.loads(meta_path.read_text(encoding="utf8"))
    fmt = str(meta.get("format", ""))
    if fmt != "hope_mixed_bit_delta":
        raise ValueError(f"not a mixed-bit delta codec: {codec_path}")
    fmt_ver = int(meta.get("format_version", -1))
    if fmt_ver not in (3, 4, FORMAT_VERSION):
        raise ValueError(f"unsupported mixed-bit format {meta.get('format_version')}")
    codebook_mode = str(meta.get("codebook_mode", "lloyd")).lower().strip()
    seed = int(meta["seed"])
    enable_deterministic(seed)
    cfg_dict = meta.get("cfg") or {}
    cfg = XsaTttConfig(
        **{
            k: v
            for k, v in cfg_dict.items()
            if k in XsaTttConfig.__dataclass_fields__
        }
    )
    model = build_model(cfg, device=device)
    model.eval()
    model.gradient_checkpointing = False
    init_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    state = load_file(str(codec_path), device="cpu")
    if meta.get("compact") and "compact/packed_zlib" in state:
        state = _expand_compact_zlib(state, meta)
    n_layers = int(meta["n_layers"])
    delta: dict[str, torch.Tensor] = {}
    for name in MODELED:
        layout = meta["layouts"][name]
        fp16_key = _tensor_key("fp16", name)
        if fp16_key in state:
            channels = state[fp16_key].numpy().astype(np.float32)
        else:
            bits = state[_tensor_key("bits", name)].numpy()
            packed = {
                b: state[_tensor_key("packed", name, f"b{b}")].numpy()
                for b in (2, 3, 4)
            }
            codebooks = {
                b: state[_tensor_key("codebook", name, f"b{b}")].numpy()
                if _tensor_key("codebook", name, f"b{b}") in state
                else np.empty((0, 1 << b), dtype=np.float16)
                for b in (2, 3, 4)
            }
            scale_dict = {
                b: state[_tensor_key("scale", name, f"b{b}")].numpy()
                for b in (2, 3, 4)
                if _tensor_key("scale", name, f"b{b}") in state
            }
            channels = dequantize_channels(
                bits,
                codebooks,
                packed,
                channel_width=int(layout["channel_width"]),
                codebook_mode=codebook_mode,
                scales=scale_dict or None,
            )
        delta[name] = _from_channels(
            name, channels, list(layout["shape"]), n_layers
        ).to(device)
    if meta.get("global_lora"):
        raise ValueError(
            "this codec has a global-LoRA residual; that decoder is not shipped"
        )
    small = {
        name: state[_tensor_key("small", name)].to(device)
        for name in meta["small_params"]
    }
    apply_delta_to(model, init_state, delta, small)
    model.eval()
    return model, meta, delta


def export_anchor_masks(
    codec_path: Path | str,
    out_path: Path | str | None = None,
) -> Path:
    """Per-weight replenish stiffness from the shipped bit allocation.

    Decode-legal: both codec sides hold the same ΔW asset, so both build
    identical masks.
    """
    codec_path = Path(codec_path)
    if codec_path.suffix != ".safetensors":
        codec_path = codec_path.with_suffix(".safetensors")
    meta = json.loads(codec_path.with_suffix(".json").read_text(encoding="utf8"))
    if str(meta.get("format", "")) != "hope_mixed_bit_delta":
        raise ValueError(f"not a mixed-bit codec: {codec_path}")
    state = load_file(str(codec_path), device="cpu")
    if meta.get("compact") and "compact/packed_zlib" in state:
        state = _expand_compact_zlib(state, meta)
    n_layers = int(meta["n_layers"])

    masks: dict[str, torch.Tensor] = {}
    for name in MODELED:
        layout = meta["layouts"][name]
        width = int(layout["channel_width"])
        if _tensor_key("fp16", name) in state:
            n_ch = int(state[_tensor_key("fp16", name)].shape[0])
            lam_ch = np.ones((n_ch,), dtype=np.float32)
        else:
            bits = state[_tensor_key("bits", name)].numpy().astype(np.float32)
            lam_ch = np.where(bits <= 0, 0.0, 4.0 ** (bits - 4.0)).astype(
                np.float32
            )
        lam_full = np.repeat(lam_ch[:, None], width, axis=1)
        masks[name] = _from_channels(
            name, lam_full, list(layout["shape"]), n_layers
        ).to(torch.float16)
    for name in meta["small_params"]:
        masks[name] = torch.ones_like(
            state[_tensor_key("small", name)], dtype=torch.float16
        )

    if out_path is None:
        out_path = codec_path.with_name(codec_path.stem + "_anchor.safetensors")
    out_path = Path(out_path)
    save_file({k: v.contiguous() for k, v in masks.items()}, str(out_path))
    return out_path


def _runtime_cfg(meta: dict[str, Any]) -> dict[str, Any]:
    """Cfg embedded in a decoded student checkpoint (AC-ready)."""
    cfg_dict = dict(meta.get("cfg") or {})
    baked = meta.get("teacher_cfg") or {}
    for key in (
        "block_size",
        "online_retrain_every",
        "online_retrain_steps",
        "retrain_mode",
        "use_bf16",
        "rope_yarn",
        "ttt_bootstrap_symbols",
        "ttt_bootstrap_steps",
    ):
        if key not in cfg_dict and key in baked:
            cfg_dict[key] = baked[key]
    cfg_dict["gradient_checkpointing"] = True
    return cfg_dict


def _cmd_decode(args: argparse.Namespace) -> int:
    device = resolve_device(args.device)
    model, meta, _delta = load_codec(Path(args.codec), device=device)
    cfg_dict = _runtime_cfg(meta)
    path = save_checkpoint_safetensors(
        model.state_dict(),
        Path(args.out),
        meta={"cfg": cfg_dict, "mixed_bit_delta": meta},
    )
    print(
        f"[mixed-bit] Init(seed={meta.get('seed')})+ΔW → {path} "
        f"(block={cfg_dict.get('block_size')} "
        f"grad_ckpt={cfg_dict.get('gradient_checkpointing')})",
        flush=True,
    )
    return 0


def _cmd_anchor(args: argparse.Namespace) -> int:
    export_anchor_masks(args.codec, args.out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    dec = sub.add_parser(
        "decode",
        help="Init(seed) + shipped mixed-bit ΔW → dense student",
    )
    dec.add_argument("--codec", type=Path, required=True)
    dec.add_argument("--out", type=Path, required=True)
    dec.add_argument("--device", default=None)
    dec.set_defaults(func=_cmd_decode)
    anchor = sub.add_parser(
        "anchor",
        help="optional replenish stiffness masks from the same codec",
    )
    anchor.add_argument("--codec", type=Path, required=True)
    anchor.add_argument("--out", type=Path, default=None)
    anchor.set_defaults(func=_cmd_anchor)
    return p


def main(argv: list[str] | None = None) -> int:
    if "COMPRESSION_DETERMINISTIC" not in os.environ:
        os.environ["COMPRESSION_DETERMINISTIC"] = "strict"
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
