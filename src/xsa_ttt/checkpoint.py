"""Safetensors checkpoint I/O for xsa_ttt."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from .config import XsaTttConfig
from .model import GPT, build_model


def cpu_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    # clone(): avoid shared-storage rejects from safetensors
    return {
        k: v.detach().to("cpu").contiguous().clone() for k, v in state.items()
    }


def save_checkpoint_safetensors(
    state: dict[str, torch.Tensor],
    path: Path | str,
    *,
    meta: dict[str, Any] | None = None,
    quiet: bool = False,
) -> Path:
    """Write ``path.safetensors`` (+ optional ``path.json`` metadata)."""
    path = Path(path)
    if path.suffix != ".safetensors":
        path = path.with_suffix(".safetensors")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(cpu_state_dict(state), str(path))
    if meta is not None:
        meta_path = path.with_suffix(".json")
        meta_path.write_text(
            json.dumps(meta, indent=2, default=str) + "\n", encoding="utf8"
        )
    if not quiet:
        print(f"[ckpt] wrote {path}", flush=True)
    return path


def load_checkpoint_safetensors(
    path: Path | str,
    *,
    device: torch.device,
    cfg: XsaTttConfig | None = None,
) -> tuple[GPT, XsaTttConfig]:
    """Load weights from ``.safetensors`` (+ sibling ``.json`` cfg if needed)."""
    path = Path(path)
    if path.suffix != ".safetensors":
        path = path.with_suffix(".safetensors")
    meta_path = path.with_suffix(".json")
    if cfg is None:
        if not meta_path.is_file():
            if not path.is_file():
                raise FileNotFoundError(
                    f"need cfg or sibling metadata: {meta_path} "
                    f"(weights also missing at {path} — wrong path prefix?)"
                )
            raise FileNotFoundError(
                f"need cfg or sibling metadata: {meta_path}"
            )
        meta = json.loads(meta_path.read_text(encoding="utf8"))
        cfg_dict = meta.get("cfg") or meta
        # Allow nested {"cfg": {...}, ...} or flat cfg keys.
        if "num_layers" not in cfg_dict and isinstance(meta.get("cfg"), dict):
            cfg_dict = meta["cfg"]
        cfg = XsaTttConfig(**{
            k: v for k, v in cfg_dict.items() if k in XsaTttConfig.__dataclass_fields__
        })
    model = build_model(cfg, device=device)
    state = load_file(str(path), device=str(device))
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, cfg
