"""Device selection: CUDA > MPS > CPU + deterministic RNG/kernels."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
_MODEL = _REPO / "model"
for _p in (_REPO, _MODEL):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)


def enable_deterministic(seed: int = 1337) -> dict:
    """Seed RNGs and apply ``COMPRESSION_DETERMINISTIC`` (default strict).

    Must run before ``build_model`` so init weights are reproducible.
    """
    from deterministic_mode import enable_deterministic_mode, force_math_sdp

    det = enable_deterministic_mode(int(seed))
    if det.get("mode") == "strict" and det.get("cuda"):
        force_math_sdp()
    return det


def resolve_device(prefer: str | None = None) -> torch.device:
    choice = (prefer or os.environ.get("XSA_TTT_DEVICE") or "auto").strip().lower()
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "mps":
        if not torch.backends.mps.is_available():
            raise SystemExit("XSA_TTT_DEVICE=mps but MPS is unavailable")
        return torch.device("mps")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("XSA_TTT_DEVICE=cuda but CUDA is unavailable")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def empty_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
