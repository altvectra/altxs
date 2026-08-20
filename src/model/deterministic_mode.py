"""Deterministic CUDA / CPU mode for reproducible compression encode/decode.

Contest encode/decode needs fixed seeds and no stochastic ops. Default is
**strict**: math SDP only, ``use_deterministic_algorithms(True)``, cudnn
deterministic, TF32 off.

For train throughput use ``COMPRESSION_DETERMINISTIC=warn``: seeds still set,
but TF32 + ``cudnn.benchmark`` stay on.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np


def enable_deterministic_mode(
    seed: int = 1337,
    *,
    strict: bool | None = None,
    warn_only: bool | None = None,
) -> dict:
    """Seed RNGs and force deterministic CUDA algorithms.

    Env:
      ``COMPRESSION_DETERMINISTIC=strict|warn|0``
        - ``strict`` (default): fail on non-deterministic ops; math SDP only
        - ``warn``: warn_only=True (old behavior; allows mem-efficient attn)
        - ``0`` / ``false``: seeds only, no deterministic_algorithms flag
    """
    seed = int(seed)
    env = str(os.environ.get("COMPRESSION_DETERMINISTIC", "strict")).strip().lower()
    if strict is None:
        if env in {"0", "false", "no", "off"}:
            strict = False
            if warn_only is None:
                warn_only = True
        elif env in {"warn", "warn_only"}:
            strict = False
            if warn_only is None:
                warn_only = True
        else:
            strict = True
            if warn_only is None:
                warn_only = False
    if warn_only is None:
        warn_only = not strict

    os.environ["PYTHONHASHSEED"] = str(seed)
    # Required by cuBLAS when deterministic algorithms are on.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = os.environ.get(
        "CUBLAS_WORKSPACE_CONFIG", ":4096:8"
    )

    random.seed(seed)
    np.random.seed(seed % (2**32))

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Strict: reproducible kernels (slower). warn/off: seed only — keep H100 throughput
    # (TF32 + cudnn.benchmark).
    if strict and not warn_only:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False
        try:
            torch.set_float32_matmul_precision("highest")
        except Exception:  # noqa: BLE001
            pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:  # noqa: BLE001
            pass

    sdp = {"flash": None, "mem_efficient": None, "math": None}
    if torch.cuda.is_available() and hasattr(torch.backends.cuda, "enable_flash_sdp"):
        # Memory-efficient / flash attention backward is non-deterministic;
        # force the math kernel for strict mode.
        if strict and not warn_only:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        sdp = {
            "flash": bool(torch.backends.cuda.flash_sdp_enabled())
            if hasattr(torch.backends.cuda, "flash_sdp_enabled")
            else None,
            "mem_efficient": bool(torch.backends.cuda.mem_efficient_sdp_enabled())
            if hasattr(torch.backends.cuda, "mem_efficient_sdp_enabled")
            else None,
            "math": bool(torch.backends.cuda.math_sdp_enabled())
            if hasattr(torch.backends.cuda, "math_sdp_enabled")
            else None,
        }

    det_ok = True
    det_error = None
    if env not in {"0", "false", "no", "off"}:
        try:
            torch.use_deterministic_algorithms(True, warn_only=bool(warn_only))
        except TypeError:
            try:
                torch.use_deterministic_algorithms(True)
            except Exception as exc:  # noqa: BLE001
                det_ok = False
                det_error = str(exc)
        except Exception as exc:  # noqa: BLE001
            det_ok = False
            det_error = str(exc)
    else:
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:  # noqa: BLE001
            pass

    return {
        "seed": seed,
        "mode": "strict" if strict and not warn_only else ("warn" if warn_only else "off"),
        "deterministic_algorithms": (
            det_ok if env not in {"0", "false", "no", "off"} else False
        ),
        "warn_only": bool(warn_only),
        "cudnn_deterministic": bool(strict and not warn_only),
        "cudnn_benchmark": bool(not (strict and not warn_only)),
        "tf32": bool(not (strict and not warn_only)),
        "sdp": sdp,
        "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cuda": bool(torch.cuda.is_available()),
        "module_file": str(Path(__file__).resolve()),
        **({"error": det_error} if det_error else {}),
    }


def force_math_sdp() -> dict:
    """Hard-disable flash/mem-efficient SDP; enable math kernel only.

    Call this immediately before training so a stale import or later library
    init cannot re-enable non-deterministic attention.
    """
    import torch

    status: dict = {"ok": False}
    if not torch.cuda.is_available():
        status["reason"] = "no_cuda"
        return status

    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        status["ok"] = True
        status["flash"] = (
            torch.backends.cuda.flash_sdp_enabled()
            if hasattr(torch.backends.cuda, "flash_sdp_enabled")
            else None
        )
        status["mem_efficient"] = (
            torch.backends.cuda.mem_efficient_sdp_enabled()
            if hasattr(torch.backends.cuda, "mem_efficient_sdp_enabled")
            else None
        )
        status["math"] = (
            torch.backends.cuda.math_sdp_enabled()
            if hasattr(torch.backends.cuda, "math_sdp_enabled")
            else None
        )
    return status


def sdpa_math_context():
    """Context manager that forces MATH backend for scaled_dot_product_attention."""
    import contextlib

    import torch

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        return sdpa_kernel(SDPBackend.MATH)
    except Exception:  # noqa: BLE001
        try:
            # Older PyTorch
            return torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_math=True,
                enable_mem_efficient=False,
            )
        except Exception:  # noqa: BLE001
            return contextlib.nullcontext()

