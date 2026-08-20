"""Apply shipped ΔW + small params onto a clean Init(seed) state dict."""

from __future__ import annotations

import torch


def apply_delta_to(
    model: torch.nn.Module,
    init_state: dict[str, torch.Tensor],
    delta: dict[str, torch.Tensor],
    small_params: dict[str, torch.Tensor],
) -> None:
    sd = dict(init_state)
    for name, d in delta.items():
        sd[name] = init_state[name].float() + d.to(init_state[name].device)
    for name, v in small_params.items():
        sd[name] = v.to(sd[name].dtype) if name in sd else v
    model.load_state_dict({k: v for k, v in sd.items()}, strict=True)
