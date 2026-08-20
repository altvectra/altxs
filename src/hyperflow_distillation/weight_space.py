"""Which dense tensors the mixed-bit ΔW codec covers.

Small 1-D parameters (norm scales, gates, lambdas, …) are shipped raw in the
codec as fp16 and applied on top of Init(seed) with the modeled ΔW.
"""

from __future__ import annotations

MODELED: dict[str, int] = {
    "qo_bank": 0,
    "kv_bank": 1,
    "mlp_up_bank": 2,
    "mlp_down_bank": 3,
    "tok_emb.weight": 4,
}
