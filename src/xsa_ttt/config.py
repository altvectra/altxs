"""large architecture defaults + NNCP-style stream lockstep.

Width/depth and motif (11L/512d, XSA-all, SmearGate, SparseAttnGate,
depth recurrence, parallel residuals, AsymLogit) follow the large profile.

``vocab_size`` is 256 for the raw-byte payload_sim stream (a BPE sidecar can
override at train time).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
EXPECTED_587_BYTES = 587_138_826


@dataclass
class XsaTttConfig:
    # --- large architecture ---
    vocab_size: int = 256
    num_layers: int = 11
    xsa_last_n: int = 11
    model_dim: int = 512
    num_heads: int = 8
    num_kv_heads: int = 4
    mlp_mult: float = 4.0
    tie_embeddings: bool = True
    tied_embed_init_std: float = 0.005
    logit_softcap: float = 30.0
    asym_logit_rescale: bool = True
    rope_base: float = 1e4
    rope_dims: int = 16
    rope_yarn: bool = False
    ln_scale: bool = True
    qk_gain_init: float = 5.0
    skip_gates_enabled: bool = True
    num_loops: int = 2
    loop_start: int = 3
    loop_end: int = 5
    enable_looping_at: float = 0.35  # fraction of pretrain; compress starts with loops on
    parallel_start_layer: int = 8
    parallel_final_lane: str = "mean"
    smear_gate_enabled: bool = True
    gate_window: int = 12
    smear_gate_bos_fix: bool = False  # large launch uses SMEAR_GATE_BOS_FIX=0
    sparse_attn_gate_enabled: bool = True
    sparse_attn_gate_init_std: float = 0.0
    sparse_attn_gate_scale: float = 0.5
    attn_out_gate_enabled: bool = False
    gated_attn_enabled: bool = False

    # --- sequence / NNCP lockstep ---
    # H100 launcher defaults to 16384.
    block_size: int = 2048
    online_retrain_every: int = 2048
    online_retrain_steps: int = 1
    # NNCP-style replay: retrain step 0 uses the newest window; steps >= 1
    # use deterministic older windows from the decoded prefix. No-op at
    # steps=1, so the default does not change existing schedules.
    online_retrain_replay: bool = True
    ttt_bootstrap_symbols: int = 0
    ttt_bootstrap_steps: int = 0
    # Replenish-mode dose ramp: +1 pass per this many decoded chunks
    # (0 = off). Keeps early boundaries from re-epoching a tiny prefix.
    # Env: XSA_REPLENISH_RAMP.
    ttt_replenish_ramp: int = 0
    # Anneal, deterministic in stream position (lockstep-free); pass count
    # tapers with the same factor down to steps_min. Two modes:
    #   "linear" (NNCP regime): lr(pos) = max(lr_min, lr0 * (1 - pos/total)),
    #     anchored to ttt_replenish_total_bytes (or the encoded stream).
    #   "exp": lr(pos) = max(lr_min, lr0 * 2^(-pos / half_mb)); half_mb=0
    #     disables.
    # lr0 is the steady TTT lr (XSA_RETRAIN_LR). Envs: XSA_REPLENISH_ANNEAL /
    # XSA_REPLENISH_LR_HALF_MB / XSA_REPLENISH_LR_MIN /
    # XSA_REPLENISH_STEPS_MIN.
    ttt_replenish_anneal: str = "exp"
    ttt_replenish_lr_half_mb: float = 0.0
    ttt_replenish_lr_min: float = 4e-5
    ttt_replenish_steps_min: int = 1
    # First N retrain boundaries run Adam at lr=0 so the second-moment
    # preconditioner warms up without moving weights (lockstep: keyed off
    # end // every). 0 = off. Env: XSA_REPLENISH_WARMUP_STEPS.
    ttt_replenish_warmup_steps: int = 16
    # Linear-anneal denominator override (bytes). Default 0 anneals over the
    # stream actually being encoded; set to a fixed corpus length so short
    # AC probes reproduce the full run's schedule prefix (hot head) instead
    # of cooling to the floor within the probe. Env: XSA_REPLENISH_TOTAL_BYTES.
    ttt_replenish_total_bytes: int = 0
    # NNCP-shape heavy training: with steps>1, average gradients over all
    # step windows (fresh chunk + spaced replay) and take ONE optimizer step
    # per boundary — the data exposure of multi-pass with the update churn of
    # steps=1. Sequential multi-step (accum off) was measured net-destructive
    # on shallow wounds (s10r64 2026-08-08). Env: XSA_REPLENISH_ACCUM.
    ttt_replenish_accum: bool = False
    # Sequential mode only: LR multiplier for replay (non-fresh) steps, so
    # extra passes polish instead of churn. 1.0 = off.
    # Env: XSA_REPLENISH_REPLAY_LR_SCALE.
    ttt_replenish_replay_lr_scale: float = 1.0
    # Recency-biased replay (NNCP-style local adaptation): with steps>1,
    # draw spaced-replay chunks from the trailing N MB of the decoded
    # prefix instead of uniformly over the whole stream. 0 = uniform.
    # Env: XSA_REPLENISH_REPLAY_RECENT_MB.
    ttt_replenish_replay_recent_mb: float = 1.0
    # Precision-aware elastic anchor: path to per-weight stiffness masks
    # (mixed_bit_delta anchor). After each replenish step every anchored
    # weight is pulled back toward its shipped (AC-start) value by
    # rate*lambda — hot-LR churn cannot erode what the bits already paid
    # for, while low-precision/dead channels stay free to heal.
    # Envs: XSA_REPLENISH_ANCHOR / XSA_REPLENISH_ANCHOR_RATE.
    ttt_replenish_anchor: str = ""
    ttt_replenish_anchor_rate: float = 0.05
    # Forward-XM boundary (arXiv:2607.27372 adapted to replenishment): at
    # each boundary explore K candidate single-step updates from the same
    # weight snapshot — all on the previous chunk, using update scales
    # 0.5x / 1x / 2x, plus 4x while end < 100 MiB. Score each
    # on a held-out prefix probe
    # (fresh chunk, plus older chunks if probe>1; never the train chunk).
    # keep only the best. "Train on the best match" applied to the update
    # instead of the data. Candidates and scores depend only on (cfg, end)
    # plus the decoded prefix, so encode and decode pick the same winner.
    # K<2 disables. Shared-gradient cost is one step + K probe forwards.
    # Envs: XSA_REPLENISH_XM_K / XSA_REPLENISH_XM_PROBE_CHUNKS /
    # XSA_REPLENISH_XM_4X_UNTIL_MB.
    ttt_replenish_xm_k: int = 3
    # 1 = fresh (just-coded) chunk only. 2 = fresh + the chunk before
    # train (skips the train window). Mean-of-two amplified 4x on this
    # student; probe=1 was the 1.3244 finish.
    ttt_replenish_xm_probe_chunks: int = 1
    # Offer a 4x update-scale candidate while prefix < this many bytes.
    # 0 disables. Both codec sides see ``end``, so the drop is lockstep-safe.
    ttt_replenish_xm_4x_until_bytes: int = 100 * 1024 * 1024
    # Escalating replay bursts over the decoded prefix: "pos:steps,pos:steps".
    # Each burst fires at the first retrain boundary at/after pos (so both
    # codec sides agree). Env: XSA_TTT_BURSTS.
    ttt_burst_schedule: str = ""
    # Bursts replenish quant-damaged weights, so they train with
    # pretrain-style hypers (fresh AdamW per burst, sequential prefix walk)
    # instead of the one-shot TTT polish hypers above.
    # Envs: XSA_BURST_LR / XSA_BURST_BETA1 / XSA_BURST_BETA2 / XSA_BURST_WD.
    ttt_burst_lr: float = 3e-4
    ttt_burst_beta1: float = 0.9
    ttt_burst_beta2: float = 0.999
    ttt_burst_weight_decay: float = 0.01
    use_bf16: bool = False  # H100: XSA_BF16=1
    # LoRA TTT hypers for the large profile (RANK=80, LR=8e-5, WD=2.0, BETA2=0.99)
    # Distill / Unsloth-style: often rank=32|64, alpha=2*r, use_rslora=True.
    ttt_lora_rank: int = 80
    # Shared online-retrain LR for both full and LoRA modes. The field name is
    # retained for checkpoint compatibility; use env ``XSA_RETRAIN_LR``.
    ttt_lora_lr: float = 8e-5
    ttt_lora_alpha: float = 144.0
    ttt_use_rslora: bool = False  # if True: scale = α/√r (rsLoRA)
    ttt_warm_start_a: bool = True
    # 0 = off (full-mode AC default). LoRA/large used 2.0 via XSA_TTT_WEIGHT_DECAY.
    ttt_weight_decay: float = 0.0
    ttt_beta1: float = 0.0
    ttt_beta2: float = 0.99
    ttt_k_lora: bool = False  # TTT_K_LORA=0 on large; distill defaults on
    ttt_o_lora: bool = True
    ttt_mlp_lora: bool = True
    # True: O-LoRA on attn ``y`` (needed for exact ΔW_O factorization).
    ttt_o_lora_on_y: bool = False
    # "full" = AdamW all params (NNCP-style); "lora" = freeze base, LoRA TTT
    retrain_mode: str = "full"
    # Optional distilled adapters for AC (also XSA_LORA_PATH env).
    lora_path: str | None = None
    # Needed for strict math-SDP + block_size=16k online retrain on 80GB.
    gradient_checkpointing: bool = True

    # --- pretrain (stream lockstep before AC) ---
    pretrain_lr: float = 3e-4
    pretrain_weight_decay: float = 0.01
    grad_clip: float = 1.0
    stream_passes: int = 1
    max_iters: int = 0  # 0 → stream_passes × (n // every)
    seed: int = 1337
    batch_size: int = 1

    # --- data ---
    # Prefer payload_sim / dense-peel BPE4096; fall back handled in data.py
    data_path: str = field(
        default_factory=lambda: str(
            _REPO / "ltcb" / "data" / "enwik9.blsmc_full.m3v2.payload_sim.bpe4096"
        )
    )
    expected_bytes: int = EXPECTED_587_BYTES  # legacy payload_lex size reference


PROFILES = ("large", "compact", "lite", "mid", "smoke")


def make_config(
    *,
    profile: str = "large",
    data_path: str | None = None,
    seed: int = 1337,
    block_size: int | None = None,
) -> XsaTttConfig:
    """Build config. Profiles (approx param count @ vocab 256, tied emb):

    - ``large``: 11L/512d (~31.9M, default)
    - ``compact``: 8L/384d, thinner MLP (~10.3M) — serious H100 bake-off
    - ``lite``: 6L/256d (~3.4M)
    - ``mid``: 4L/192d (~1.3M) for 10MB probes
    - ``smoke``: tiny dims for local AC round-trip
    """
    cfg = XsaTttConfig(seed=int(seed))
    if data_path:
        cfg.data_path = str(data_path)

    prof = (profile or "large").strip().lower()
    if prof == "smoke":
        cfg.num_layers = 2
        cfg.xsa_last_n = 2
        cfg.model_dim = 64
        cfg.num_heads = 4
        cfg.num_kv_heads = 2
        cfg.mlp_mult = 2.0
        cfg.block_size = 128
        cfg.online_retrain_every = 128
        cfg.ttt_lora_rank = 8
        cfg.num_loops = 0
        cfg.parallel_start_layer = 99
        cfg.rope_dims = 8
        cfg.stream_passes = 1
    elif prof == "mid":
        cfg.num_layers = 4
        cfg.xsa_last_n = 4
        cfg.model_dim = 192
        cfg.num_heads = 6
        cfg.num_kv_heads = 2
        cfg.mlp_mult = 3.0
        cfg.block_size = 1024
        cfg.online_retrain_every = 1024
        cfg.ttt_lora_rank = 32
        cfg.num_loops = 1
        cfg.loop_start = 1
        cfg.loop_end = 2
        cfg.parallel_start_layer = 3
        cfg.rope_dims = 16
    elif prof == "lite":
        # ~3.4M: keep XSA-all + gates; one depth loop; thinner MLP
        cfg.num_layers = 6
        cfg.xsa_last_n = 6
        cfg.model_dim = 256
        cfg.num_heads = 8
        cfg.num_kv_heads = 2
        cfg.mlp_mult = 3.0
        cfg.ttt_lora_rank = 40
        cfg.num_loops = 1
        cfg.loop_start = 1
        cfg.loop_end = 2
        cfg.parallel_start_layer = 4
        cfg.rope_dims = 16
    elif prof == "compact":
        # ~10.3M: same motif as large, fewer layers / smaller width
        cfg.num_layers = 8
        cfg.xsa_last_n = 8
        cfg.model_dim = 384
        cfg.num_heads = 6
        cfg.num_kv_heads = 2
        cfg.mlp_mult = 3.0
        cfg.ttt_lora_rank = 48
        cfg.num_loops = 1
        cfg.loop_start = 2
        cfg.loop_end = 3
        cfg.parallel_start_layer = 6
        cfg.rope_dims = 16
    elif prof not in {"large", "full", "default"}:
        raise ValueError(f"unknown profile: {profile!r} (choose from {PROFILES})")

    if block_size is not None:
        cfg.block_size = int(block_size)
        cfg.online_retrain_every = int(block_size)

    # Env overrides (compress / train launchers)
    if "XSA_BLOCK_SIZE" in os.environ:
        b = int(os.environ["XSA_BLOCK_SIZE"])
        cfg.block_size = b
        cfg.online_retrain_every = int(
            os.environ.get("XSA_RETRAIN_EVERY", b)
        )
    if "XSA_RETRAIN_EVERY" in os.environ:
        cfg.online_retrain_every = int(os.environ["XSA_RETRAIN_EVERY"])
    if "XSA_RETRAIN_STEPS" in os.environ:
        cfg.online_retrain_steps = int(os.environ["XSA_RETRAIN_STEPS"])
    if "XSA_RETRAIN_REPLAY" in os.environ:
        cfg.online_retrain_replay = os.environ[
            "XSA_RETRAIN_REPLAY"
        ].strip().lower() not in {"0", "false", "no", "off"}
    if "XSA_REPLENISH_RAMP" in os.environ:
        cfg.ttt_replenish_ramp = int(os.environ["XSA_REPLENISH_RAMP"])
    if "XSA_REPLENISH_ANNEAL" in os.environ:
        cfg.ttt_replenish_anneal = (
            os.environ["XSA_REPLENISH_ANNEAL"].strip().lower()
        )
    if "XSA_REPLENISH_LR_HALF_MB" in os.environ:
        cfg.ttt_replenish_lr_half_mb = float(
            os.environ["XSA_REPLENISH_LR_HALF_MB"]
        )
    if "XSA_REPLENISH_LR_MIN" in os.environ:
        cfg.ttt_replenish_lr_min = float(os.environ["XSA_REPLENISH_LR_MIN"])
    if "XSA_REPLENISH_STEPS_MIN" in os.environ:
        cfg.ttt_replenish_steps_min = int(
            os.environ["XSA_REPLENISH_STEPS_MIN"]
        )
    if "XSA_REPLENISH_TOTAL_BYTES" in os.environ:
        cfg.ttt_replenish_total_bytes = int(
            os.environ["XSA_REPLENISH_TOTAL_BYTES"]
        )
    if "XSA_REPLENISH_WARMUP_STEPS" in os.environ:
        cfg.ttt_replenish_warmup_steps = int(
            os.environ["XSA_REPLENISH_WARMUP_STEPS"]
        )
    if "XSA_REPLENISH_ACCUM" in os.environ:
        cfg.ttt_replenish_accum = os.environ[
            "XSA_REPLENISH_ACCUM"
        ].strip().lower() not in {"0", "false", "no", "off"}
    if "XSA_REPLENISH_REPLAY_LR_SCALE" in os.environ:
        cfg.ttt_replenish_replay_lr_scale = float(
            os.environ["XSA_REPLENISH_REPLAY_LR_SCALE"]
        )
    if "XSA_REPLENISH_REPLAY_RECENT_MB" in os.environ:
        cfg.ttt_replenish_replay_recent_mb = float(
            os.environ["XSA_REPLENISH_REPLAY_RECENT_MB"]
        )
    if "XSA_REPLENISH_ANCHOR" in os.environ:
        cfg.ttt_replenish_anchor = os.environ["XSA_REPLENISH_ANCHOR"].strip()
    if "XSA_REPLENISH_ANCHOR_RATE" in os.environ:
        cfg.ttt_replenish_anchor_rate = float(
            os.environ["XSA_REPLENISH_ANCHOR_RATE"]
        )
    if "XSA_REPLENISH_XM_K" in os.environ:
        cfg.ttt_replenish_xm_k = int(os.environ["XSA_REPLENISH_XM_K"])
    if "XSA_REPLENISH_XM_PROBE_CHUNKS" in os.environ:
        cfg.ttt_replenish_xm_probe_chunks = int(
            os.environ["XSA_REPLENISH_XM_PROBE_CHUNKS"]
        )
    if "XSA_REPLENISH_XM_4X_UNTIL_MB" in os.environ:
        cfg.ttt_replenish_xm_4x_until_bytes = int(
            float(os.environ["XSA_REPLENISH_XM_4X_UNTIL_MB"]) * 1024 * 1024
        )
    if "XSA_TTT_BURSTS" in os.environ:
        cfg.ttt_burst_schedule = os.environ["XSA_TTT_BURSTS"].strip()
    if "XSA_BURST_LR" in os.environ:
        cfg.ttt_burst_lr = float(os.environ["XSA_BURST_LR"])
    if "XSA_BURST_BETA1" in os.environ:
        cfg.ttt_burst_beta1 = float(os.environ["XSA_BURST_BETA1"])
    if "XSA_BURST_BETA2" in os.environ:
        cfg.ttt_burst_beta2 = float(os.environ["XSA_BURST_BETA2"])
    if "XSA_BURST_WD" in os.environ:
        cfg.ttt_burst_weight_decay = float(os.environ["XSA_BURST_WD"])
    if "XSA_TTT_BOOTSTRAP_SYMBOLS" in os.environ:
        cfg.ttt_bootstrap_symbols = int(os.environ["XSA_TTT_BOOTSTRAP_SYMBOLS"])
    if "XSA_TTT_BOOTSTRAP_STEPS" in os.environ:
        cfg.ttt_bootstrap_steps = int(os.environ["XSA_TTT_BOOTSTRAP_STEPS"])
    if "XSA_RETRAIN_MODE" in os.environ:
        cfg.retrain_mode = os.environ["XSA_RETRAIN_MODE"].strip().lower()
    if "XSA_STREAM_PASSES" in os.environ:
        cfg.stream_passes = int(os.environ["XSA_STREAM_PASSES"])
    if "XSA_MAX_ITERS" in os.environ:
        cfg.max_iters = int(os.environ["XSA_MAX_ITERS"])
    if "XSA_DATA" in os.environ:
        cfg.data_path = os.environ["XSA_DATA"]
    # Canonical shared name; legacy aliases remain readable for old launchers.
    if "XSA_RETRAIN_LR" in os.environ:
        cfg.ttt_lora_lr = float(os.environ["XSA_RETRAIN_LR"])
    elif "XSA_TTT_LR" in os.environ:
        cfg.ttt_lora_lr = float(os.environ["XSA_TTT_LR"])
    elif "XSA_TTT_LORA_LR" in os.environ:
        cfg.ttt_lora_lr = float(os.environ["XSA_TTT_LORA_LR"])
    if "XSA_TTT_WEIGHT_DECAY" in os.environ:
        cfg.ttt_weight_decay = float(os.environ["XSA_TTT_WEIGHT_DECAY"])
    if "XSA_TTT_BETA1" in os.environ:
        cfg.ttt_beta1 = float(os.environ["XSA_TTT_BETA1"])
    elif (cfg.retrain_mode or "").lower() == "replenish":
        # Replenish uses (0.9, 0.999); other modes keep dataclass defaults.
        cfg.ttt_beta1 = 0.9
    if "XSA_TTT_BETA2" in os.environ:
        cfg.ttt_beta2 = float(os.environ["XSA_TTT_BETA2"])
    elif (cfg.retrain_mode or "").lower() == "replenish":
        cfg.ttt_beta2 = 0.999
    if "XSA_GRAD_CLIP" in os.environ:
        cfg.grad_clip = float(os.environ["XSA_GRAD_CLIP"])
    if "XSA_TTT_LORA_RANK" in os.environ:
        cfg.ttt_lora_rank = int(os.environ["XSA_TTT_LORA_RANK"])
    if "XSA_TTT_LORA_ALPHA" in os.environ:
        cfg.ttt_lora_alpha = float(os.environ["XSA_TTT_LORA_ALPHA"])
    if "XSA_TTT_USE_RSLORA" in os.environ:
        cfg.ttt_use_rslora = str(os.environ["XSA_TTT_USE_RSLORA"]).strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
    if "XSA_TTT_K_LORA" in os.environ:
        cfg.ttt_k_lora = str(os.environ["XSA_TTT_K_LORA"]).strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
    if "XSA_TTT_O_LORA" in os.environ:
        cfg.ttt_o_lora = str(os.environ["XSA_TTT_O_LORA"]).strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
    if "XSA_TTT_MLP_LORA" in os.environ:
        cfg.ttt_mlp_lora = str(os.environ["XSA_TTT_MLP_LORA"]).strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
    if "XSA_LORA_PATH" in os.environ:
        cfg.lora_path = os.environ["XSA_LORA_PATH"]
    if "XSA_PRETRAIN_LR" in os.environ:
        cfg.pretrain_lr = float(os.environ["XSA_PRETRAIN_LR"])
    if "XSA_PRETRAIN_WD" in os.environ:
        cfg.pretrain_weight_decay = float(os.environ["XSA_PRETRAIN_WD"])
    if "XSA_ENABLE_LOOPING_AT" in os.environ:
        # 0 → depth recurrence active from step 0 (continued/replenishment runs).
        cfg.enable_looping_at = float(os.environ["XSA_ENABLE_LOOPING_AT"])
    if "XSA_BF16" in os.environ:
        cfg.use_bf16 = str(os.environ["XSA_BF16"]).strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
    if "XSA_ROPE_YARN" in os.environ:
        cfg.rope_yarn = str(os.environ["XSA_ROPE_YARN"]).strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }

    return cfg


def should_online_retrain_at(end: int, *, every: int) -> bool:
    """NNCP lockstep boundary: exact multiples of ``every``."""
    if end <= 0 or every <= 0:
        return False
    return end % every == 0


def parse_burst_schedule(spec: str) -> list[tuple[int, int]]:
    """Parse ``"pos:steps,pos:steps"`` into sorted ``[(pos, steps), ...]``.

    Invalid entries raise ValueError (a silently-dropped burst would break
    encode/decode lockstep if the two sides disagreed).
    """
    out: list[tuple[int, int]] = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        pos_s, sep, steps_s = part.partition(":")
        if not sep:
            raise ValueError(f"burst entry {part!r} must be pos:steps")
        pos, steps = int(pos_s), int(steps_s)
        if pos <= 0 or steps <= 0:
            raise ValueError(f"burst entry {part!r} must be positive")
        out.append((pos, steps))
    out.sort()
    return out


def count_parameters(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))
