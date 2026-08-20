"""Shared incremental AC prob path (windowed KV cache) — encode ≡ decode.

Both codec sides compute every probability row through the *same* fixed-shape
W-token window step (``XSA_AC_WINDOW``): prefill the previous ``block_size``
context once per retrain segment, then W-token steps over fixed-size (static)
KV buffers with prefix-masked attention. Causal masking makes row j bitwise
independent of window tokens after position j, so a decode rerun with a
confirmed prefix and an arbitrary suffix still matches the encoder's row j.

  * Encode teacher-forces the true tokens — ONE full-attention step per W
    symbols. No draft.
  * Decode (``XSA_AC_DECODE_ROW=1``, default when ``W>1``): one newly known
    token at a time on a W=1 persist/mega (or eager) kernel, then commit
    that KV row. Encode uses a separate W-encode megakernel (per-row mega
    QKV/attn + tiled GEMV FFN shared with decode) so integer frequencies match.
    On CUDA, probabilities, decoded symbols and AC state remain on HBM.
    ``XSA_AC_DECODE_K`` (default 64) uses GPU AC after the lockstep
    one-row graph and an eager ``commit_window``. Segment carry on the
    device is the host ``_softmax_np`` row (not a raw ``torch.softmax``).
    ``XSA_AC_DECODE_UNROLL`` keeps a second-graph A/B.
    ``XSA_AC_MEGA_ENCODE=0`` keeps fused-slot GEMMs (will not lockstep).
    ``XSA_AC_DECODE_ROW=0`` restores the old full-window rerun.
  * ``XSA_AC_DRAFT=1`` optionally proposes a suffix with HCA-style draft
    attention before each full-window verify. Draft is off under one-row
    decode (already one token per step). AC rows always come from the
    verify / one-row kernel — same as encode.

On CUDA the step's shapes are fully static, so it can be captured into a
CUDA graph (``XSA_AC_GRAPH=1``) and replayed, fused with
``torch.compile`` (``XSA_AC_COMPILE=1``, mode via ``XSA_AC_COMPILE_MODE``),
or run the fused step (``XSA_AC_FUSED=1``, the default: W=1 multi-SM megakernel,
else per-slot Triton + packed QKV; CUDA graph unless ``XSA_AC_NO_GRAPH=1``).
``XSA_AC_PERSISTENT=0`` disables persist; ``XSA_AC_MEGA=0`` keeps the
cuBLAS itinerary. CUDA W=1 and W>1 one-row decode keep tokens, probs, and
the range coder on HBM (``XSA_AC_GPU_AC=0`` falls back to numpy). Fused+graph
wins over compile; compile wins over eager graphs. Fused numerics are not
bit-identical to eager — both codec sides must match.
Static AC buffers are ``mark_static_address``'d so inductor CUDA graphs
keep the in-place ``copy_`` writes (otherwise: "skipping cudagraphs due
to mutated inputs"). The CPU/eager path runs the identical ops, so tests
and graph replays agree. Encode and decode must both compile or both stay
eager — fused numerics are not bit-identical to eager.

Consequences:
  * Decode no longer needs the source product: ``decompress_full_sha_incremental``
    blind-decodes the segment-framed ``payload_*_fullsha.bin`` bitstream.
  * The bitstream is NOT interchangeable with the legacy chunked-TF encode
    (rows differ slightly). Incremental is the default for full-corpus AC;
    set ``XSA_AC_INCREMENTAL=0`` for the legacy encode. Moving an existing
    legacy bitstream to this format requires one re-encode.
  * Replenish / Forward-XM retrain at segment boundaries is unchanged (it only
    reads the agreed prefix), so bpw search results carry over.

Rotary/YaRN consistency: chunk-TF warms the per-block cos/sin cache lazily
(call-order dependent). Here both sides warm every block's table to
``block_size + online_retrain_every`` up front, so every later forward —
prefill, extend, retrain, XM probe — slices the same YaRN table.
"""

from __future__ import annotations

import hashlib
import math
import os
import struct
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from .compress import (  # noqa: F401  (sys.path side effect for coder import)
    _amp_ctx,
    _as_symbol_array,
    _chunk_ctx_start,
    _init_online_state,
    _online_steps_at,
    _scheduled_replenish_lr,
    _seg_decoded_to_bytes,
    _seg_roundtrip_mismatch,
    _softmax_np,
    _sym_dtype,
    _symbols_for_hash,
    _xm_share_str,
    online_retrain,
    uniform_probs,
)
from .config import XsaTttConfig
from .model import GPT, StaticState, _cudagraph_mark_step_begin, _mark_static_address


def compile_static_extend(model: GPT, state: StaticState, **kwargs):
    """Return a ``torch.compile`` copy of ``model.static_extend``.

    Thin wrapper around :meth:`GPT.compile_static_extend`. The original
    method stays eager. Use ``XSA_AC_COMPILE=1`` on :class:`StaticACEngine`
    to install this as the encode/decode verify step.
    """
    return model.compile_static_extend(state, **kwargs)

from arithmetic_coder_lm import decode_with_probs, encode_with_probs  # noqa: E402


def warm_rotary(model: GPT, total_len: int, device: torch.device) -> None:
    """Precompute every block's cos/sin table at the max segment length.

    Later calls with smaller ``seq_len`` slice this table, so the YaRN base is
    identical for all forwards on both codec sides regardless of call order.
    """
    for block in model.blocks:
        block.attn.rotary(int(total_len), device, torch.float32)


def _persist_note(model: GPT, persist_ok: bool) -> str:
    if not persist_ok or getattr(model, "_persist_disabled", False):
        return "fused slots"
    ws = getattr(model, "_persist_ws", None)
    if (
        ws is not None
        and getattr(ws, "use_mega", False)
        and not getattr(ws, "_mega_failed", False)
    ):
        mode = getattr(ws, "mega_mode", "mega")
        if mode == "one":
            return "persistent W=1 megakernel (one 8-CTA launch)"
        n_heads = int(getattr(ws, "n_heads", 8))
        attn = getattr(ws, "mega_attn", "fp32")
        from .mega_encode import _ffn_banner

        return (
            f"persistent W=1 megakernel "
            f"(QKV {n_heads} CTA + mega attn[{attn}] / slot + {_ffn_banner()})"
        )
    return "persistent W=1 stack"


def _ac_window() -> int:
    """Window W for the incremental AC step (format parameter).

    Encode runs one fixed-shape W-token step per W symbols (teacher-forced).
    Decode with ``XSA_AC_DECODE_ROW=1`` (default when W>1) steps one newly
    known token on a W=1 kernel; ``=0`` reruns the full W-window. Changing
    W requires a re-encode.
    """
    try:
        return max(1, int(os.environ.get("XSA_AC_WINDOW", "64")))
    except ValueError:
        return 64


def _env_on(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _env_int(name: str, default: int) -> int:
    try:
        return max(0, int(os.environ.get(name, str(default))))
    except ValueError:
        return default


def _want_decode_row(win: int) -> bool:
    """One-row decode inside a W>1 window (mega / persist / eager W=1)."""
    if int(win) <= 1:
        return False
    return _env_on("XSA_AC_DECODE_ROW", "1")


def _zero_static_state(st: StaticState) -> None:
    """Zero every buffer on one :class:`StaticState` (graph / fail cleanup)."""
    st.pos.fill_(0)
    st.prev_raw.zero_()
    st.cand_prev_raw.zero_()
    st.token.zero_()
    for sk in st.slots:
        if sk.k is not None:
            sk.k.zero_()
            sk.v.zero_()  # type: ignore[union-attr]


def _draft_cfg(win: int) -> tuple[int, int, int]:
    """Decode-only draft knobs: ``(last_n, pool, window)``. All-zero = off.

    Off unless ``XSA_AC_DRAFT=1``. Then HCA-style (pool=128, window=128);
    ``XSA_AC_DRAFT_POOL=0`` uses the last-N slice (``XSA_AC_DRAFT_LOCAL``).
    """
    if win <= 1 or not _env_on("XSA_AC_DRAFT", "0"):
        return 0, 0, 0
    pool = _env_int("XSA_AC_DRAFT_POOL", 128)
    if pool > 0:
        window = _env_int("XSA_AC_DRAFT_WINDOW", 128)
        return 0, pool, window
    return _env_int("XSA_AC_DRAFT_LOCAL", 256), 0, 0


class StaticACEngine:
    """Persistent prob engine for the incremental AC path.

    Owns the static device buffers (:meth:`GPT.make_static_state`) and — on
    CUDA — a fused step (``XSA_AC_FUSED=1``) captured into a CUDA graph
    unless ``XSA_AC_NO_GRAPH=1``, a ``torch.compile`` copy
    (``XSA_AC_COMPILE=1``), or an eager-step graph (``XSA_AC_GRAPH=1``).
    Fused+graph is the decode-speed path. On CPU (tests) the eager step runs.

    Per retrain segment, :meth:`reset_segment` rewinds ``pos`` and eagerly
    prefills the previous ``block_size`` context into the buffers. Weight
    mutations by replenish/XM retrain are in-place, so graph replays see the
    updated weights without recapture. ``prev_raw`` (smear seed for a
    window's first token) is only promoted on :meth:`advance_window`, so
    speculative re-runs of the same window are bit-identical.

    When ``XSA_AC_DECODE_ROW=1`` and W>1 a second ``row_state`` (win=1)
    mirrors the prefill prefix; decode commits one KV row per confirmed
    token via the W=1 persist/mega kernel. Encode teacher-forces the
    W-window through a separate encode megakernel (per-row mega QKV/attn
    + tiled GEMV FFN shared with decode) so integer frequencies match.
    ``XSA_AC_ENCODE_ROW=1`` makes encode use the one-row kernel instead.
    """

    def __init__(self, model: GPT, cfg: XsaTttConfig, device: torch.device):
        self.model = model
        self.device = device
        self.vocab = int(cfg.vocab_size)
        self.use_bf16 = bool(cfg.use_bf16)
        self.block = int(cfg.block_size)
        self.win = _ac_window()
        self.use_decode_row = _want_decode_row(self.win)
        self.encode_row = self.use_decode_row and _env_on(
            "XSA_AC_ENCODE_ROW", "0"
        )
        every = max(1, int(cfg.online_retrain_every))
        self.every = every
        n_windows = (every + self.win - 1) // self.win
        self.max_len = self.block + n_windows * self.win
        warm_rotary(model, self.max_len, device)
        # Cached compute-dtype banks/scales (no per-step ``.to``). Recast
        # after every replenish so fused/eager steps see the new weights.
        # Must run before make_static_state so smear/KV buffers match.
        model.prepare_ac_compute_weights()
        self.state: StaticState = model.make_static_state(
            self.max_len, device, win=self.win
        )
        self.row_state: StaticState | None = None
        self._row_graph = None
        self._row_graph_has_softmax = False
        self._row_graph_has_commit = False
        self._row_decode_graph = None
        self._row_decode_graph_k = 0
        self._row_decode_tokens = None
        self._row_decode_graph_failed = False
        self._row_decode_graph_has_forward = False
        self._row_fused_commit = False
        self._row_gpu_coder = None
        self._row_pos = 0
        # Persistent token / prob staging: one HtoD and one DtoH per window,
        # no per-step ``.cpu()`` / ``as_tensor(..., device=cuda)`` alloc+sync.
        pin = device.type == "cuda"
        self._token_pin = torch.zeros(self.win, dtype=torch.long, pin_memory=pin)
        self._token_np = self._token_pin.numpy()
        self._probs_dev = torch.empty_like(self.state.logits)
        _mark_static_address(self._probs_dev)
        self._probs_pin = torch.empty(
            self.win, self.vocab, dtype=torch.float32, pin_memory=pin
        )
        self._probs_np = self._probs_pin.numpy()
        if self.use_decode_row:
            self.row_state = model.make_static_state(
                self.max_len, device, win=1
            )
            self._row_token_pin = torch.zeros(
                1, dtype=torch.long, pin_memory=pin
            )
            self._row_token_np = self._row_token_pin.numpy()
            self._row_probs_dev = torch.empty_like(self.row_state.logits)
            _mark_static_address(self._row_probs_dev)
            self._row_probs_pin = torch.empty(
                self.vocab, dtype=torch.float32, pin_memory=pin
            )
            self._row_probs_np = self._row_probs_pin.numpy()
        self._graph_has_softmax = False
        self.uniform = uniform_probs(self.vocab)
        if device.type == "cuda":
            self._carry_dev = torch.empty(
                self.vocab, dtype=torch.float32, device=device
            )
            self._uniform_dev = torch.full(
                (self.vocab,),
                1.0 / float(self.vocab),
                dtype=torch.float32,
                device=device,
            )
            _mark_static_address(self._carry_dev)
        else:
            self._carry_dev = None
            self._uniform_dev = None
        self._gpu_ac = False
        self._gpu_row_ac = False
        if device.type == "cuda" and self.win == 1 and _env_on("XSA_AC_GPU_AC", "1"):
            from .gpu_ac import can_gpu_ac

            self._gpu_ac = can_gpu_ac(device)
        if (
            device.type == "cuda"
            and self.use_decode_row
            and _env_on("XSA_AC_GPU_AC", "1")
        ):
            from .gpu_ac import can_gpu_ac

            self._gpu_row_ac = can_gpu_ac(device)
        # W>1 encode still consumes host rows even when blind decode uses the
        # HBM coder. Keep its segment carry available on the host.
        self._keep_host_carry = not self._gpu_ac
        self.n_steps = 0  # full-attention verify / encode steps
        self.n_drafts = 0  # sparse-attention draft steps (decode only)
        self.draft_local, self.draft_pool, self.draft_win = _draft_cfg(self.win)
        self.use_draft = self.draft_pool > 0 or self.draft_local > 0
        if self.use_decode_row:
            self.use_draft = False
        self._carry: np.ndarray | None = None
        self._ctx_len = 0
        self._win_start = 0  # host-side prefix dest for the open window
        self._graph = None
        self._compiled_extend = None
        self._compile_mode = ""
        self._fused = False
        # Fused (+ CUDA graph) beats compile; compile beats eager graphs.
        # All three are CUDA-only. Draft stays on the eager method.
        want_fused = _env_on("XSA_AC_FUSED", "1")
        want_compile = _env_on("XSA_AC_COMPILE", "0")
        want_graph = _env_on("XSA_AC_GRAPH", "0")
        no_graph = _env_on("XSA_AC_NO_GRAPH", "0")
        if device.type == "cuda" and want_fused:
            self._fused = True
            graph_note = "ungraphed"
            persist_note = "fused slots"
            row_graph_note = "ungraphed"
            persist_ok = False
            encode_ok = False
            want_enc_mega = (
                self.win > 1 and self.use_decode_row and not self.encode_row
            )
            if no_graph:
                if want_enc_mega:
                    encode_ok = self._prepare_encode_mega()
                self._warmup_fused()
                if self.use_decode_row:
                    persist_ok = self._prepare_persist_ws()
                    self._warmup_row_fused()
                    persist_note = _persist_note(self.model, persist_ok)
                else:
                    persist_note = _persist_note(
                        self.model,
                        not getattr(self.model, "_persist_disabled", False),
                    )
            else:
                if want_enc_mega:
                    encode_ok = self._prepare_encode_mega()
                persist_ok = self._prepare_persist_ws()
                try:
                    self._capture_graph(self._step_fused_rows)
                    self._graph_has_softmax = True
                    graph_note = "CUDA graph"
                except Exception as exc:  # pragma: no cover - env dependent
                    self._graph = None
                    self._graph_has_softmax = False
                    self._warmup_fused()
                    print(
                        f"[AC incr] fused CUDA graph capture failed ({exc!r}); "
                        "using ungraphed fused steps",
                        file=sys.stderr,
                        flush=True,
                    )
                persist_note = _persist_note(self.model, persist_ok)
                if self.use_decode_row:
                    try:
                        self._capture_row_graph(self._step_row_fused_rows)
                        self._row_graph_has_softmax = True
                        self._row_graph_has_commit = False
                        row_graph_note = "CUDA graph"
                    except Exception as exc:  # pragma: no cover - env dependent
                        self._row_graph = None
                        self._row_graph_has_softmax = False
                        self._row_graph_has_commit = False
                        print(
                            f"[AC incr] one-row decode CUDA graph capture "
                            f"failed ({exc!r}); using ungraphed row steps",
                            file=sys.stderr,
                            flush=True,
                        )
            if encode_ok:
                ews = getattr(self.model, "_encode_ws", None)
                attn = getattr(ews, "mega_attn", "dot") if ews is not None else "dot"
                from .mega_encode import _ffn_banner

                ffn = _ffn_banner(ews)
                encode_persist = f"canonical W-encode mega (attn[{attn}] + {ffn})"
            elif self.use_decode_row:
                encode_persist = "fused slots"
            else:
                encode_persist = persist_note
            print(
                f"[AC incr] fused {self.win}-token window step "
                f"({graph_note}; {encode_persist}; "
                "XSA_AC_FUSED=0 for eager, XSA_AC_NO_GRAPH=1 to skip capture)",
                file=sys.stderr,
                flush=True,
            )
            if self.use_decode_row:
                if self.encode_row:
                    enc_note = "one-row encode+decode"
                elif encode_ok:
                    enc_note = "canonical W-encode mega / one-row decode"
                else:
                    enc_note = "W-batched fused encode / one-row decode"
                print(
                    f"[AC incr] {enc_note} "
                    f"({row_graph_note}; {persist_note}; "
                    "XSA_AC_DECODE_ROW=0 to rerun the full window)",
                    file=sys.stderr,
                    flush=True,
                )
            if self._gpu_ac:
                print(
                    "[AC incr] W=1 HBM range coder "
                    "(tokens/probs/state stay on device until segment flush)",
                    file=sys.stderr,
                    flush=True,
                )
            elif self.win == 1 and _env_on("XSA_AC_GPU_AC", "1"):
                print(
                    "[AC incr] W=1 host range coder (HBM AC off)",
                    file=sys.stderr,
                    flush=True,
                )
        elif device.type == "cuda" and want_compile:
            try:
                self._install_compiled_extend()
                print(
                    f"[AC incr] torch.compile {self._compile_mode} copy of "
                    f"the {self.win}-token window step "
                    "(tf32=high; XSA_AC_COMPILE=0 for eager)",
                    file=sys.stderr,
                    flush=True,
                )
            except Exception as exc:  # pragma: no cover - env dependent
                self._compiled_extend = None
                self._compile_mode = ""
                self._reset_all()
                print(
                    f"[AC incr] torch.compile failed ({exc!r}); "
                    "falling back to eager window steps",
                    file=sys.stderr,
                    flush=True,
                )
            self._maybe_capture_row_eager_graph(no_graph)
        elif device.type == "cuda" and want_graph and not no_graph:
            # Graph replay runs the identical kernels as the eager step, so a
            # capture failure can safely fall back to eager (same numerics,
            # just slower) without changing the bitstream format.
            try:
                self._capture_graph()
                print(
                    f"[AC incr] CUDA graph captured for the {self.win}-token "
                    "window step",
                    file=sys.stderr,
                    flush=True,
                )
            except Exception as exc:  # pragma: no cover - env dependent
                self._graph = None
                self._reset_all()
                print(
                    f"[AC incr] CUDA graph capture failed ({exc!r}); "
                    "falling back to eager window steps",
                    file=sys.stderr,
                    flush=True,
                )
            self._maybe_capture_row_eager_graph(no_graph=False)
        else:
            print(
                f"[AC incr] eager {self.win}-token window step "
                "(XSA_AC_FUSED=1 for Triton, XSA_AC_COMPILE=1, "
                "XSA_AC_GRAPH=1 to capture)",
                file=sys.stderr,
                flush=True,
            )
            if self.use_decode_row:
                print(
                    "[AC incr] one-row decode (eager W=1 kernel; "
                    "XSA_AC_DECODE_ROW=0 to rerun the full window)",
                    file=sys.stderr,
                    flush=True,
                )

    def draft_kind(self) -> str:
        if not self.use_draft:
            return "off"
        if self.draft_pool > 0:
            return f"HCA pool={self.draft_pool} window={self.draft_win}"
        return f"sparse local={self.draft_local}"

    def step_kind(self) -> str:
        if self._fused and self._graph is not None:
            return "fused-cuda-graph"
        if self._fused:
            return "fused"
        if self._compiled_extend is not None:
            return f"compile:{self._compile_mode}"
        if self._graph is not None:
            return "cuda-graph"
        return "eager"

    def decode_kind(self) -> str:
        if not self.use_decode_row:
            return "window"
        suffix = ""
        if self._gpu_row_ac:
            k = min(64, _env_int("XSA_AC_DECODE_K", 64))
            suffix = f"+gpu-ac-k{k}" if k > 1 else "+gpu-ac"
        if self._row_graph is not None:
            return "row-cuda-graph" + suffix
        if self._fused:
            return "row-fused" + suffix
        return "row-eager" + suffix

    def refresh_ac_weights(self) -> None:
        """Recast bank/scale views after in-place replenish updates.

        CUDA graphs pin raw data_ptrs. Retrain allocates a 32k activation
        tree that can recycle a captured address if any workspace tensor
        was replaced (or its last Python ref dropped). Drop and recapture
        so encode and decode both see the post-replenish bindings.
        """
        self.model.prepare_ac_compute_weights(
            getattr(self.model, "_ac_compute_dtype", None)
        )
        self._recapture_fused_graphs()

    def _recapture_fused_graphs(self) -> None:
        """Invalidate captured steps; recapture window/row graphs if fused."""
        self._row_decode_graph = None
        self._row_decode_graph_k = 0
        self._row_decode_tokens = None
        self._row_decode_graph_failed = False
        self._row_decode_graph_has_forward = False
        self._row_fused_commit = False
        if self.device.type != "cuda" or not self._fused:
            self._graph = None
            self._graph_has_softmax = False
            self._row_graph = None
            self._row_graph_has_softmax = False
            self._row_graph_has_commit = False
            return
        if _env_on("XSA_AC_NO_GRAPH", "0"):
            self._graph = None
            self._graph_has_softmax = False
            self._row_graph = None
            self._row_graph_has_softmax = False
            self._row_graph_has_commit = False
            return
        try:
            self._capture_graph(self._step_fused_rows)
            self._graph_has_softmax = True
        except Exception as exc:  # pragma: no cover - CUDA/runtime dependent
            self._graph = None
            self._graph_has_softmax = False
            print(
                f"[AC incr] post-retrain window graph recapture failed "
                f"({exc!r}); using ungraphed fused steps",
                file=sys.stderr,
                flush=True,
            )
        if self.use_decode_row:
            try:
                self._capture_row_graph(self._step_row_fused_rows)
                self._row_graph_has_softmax = True
                self._row_graph_has_commit = False
            except Exception as exc:  # pragma: no cover - CUDA/runtime dependent
                self._row_graph = None
                self._row_graph_has_softmax = False
                self._row_graph_has_commit = False
                print(
                    f"[AC incr] post-retrain one-row graph recapture failed "
                    f"({exc!r}); using ungraphed row steps",
                    file=sys.stderr,
                    flush=True,
                )

    def _step_eager(self) -> None:
        with _amp_ctx(self.device, self.use_bf16):
            self.model.static_extend(self.state)

    def _step_fused(self) -> None:
        # Weights are already in the compute dtype — skip autocast copies.
        self.model.static_extend_fused(self.state)

    def _softmax_into_dev(self) -> None:
        """Window softmax on GPU into the static ``_probs_dev`` buffer."""
        torch.softmax(self.state.logits, dim=-1, out=self._probs_dev)

    def _step_fused_rows(self) -> None:
        """Fused step + softmax (captured together so replay has no extra launch)."""
        self._step_fused()
        self._softmax_into_dev()

    def _persist_target(self) -> StaticState:
        """State persist/mega attaches to (W=1 row_state when decode-row)."""
        if self.row_state is not None:
            return self.row_state
        return self.state

    def _step_row_eager(self) -> None:
        assert self.row_state is not None
        with _amp_ctx(self.device, self.use_bf16):
            self.model.static_extend(self.row_state)

    def _step_row_fused(self) -> None:
        assert self.row_state is not None
        self.model.static_extend_fused(self.row_state)

    def _softmax_row_into_dev(self) -> None:
        assert self.row_state is not None
        torch.softmax(self.row_state.logits, dim=-1, out=self._row_probs_dev)

    def _step_row_fused_rows(self) -> None:
        self._step_row_fused()
        self._softmax_row_into_dev()

    def _step_row_eager_rows(self) -> None:
        self._step_row_eager()
        self._softmax_row_into_dev()

    def _copy_window_tokens(self, tokens: np.ndarray) -> None:
        """Host→device token write via pinned staging (no per-call ``as_tensor``)."""
        src = np.ascontiguousarray(tokens, dtype=np.int64).reshape(-1)
        self._token_np[:] = src
        self.state.token.copy_(
            self._token_pin.view(1, -1),
            non_blocking=self.device.type == "cuda",
        )

    def _rows_to_numpy(self) -> np.ndarray:
        """One DtoH of (W, V) into pinned memory; copy so the next step is safe."""
        if self.device.type == "cuda":
            self._probs_pin.copy_(
                self._probs_dev.view(self.win, self.vocab), non_blocking=True
            )
            torch.cuda.current_stream().synchronize()
        else:
            self._probs_pin.copy_(self._probs_dev.view(self.win, self.vocab))
        return self._probs_np.copy()

    def _row_to_numpy(self) -> np.ndarray:
        """One DtoH of (V,) for the one-row decode kernel."""
        if self.device.type == "cuda":
            self._row_probs_pin.copy_(
                self._row_probs_dev.view(self.vocab), non_blocking=True
            )
            torch.cuda.current_stream().synchronize()
        else:
            self._row_probs_pin.copy_(self._row_probs_dev.view(self.vocab))
        return self._row_probs_np.copy()

    def _prepare_persist_ws(self) -> bool:
        """Allocate + JIT persist on the default stream. True if the kernel runs."""
        if self.device.type != "cuda":
            return False
        st = self._persist_target()
        st.set_step_mode(True)
        try:
            from .persistent_step import prepare_persistent, try_persistent_extend

            if getattr(self.model, "_persist_ws", None) is None:
                ws = prepare_persistent(self.model, st)
                if ws is not None:
                    self.model._persist_ws = ws
            st.mark_static_addresses()
            if not try_persistent_extend(self.model, st):
                self._reset_all()
                return False
            torch.cuda.synchronize()
        except Exception as exc:
            self.model._persist_disabled = True
            print(
                f"[AC incr] persistent stack warmup failed ({exc!r}); "
                "using per-slot fused steps",
                file=sys.stderr,
                flush=True,
            )
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            self._reset_all()
            return False
        self._reset_all()
        return True

    def _prepare_encode_mega(self) -> bool:
        """Allocate the W-encode megakernel on the W-window state."""
        if self.device.type != "cuda" or int(self.win) < 2:
            return False
        self.state.set_step_mode(True)
        try:
            from .mega_encode import prepare_mega_encode, try_mega_encode

            if getattr(self.model, "_encode_ws", None) is None:
                ws = prepare_mega_encode(self.model, self.state)
                if ws is not None:
                    self.model._encode_ws = ws
            self.state.mark_static_addresses()
            if not try_mega_encode(self.model, self.state):
                self._reset_all()
                return False
            torch.cuda.synchronize()
        except Exception as exc:
            self.model._encode_disabled = True
            print(
                f"[AC incr] W-encode megakernel warmup failed ({exc!r}); "
                "using per-slot fused encode",
                file=sys.stderr,
                flush=True,
            )
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            self._reset_all()
            return False
        self._reset_all()
        return True

    def _warmup_fused(self) -> None:
        """JIT Triton kernels on the static buffers, then wipe."""
        self.state.set_step_mode(True)
        for _ in range(3):
            self._step_fused()
        self._reset_all()
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def _warmup_row_fused(self) -> None:
        """JIT the W=1 persist/fused row kernel, then wipe."""
        assert self.row_state is not None
        self.row_state.set_step_mode(True)
        for _ in range(3):
            self._step_row_fused()
        self._reset_all()
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def _step_compiled(self) -> None:
        assert self._compiled_extend is not None
        _cudagraph_mark_step_begin()
        with _amp_ctx(self.device, self.use_bf16):
            self._compiled_extend()

    def _install_compiled_extend(self) -> None:
        """Allocate KV, pin buffers, compile static_extend, warm, wipe.

        ``StaticKV.ensure`` must run once in eager first so Dynamo does not
        specialize on ``k is None``. Those K/V tensors plus the window
        buffers are marked static-address so inductor CUDA graphs keep the
        in-place ``copy_`` writes. Draft still uses the eager method.
        """
        self.state.set_step_mode(True)
        self._step_eager()
        self.state.mark_static_addresses()
        mode = (
            os.environ.get("XSA_AC_COMPILE_MODE", "reduce-overhead").strip()
            or "reduce-overhead"
        )
        self._compiled_extend = self.model.compile_static_extend(
            self.state, mode=mode
        )
        self._compile_mode = mode
        for _ in range(3):
            self._step_compiled()
        self._reset_all()
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def _reset_all(self) -> None:
        """Zero every static buffer (post-capture / post-failure cleanup)."""
        _zero_static_state(self.state)
        if self.row_state is not None:
            _zero_static_state(self.row_state)

    def _capture_graph_on(
        self, state: StaticState, step, attr: str
    ) -> None:
        """Warm up then capture ``step`` against ``state``; wipe pollution."""
        state.set_step_mode(True)
        state.mark_static_addresses()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                step()
            side.synchronize()
            graph = torch.cuda.CUDAGraph()
            try:
                ctx = torch.cuda.graph(
                    graph, stream=side, capture_error_mode="thread_local"
                )
            except TypeError:
                ctx = torch.cuda.graph(graph, stream=side)
            with ctx:
                step()
        torch.cuda.current_stream().wait_stream(side)
        setattr(self, attr, graph)
        self._reset_all()
        torch.cuda.synchronize()

    def _capture_graph(self, step=None) -> None:
        """Warm up then capture ``step`` (default eager); wipe buffer pollution.

        Replenish recasts AC views in-place, so replay sees new weights
        without recapture. ``step`` must be the same callable used at
        runtime (fused or eager) — mixed capture/replay changes numerics.
        """
        if step is None:
            step = self._step_eager
        self._capture_graph_on(self.state, step, "_graph")

    def _capture_row_graph(self, step) -> None:
        """Capture the W=1 decode-row kernel into ``_row_graph``."""
        assert self.row_state is not None
        self._capture_graph_on(self.row_state, step, "_row_graph")

    def _maybe_capture_row_eager_graph(self, no_graph: bool) -> None:
        if not self.use_decode_row or no_graph:
            return
        try:
            self._capture_row_graph(self._step_row_eager_rows)
            self._row_graph_has_softmax = True
            print(
                "[AC incr] CUDA graph captured for one-row decode",
                file=sys.stderr,
                flush=True,
            )
        except Exception as exc:  # pragma: no cover - env dependent
            self._row_graph = None
            self._row_graph_has_softmax = False
            print(
                f"[AC incr] one-row decode CUDA graph capture failed "
                f"({exc!r}); using eager row steps",
                file=sys.stderr,
                flush=True,
            )

    def reset_segment(self, prefix: np.ndarray, start: int) -> None:
        """Rewind to a segment start; eager-prefill its context window."""
        self.model.eval()
        st = self.state
        _p0, ctx_start = _chunk_ctx_start(int(start), self.block)
        ctx = np.asarray(prefix[ctx_start:start], dtype=np.int64)
        st.pos.fill_(0)
        self._ctx_len = int(ctx.shape[0])
        self._win_start = self._ctx_len
        if ctx.size:
            ids = torch.as_tensor(ctx, device=self.device).unsqueeze(0)
            st.set_step_mode(False)
            self.model._set_ac_step_active(True)
            try:
                with torch.no_grad(), _amp_ctx(self.device, self.use_bf16):
                    cache = self.model.prefill_cache(ids, kv_slots=st.slots)
                    logits = self.model.logits_from_cache(cache)
            finally:
                self.model._set_ac_step_active(False)
            st.set_step_mode(True)
            st.prev_raw.copy_(cache.last_raw_emb)
            # Host AC / encode rows use ``_softmax_np`` (CPU + renormalize).
            # Raw ``torch.softmax`` on HBM disagrees by a few integer masses
            # and desyncs GPU AC by local=8 (148 vs 144 on seg 4).
            take = logits.reshape(-1)[: self.vocab]
            carry = _softmax_np(take, vocab_size=self.vocab)
            self._carry = carry
            self._copy_carry_dev(carry)
        else:
            # Stream start: no context, first row is uniform, smear sees zeros
            # (identical to the no-previous-token embed path).
            st.prev_raw.zero_()
            self._carry = None
            self._copy_carry_dev(self.uniform)
        self._rebind_packed_kv(st, getattr(self.model, "_encode_ws", None))
        self._sync_row_from_prefix()
        if self.row_state is not None:
            self._rebind_packed_kv(
                self.row_state, getattr(self.model, "_persist_ws", None)
            )

    def _rebind_packed_kv(self, state: StaticState, ws) -> None:
        """Keep slot views on ``ws.k_pack`` after eager prefill.

        Prefill writes ``slot.k``. Mega encode/decode read ``k_pack``. If
        ``StaticKV.ensure`` ever replaced the slot tensor, those two diverge
        and only one codec side sees the new prefix. Re-pack and drop the
        previous segment's tail so a leaked attn length cannot mix eras.
        """
        if ws is None or getattr(ws, "k_pack", None) is None:
            return
        max_len = int(ws.max_len)
        n_kv = int(ws.n_kv)
        hd = int(ws.hd)
        n = min(int(self._ctx_len), max_len)
        for i, sk in enumerate(state.slots):
            if sk.k is None or sk.v is None:
                continue
            pack_k = ws.k_pack[i]
            pack_v = ws.v_pack[i]
            if sk.k.data_ptr() != pack_k.data_ptr():
                src_k = sk.k.reshape(max_len, n_kv, hd).permute(1, 0, 2)
                src_v = sk.v.reshape(max_len, n_kv, hd).permute(1, 0, 2)
                pack_k.copy_(src_k.to(dtype=pack_k.dtype))
                pack_v.copy_(src_v.to(dtype=pack_v.dtype))
                sk.k = pack_k.permute(1, 0, 2).unsqueeze(0)
                sk.v = pack_v.permute(1, 0, 2).unsqueeze(0)
                _mark_static_address(sk.k)
                _mark_static_address(sk.v)
            if n < max_len:
                pack_k[:, n:].zero_()
                pack_v[:, n:].zero_()

    def _sync_row_from_prefix(self) -> None:
        """Copy prefill prefix + smear seed into the W=1 decode state."""
        if self.row_state is None:
            return
        src, dst = self.state, self.row_state
        dst.prev_raw.copy_(src.prev_raw)
        dst.cand_prev_raw.zero_()
        dst.token.zero_()
        n = int(self._ctx_len)
        self._row_pos = n
        dst.pos.fill_(n)
        dst.set_step_mode(True)
        for a, b in zip(src.slots, dst.slots):
            if a.k is None:
                continue
            if b.k is None:
                b.ensure(a.k)
            assert b.k is not None and b.v is not None and a.v is not None
            if n > 0:
                b.k[:, :n].copy_(a.k[:, :n])
                b.v[:, :n].copy_(a.v[:, :n])
            if n < b.max_len:
                b.k[:, n:].zero_()
                b.v[:, n:].zero_()

    def _copy_carry_dev(self, row: np.ndarray) -> None:
        """Upload the host-AC carry so GPU AC sees the same float32 bytes."""
        if self._carry_dev is None:
            return
        src = np.ascontiguousarray(row, dtype=np.float32).reshape(self.vocab)
        self._carry_dev.copy_(torch.as_tensor(src, dtype=torch.float32))

    def carry_row(self) -> np.ndarray:
        """Row for the first symbol of the segment (uniform at stream start)."""
        if self._carry is None:
            return self.uniform
        return self._carry

    def carry_row_dev(self) -> torch.Tensor:
        """Device carry — same bytes as :meth:`carry_row` / encode rows[0]."""
        assert self._carry_dev is not None
        return self._carry_dev

    def run_window_dev(self, start_local: int) -> None:
        """W=1 step with tokens already on ``state.token``. No DtoH."""
        st = self.state
        self._win_start = self._ctx_len + int(start_local)
        st.pos.fill_(self._win_start)
        if self._graph is not None:
            self._graph.replay()
        elif self._fused:
            self._step_fused()
        elif self._compiled_extend is not None:
            self._step_compiled()
        else:
            self._step_eager()
        if not self._graph_has_softmax:
            self._softmax_into_dev()
        self.n_steps += 1

    def run_window(self, tokens: np.ndarray, start_local: int) -> np.ndarray:
        """One fixed-shape W-token step at segment offset ``start_local``.

        ``tokens`` (W,) are true symbols on encode, speculative on decode.
        Returns rows (W, V): rows[j] = P(symbol start+start_local+j+1 | ...).
        Row j is bitwise independent of tokens[j+1:]. Softmax stays on GPU;
        the coder is still numpy, so this does one pinned DtoH of (W, V)
        after the step — not a sync per layer.
        """
        st = self.state
        self._win_start = self._ctx_len + int(start_local)
        self._copy_window_tokens(tokens)
        st.pos.fill_(self._win_start)
        if self._graph is not None:
            self._graph.replay()
        elif self._fused:
            self._step_fused()
        elif self._compiled_extend is not None:
            self._step_compiled()
        else:
            self._step_eager()
        if not self._graph_has_softmax:
            self._softmax_into_dev()
        self.n_steps += 1
        return self._rows_to_numpy()

    def draft_window(self, tokens: np.ndarray, start_local: int) -> np.ndarray:
        """HCA / last-N propose step (decode only; not AC rows).

        Same weights and W-shape as :meth:`run_window`. Default scores
        mean-pooled prefix blocks (``draft_pool``) plus a raw local window
        and the causal tail — not the full 32k buffer. Writes the tail K/V
        (overwritten by the following verify) and does not commit. Never
        captured — shapes depend on ``pos``.
        """
        st = self.state
        self._win_start = self._ctx_len + int(start_local)
        self._copy_window_tokens(tokens)
        st.pos.fill_(self._win_start)
        start = (
            max(0, self._win_start - self.draft_local) if self.draft_local else 0
        )
        st.set_draft_range(
            self.draft_local,
            start,
            self._win_start,
            pool=self.draft_pool,
            window=self.draft_win,
        )
        try:
            with _amp_ctx(self.device, self.use_bf16):
                self.model.static_extend(self.state)
        finally:
            st.set_draft_range(0, 0, 0)
        self.n_drafts += 1
        self._softmax_into_dev()
        return self._rows_to_numpy()

    def advance_window(self) -> None:
        """Finalize a full window: commit tail K/V into prefix, promote smear."""
        self.state.commit_window(self._win_start)
        self.state.prev_raw.copy_(self.state.cand_prev_raw)

    def run_decode_row(self, token: int) -> np.ndarray:
        """One newly known token on the W=1 kernel; returns P(next).

        Appends that token's K/V into the row-state prefix (win=1 commit).
        Does not call :meth:`advance_window` on the W-encode state.
        """
        st = self.row_state
        if st is None:
            raise RuntimeError("run_decode_row requires XSA_AC_DECODE_ROW=1")
        self._row_token_np[0] = int(token)
        st.token.copy_(
            self._row_token_pin.view(1, 1),
            non_blocking=self.device.type == "cuda",
        )
        st.pos.fill_(self._row_pos)
        if self._row_graph is not None:
            self._row_graph.replay()
        elif self._fused:
            self._step_row_fused()
        else:
            self._step_row_eager()
        if not self._row_graph_has_softmax:
            self._softmax_row_into_dev()
        st.commit_window(self._row_pos)
        st.prev_raw.copy_(st.cand_prev_raw)
        self._row_pos += 1
        self.n_steps += 1
        return self._row_to_numpy()

    def run_decode_row_dev(self, token: torch.Tensor) -> None:
        """Run one confirmed row entirely on-device; leave P(next) in HBM."""
        st = self.row_state
        if st is None:
            raise RuntimeError("run_decode_row_dev requires XSA_AC_DECODE_ROW=1")
        st.token.copy_(token.view(-1)[:1].view(1, 1))
        st.pos.fill_(self._row_pos)
        if self._row_graph is not None:
            self._row_graph.replay()
        elif self._fused:
            self._step_row_fused()
        else:
            self._step_row_eager()
        if not self._row_graph_has_softmax:
            self._softmax_row_into_dev()
        st.commit_window(self._row_pos)
        st.prev_raw.copy_(st.cand_prev_raw)
        self._row_pos += 1
        self.n_steps += 1

    def _row_forward_probs(self) -> None:
        """One-row forward + softmax without commit (packed A/B only)."""
        if self._row_graph is not None and not self._row_graph_has_commit:
            self._row_graph.replay()
            if not self._row_graph_has_softmax:
                self._softmax_row_into_dev()
        elif self._fused:
            self._step_row_fused_rows()
        else:
            self._step_row_eager_rows()

    def _row_commit_decode_advance(
        self, coder, decoded: torch.Tensor, j: int
    ) -> None:
        """Land the tail KV, GPU-AC the next symbol, promote smear/pos."""
        st = self.row_state
        assert st is not None
        from .row_commit import advance_row_state, commit_packed_row

        ws = getattr(self.model, "_persist_ws", None)
        fused_commit = ws is not None and commit_packed_row(ws, st)
        sym = coder.decode_symbol(self._row_probs_dev.view(-1))
        if fused_commit:
            advance_row_state(st, sym, decoded, j)
        else:
            for sk in st.slots:
                if sk.k is None:
                    continue
                assert sk.v is not None
                sk.k.index_copy_(1, st.pos, sk.k[:, -1:])
                sk.v.index_copy_(1, st.pos, sk.v[:, -1:])
            advance_row_state(st, sym, decoded, j)
        self._row_fused_commit = bool(fused_commit)

    def _row_decode_graph_step(self, coder, decoded: torch.Tensor, j: int) -> None:
        """Packed A/B step: recapture the megakernel with commit+AC."""
        self._row_forward_probs()
        self._row_commit_decode_advance(coder, decoded, j)

    def capture_row_decode_graph(self, coder, k: int = 1) -> None:
        """Capture GPU AC + packed-KV commit. Forward stays on ``_row_graph``.

        The one-row graph already has the lockstep kernels; this graph
        only commits the tail and decodes. ``XSA_AC_DECODE_UNROLL``
        recaptures the packed forward for A/B.
        """
        if self.device.type != "cuda" or self.row_state is None:
            raise RuntimeError("row decode graph requires CUDA decode-row state")
        unroll = min(64, _env_int("XSA_AC_DECODE_UNROLL", 1))
        k = max(1, unroll if unroll > 1 else 1)
        include_forward = k > 1
        decoded = torch.empty(k, dtype=torch.int64, device=self.device)
        _mark_static_address(decoded)
        from .row_commit import advance_row_state, commit_packed_row

        self.row_state.mark_static_addresses()
        ws = getattr(self.model, "_persist_ws", None)
        if ws is not None and commit_packed_row(ws, self.row_state):
            for j in range(k):
                advance_row_state(self.row_state, coder.sym, decoded, j)
            self._row_fused_commit = True
        step = (
            self._row_decode_graph_step
            if include_forward
            else self._row_commit_decode_advance
        )
        graph = torch.cuda.CUDAGraph()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                for j in range(k):
                    step(coder, decoded, j)
            side.synchronize()
            try:
                ctx = torch.cuda.graph(
                    graph, stream=side, capture_error_mode="thread_local"
                )
            except TypeError:
                ctx = torch.cuda.graph(graph, stream=side)
            with ctx:
                for j in range(k):
                    step(coder, decoded, j)
        torch.cuda.current_stream().wait_stream(side)
        side.synchronize()
        self._row_decode_graph = graph
        self._row_decode_graph_k = k
        self._row_decode_tokens = decoded
        self._row_decode_graph_has_forward = include_forward
        commit_note = "fused packed-KV commit" if self._row_fused_commit else "index commit"
        if include_forward:
            print(
                f"[AC incr] captured {k}-step GPU decode graph ({commit_note})",
                file=sys.stderr,
                flush=True,
            )
        else:
            print(
                f"[AC incr] captured GPU-AC + packed-KV commit graph "
                f"({commit_note}; one-row forward is the lockstep graph)",
                file=sys.stderr,
                flush=True,
            )

    def replay_row_decode_graph(self, out: torch.Tensor, start: int) -> int:
        """Replay one k-step block and queue its decoded symbols into ``out``."""
        graph = self._row_decode_graph
        decoded = self._row_decode_tokens
        k = int(self._row_decode_graph_k)
        if graph is None or decoded is None or k <= 0:
            raise RuntimeError("row decode graph is not captured")
        graph.replay()
        out[start : start + k].copy_(decoded)
        self._row_pos += k
        self.n_steps += k
        return k


def _segment_rows_one_row(
    engine: StaticACEngine, arr: np.ndarray, seg, *, start: int
) -> np.ndarray:
    """Teacher-forced rows via the W=1 decode kernel (canonical lockstep)."""
    engine.reset_segment(arr, start)
    if isinstance(seg, (bytes, bytearray)):
        seg_np = np.frombuffer(seg, dtype=np.uint8).astype(np.int64)
    else:
        seg_np = np.asarray(seg, dtype=np.int64)
    n = int(seg_np.shape[0])
    rows_out = np.empty((n, engine.vocab), dtype=np.float32)
    rows_out[0] = engine.carry_row()
    for i in range(n - 1):
        rows_out[i + 1] = engine.run_decode_row(int(seg_np[i]))
    return rows_out


def _segment_rows_tf(
    engine: StaticACEngine, arr: np.ndarray, seg, *, start: int
) -> np.ndarray:
    """Teacher-forced prob rows for one segment (encode side / tests).

    rows[i] = P(seg[i] | everything before). Default is one W-window step
    per W symbols. ``XSA_AC_ENCODE_ROW=1`` uses the one-row kernel so
    encode frequencies match one-row decode when batched GEMMs diverge.
    """
    if engine.encode_row:
        return _segment_rows_one_row(engine, arr, seg, start=start)
    engine.reset_segment(arr, start)
    n = len(seg)
    win = engine.win
    if isinstance(seg, (bytes, bytearray)):
        seg_np = np.frombuffer(seg, dtype=np.uint8).astype(np.int64)
    else:
        seg_np = np.asarray(seg, dtype=np.int64)
    rows_out = np.empty((n, engine.vocab), dtype=np.float32)
    carry = engine.carry_row()
    w0 = 0
    while w0 < n:
        valid = min(win, n - w0)
        tokens = np.zeros(win, dtype=np.int64)
        tokens[:valid] = seg_np[w0 : w0 + valid]
        rows = engine.run_window(tokens, w0)
        rows_out[w0] = carry
        if valid > 1:
            rows_out[w0 + 1 : w0 + valid] = rows[: valid - 1]
        carry = rows[valid - 1]
        if valid == win:
            engine.advance_window()
        w0 += valid
    return rows_out


def _seg_np(seg) -> np.ndarray:
    if isinstance(seg, (bytes, bytearray)):
        return np.frombuffer(seg, dtype=np.uint8).astype(np.int64)
    return np.asarray(seg, dtype=np.int64)


def _encode_segment_gpu(
    engine: StaticACEngine,
    arr: np.ndarray,
    seg,
    *,
    start: int,
) -> bytes:
    """W=1 encode: one HtoD of tokens, one DtoH of the bitstream."""
    from .gpu_ac import GpuRangeCoder

    engine.reset_segment(arr, start)
    seg_np = _seg_np(seg)
    n = int(seg_np.shape[0])
    if n <= 0:
        return b""
    seg_dev = torch.as_tensor(seg_np, device=engine.device, dtype=torch.int64)
    coder = GpuRangeCoder(
        engine.device, max_bits=n * 48 + 4096, vocab=engine.vocab
    )
    coder.encode_symbol(engine.carry_row_dev(), seg_dev[0])
    for i in range(n - 1):
        engine.state.token.copy_(seg_dev[i : i + 1].view(1, -1))
        engine.run_window_dev(i)
        engine.advance_window()
        coder.encode_symbol(engine._probs_dev.view(-1), seg_dev[i + 1])
    return coder.finish()


def _decode_segment_gpu(
    engine: StaticACEngine,
    arr: np.ndarray,
    seg_payload: bytes,
    *,
    start: int,
    n_seg: int,
):
    """W=1 decode: payload HtoD once, symbols DtoH once."""
    from .gpu_ac import GpuRangeCoder

    engine.reset_segment(arr, start)
    n = int(n_seg)
    out = torch.empty(n, dtype=torch.int64, device=engine.device)
    coder = GpuRangeCoder(
        engine.device,
        max_bits=max(len(seg_payload) * 8, 32) + 64,
        vocab=engine.vocab,
    )
    coder.begin_decode(seg_payload)
    out[0:1].copy_(coder.decode_symbol(engine.carry_row_dev()))
    for i in range(n - 1):
        engine.state.token.copy_(out[i : i + 1].view(1, -1))
        engine.run_window_dev(i)
        engine.advance_window()
        out[i + 1 : i + 2].copy_(coder.decode_symbol(engine._probs_dev.view(-1)))
    host = out.detach().cpu().numpy()
    if engine.vocab <= 256:
        return bytes(host.astype(np.uint8).tolist())
    return host.astype(np.int64).tolist()


def _encode_segment_incremental(
    engine: StaticACEngine,
    arr: np.ndarray,
    seg,
    *,
    start: int,
) -> bytes:
    if engine._gpu_ac:
        return _encode_segment_gpu(engine, arr, seg, start=start)
    rows = _segment_rows_tf(engine, arr, seg, start=start)

    def probs_fn(i: int, _prefix):
        return rows[i]

    return encode_with_probs(
        seg, probs_fn, desc=None, alphabet_size=engine.vocab
    )


def _decode_segment_row(
    engine: StaticACEngine,
    arr: np.ndarray,
    seg_payload: bytes,
    *,
    start: int,
    n_seg: int,
):
    """W-semantic decode: one newly known token per W=1 kernel step."""
    engine.reset_segment(arr, start)

    def probs_fn(i: int, buf):
        if i == 0:
            return engine.carry_row()
        return engine.run_decode_row(int(buf[i - 1]))

    return decode_with_probs(
        seg_payload,
        int(n_seg),
        probs_fn,
        desc=None,
        alphabet_size=engine.vocab,
    )


def _decode_segment_row_gpu(
    engine: StaticACEngine,
    arr: np.ndarray,
    seg_payload: bytes,
    *,
    start: int,
    n_seg: int,
):
    """W-semantic one-row decode with tokens, probabilities and AC on HBM."""
    from .gpu_ac import GpuRangeCoder

    engine.reset_segment(arr, start)
    n = int(n_seg)
    if n <= 0:
        return b"" if engine.vocab <= 256 else []
    out = torch.empty(n, dtype=torch.int64, device=engine.device)
    capacity = max(engine.every * 48 + 4096, len(seg_payload) * 8 + 64)
    coder = engine._row_gpu_coder
    if coder is None or int(coder.max_bits) < capacity:
        coder = GpuRangeCoder(
            engine.device,
            max_bits=capacity,
            vocab=engine.vocab,
        )
        engine._row_gpu_coder = coder
        engine._row_decode_graph = None
        engine._row_decode_graph_k = 0
        engine._row_decode_tokens = None
        engine._row_decode_graph_failed = False
        engine._row_decode_graph_has_forward = False
        engine._row_fused_commit = False
    unroll = min(64, _env_int("XSA_AC_DECODE_UNROLL", 1))
    if (
        unroll > 1
        and engine._row_decode_graph is None
        and not engine._row_decode_graph_failed
        and n > 1
    ):
        try:
            coder.begin_decode(bytes(max(64, unroll * 8 + 8)))
            engine.capture_row_decode_graph(coder)
        except Exception as exc:  # pragma: no cover - CUDA/runtime dependent
            engine._row_decode_graph = None
            engine._row_decode_graph_k = 0
            engine._row_decode_tokens = None
            engine._row_decode_graph_failed = True
            engine._row_decode_graph_has_forward = False
            print(
                f"[AC incr] GPU decode graph capture failed "
                f"({exc!r}); using GPU one-row loop",
                file=sys.stderr,
                flush=True,
            )
        engine.reset_segment(arr, start)
    coder.begin_decode(seg_payload)
    out[0:1].copy_(coder.decode_symbol(engine.carry_row_dev()))
    i = 1
    graph_k = int(engine._row_decode_graph_k)
    if engine._row_decode_graph is not None and graph_k >= 1:
        assert engine.row_state is not None
        while i + graph_k <= n:
            engine.row_state.token.copy_(out[i - 1 : i].view(1, 1))
            engine.row_state.pos.fill_(engine._row_pos)
            if not engine._row_decode_graph_has_forward:
                engine._row_forward_probs()
            engine.replay_row_decode_graph(out, i)
            i += graph_k
    while i < n:
        engine.run_decode_row_dev(out[i - 1 : i])
        out[i : i + 1].copy_(
            coder.decode_symbol(engine._row_probs_dev.view(-1))
        )
        i += 1
    host = out.detach().cpu().numpy()
    if engine.vocab <= 256:
        return bytes(host.astype(np.uint8).tolist())
    return host.astype(np.int64).tolist()


def _decode_segment_incremental(
    engine: StaticACEngine,
    arr: np.ndarray,
    seg_payload: bytes,
    *,
    start: int,
    n_seg: int,
):
    """Windowed decode of one segment (same rows as encode).

    Default W>1 path (``XSA_AC_DECODE_ROW=1``) steps one confirmed token on
    the W=1 kernel. ``XSA_AC_DECODE_ROW=0`` reruns a full-attention W-step
    per newly confirmed prefix token. Draft (``XSA_AC_DRAFT=1``) only
    applies to the full-window verify path.
    """
    if engine._gpu_ac:
        return _decode_segment_gpu(
            engine, arr, seg_payload, start=start, n_seg=n_seg
        )
    if engine.use_decode_row and engine._gpu_row_ac:
        return _decode_segment_row_gpu(
            engine, arr, seg_payload, start=start, n_seg=n_seg
        )
    if engine.use_decode_row:
        return _decode_segment_row(
            engine, arr, seg_payload, start=start, n_seg=n_seg
        )
    engine.reset_segment(arr, start)
    win = engine.win
    use_draft = engine.use_draft and win > 1
    st: dict = {
        "w0": -1,
        "tokens": None,
        "rows": None,
        "n_actual": 0,
        "carry": engine.carry_row(),
    }

    def _ensure_rows(confirmed: int, buf, w0: int) -> np.ndarray:
        """Return verify rows whose prefix ``tokens[0:confirmed]`` matches."""
        tokens = st["tokens"]
        if tokens is not None and st["rows"] is not None:
            while int(st["n_actual"]) < confirmed:
                j = int(st["n_actual"])
                if int(tokens[j]) != int(buf[w0 + j]):
                    break
                st["n_actual"] = j + 1
            if int(st["n_actual"]) >= confirmed:
                return st["rows"]
        if tokens is None:
            tokens = np.zeros(win, dtype=np.int64)
            st["tokens"] = tokens
        tokens[:confirmed] = np.asarray(buf[w0 : w0 + confirmed], dtype=np.int64)
        st["n_actual"] = confirmed
        if use_draft and confirmed < win:
            tokens[confirmed:] = 0
            drows = engine.draft_window(tokens, w0)
            for loc in range(max(confirmed, 1), win):
                tokens[loc] = int(np.argmax(drows[loc - 1]))
        st["rows"] = engine.run_window(tokens, w0)
        return st["rows"]

    def probs_fn(i: int, buf):
        local = i - st["w0"] if st["w0"] >= 0 else win
        if local >= win:
            # Close the previous (fully decoded) window and open one at i.
            # First row of the new window is the carry — no step yet.
            if st["w0"] >= 0:
                valid = min(win, int(n_seg) - st["w0"])
                rows = _ensure_rows(valid, buf, st["w0"])
                st["carry"] = rows[valid - 1]
                if valid == win:
                    engine.advance_window()
            st["w0"] = i
            st["tokens"] = None
            st["rows"] = None
            st["n_actual"] = 0
            return st["carry"]
        # P(symbol i) = rows[local-1] with tokens[0:local] = buf[w0:i].
        return _ensure_rows(local, buf, st["w0"])[local - 1]

    return decode_with_probs(
        seg_payload,
        int(n_seg),
        probs_fn,
        desc=None,
        alphabet_size=engine.vocab,
    )


def _require_no_lora(lora) -> None:
    if lora is not None:
        raise RuntimeError(
            "incremental AC path does not support LoRA adapters "
            "(prefill/extend has no LoRA hooks); unset XSA_LORA_PATH"
        )


def compress_full_sha_incremental(
    model: GPT,
    data: bytes | np.ndarray,
    *,
    cfg: XsaTttConfig,
    device: torch.device,
    online_retrain_enabled: bool = True,
    progress: bool = True,
    verify_decode: bool = False,
    verify_end: bool = False,  # unsupported alias — mapped to verify_decode
    row_batch: int = 1024,  # accepted for signature parity; unused
    decoded_path: str | Path | None = None,
) -> dict[str, Any]:
    """Full-corpus AC via the shared incremental prob path.

    Framing is identical to ``compress_full_sha_lockstep`` (``<u64 len><seg>``
    per retrain segment), but every row comes from prefill+extend, so the
    bitstream is decodable *blind* with ``decompress_full_sha_incremental``.

    ``verify_decode=True`` additionally decodes every segment with a fresh
    stepper (the standalone decoder's exact op sequence) and SHA-checks the
    stream — roughly doubles the extend cost.
    """
    del row_batch
    do_seg_verify = bool(verify_decode) or bool(verify_end)

    vocab_size = int(cfg.vocab_size)
    arr = _as_symbol_array(data, vocab_size=vocab_size)
    if vocab_size <= 256:
        symbols: bytes | np.ndarray = (
            bytes(data)
            if isinstance(data, (bytes, bytearray))
            else (arr.tobytes() if not isinstance(arr, np.memmap) else bytes(arr))
        )
    else:
        symbols = arr

    n = int(arr.shape[0])
    every = max(1, int(cfg.online_retrain_every))
    seed = int(cfg.seed)
    lora, optimizer = _init_online_state(model, cfg, device, online_retrain_enabled)
    _require_no_lora(lora)
    burst_state: dict = {}

    model.eval()
    engine = StaticACEngine(model, cfg, device)

    src_sha = hashlib.sha256(_symbols_for_hash(arr)).hexdigest()
    dec_hasher = hashlib.sha256() if do_seg_verify else None
    out_parts = bytearray()
    seg_rate_rows: list[dict[str, Any]] = []
    payload_bytes = 0
    retrain_count = 0
    retrain_ce_sum = 0.0
    xm_counts: dict[int, int] = {}
    first_mismatch: int | None = None
    decoded_out: Path | None = (
        Path(decoded_path) if decoded_path and do_seg_verify else None
    )
    decoded_bytes_written = 0
    dec_fp = None
    t_start = time.time()

    indices = range(0, n, every)
    if progress:
        from tqdm import tqdm

        n_win_steps = (every + engine.win - 1) // engine.win
        dec_note = (
            f", decode={engine.decode_kind()}"
            if engine.use_decode_row
            else ""
        )
        enc_note = ", one-row encode" if engine.encode_row else (
            f"~{n_win_steps} full steps/seg"
        )
        print(
            f"[AC incr] starting ({n:,} sym, segment={every}, V={vocab_size}, "
            f"{'encode+decode' if do_seg_verify else 'encode-only'}, "
            f"windowed prob path W={engine.win}{dec_note}, "
            f"{enc_note}; bar moves after each segment)…",
            file=sys.stderr,
            flush=True,
        )
        indices = tqdm(
            indices,
            total=(n + every - 1) // every,
            desc="AC incr",
            unit="seg",
            file=sys.stderr,
            dynamic_ncols=True,
            mininterval=1.0,
            disable=False,
        )

    try:
        if decoded_out is not None:
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
                engine.refresh_ac_weights()
                xm_last = getattr(model, "_xm_last", None)
                if xm_last is not None and int(xm_last.get("end", -1)) == int(start):
                    chosen = int(xm_last.get("chosen", 0))
                    xm_counts[chosen] = xm_counts.get(chosen, 0) + 1
                    if progress and hasattr(indices, "set_postfix_str"):
                        indices.set_postfix_str(
                            _xm_share_str(xm_counts), refresh=False
                        )

            end = min(n, start + every)
            seg = symbols[start:end]
            seg_payload = _encode_segment_incremental(
                engine, arr, seg, start=start
            )
            if do_seg_verify:
                seg_decoded = _decode_segment_incremental(
                    engine,
                    arr,
                    seg_payload,
                    start=start,
                    n_seg=end - start,
                )
                if first_mismatch is None:
                    first_mismatch = _seg_roundtrip_mismatch(
                        seg, seg_decoded, vocab_size=vocab_size, start=start
                    )
                assert dec_hasher is not None
                dec_blob = _seg_decoded_to_bytes(seg_decoded, vocab_size=vocab_size)
                dec_hasher.update(dec_blob)
                if dec_fp is not None:
                    dec_fp.write(dec_blob)
                    dec_fp.flush()
                    decoded_bytes_written += len(dec_blob)

            out_parts.extend(struct.pack("<Q", len(seg_payload)))
            out_parts.extend(seg_payload)
            payload_bytes += len(seg_payload)
            seg_rate_rows.append(
                {"pos": int(end), "bpb": len(seg_payload) * 8.0 / max(1, end - start)}
            )
            win = max(1, (1 << 20) // every)
            if progress and len(seg_rate_rows) % win == 0:
                recent = seg_rate_rows[-win:]
                recent_bpb = sum(r["bpb"] for r in recent) / len(recent)
                cum_bpb = payload_bytes * 8.0 / max(1, end)
                lr_now = _scheduled_replenish_lr(cfg, int(end), n_total=n)
                sps = end / max(1e-9, time.time() - t_start)
                xm_note = f" xm[{_xm_share_str(xm_counts)}]" if xm_counts else ""
                from tqdm import tqdm as _tqdm

                _tqdm.write(
                    f"[ac] pos={end / 1e6:.1f}MB cum_bpb={cum_bpb:.4f} "
                    f"last{win}seg={recent_bpb:.4f} lr={lr_now:.3e} "
                    f"sym/s={sps:,.0f}" + xm_note,
                    file=sys.stderr,
                )
    finally:
        if dec_fp is not None:
            dec_fp.close()

    dec_sha: str | None = None
    sha_ok: bool | None = None
    if do_seg_verify:
        assert dec_hasher is not None
        dec_sha = dec_hasher.hexdigest()
        sha_ok = dec_sha == src_sha and first_mismatch is None

    bpb = (payload_bytes * 8) / max(1, n)
    return {
        "payload": bytes(out_parts),
        "n_bytes": n,
        "n_symbols": n,
        "payload_bytes": payload_bytes,
        "bpb": bpb,
        "bits_per_symbol": bpb,
        "vocab_size": vocab_size,
        "retrain_count": retrain_count,
        "retrain_ce_mean": (
            retrain_ce_sum / max(1, retrain_count) if retrain_count else float("nan")
        ),
        "infer_mode": "segmented_incremental_sha",
        "chunk_bpb_rows": seg_rate_rows,
        "sha256": src_sha,
        "decoded_sha256": dec_sha,
        "sha256_ok": sha_ok,
        "roundtrip_ok": sha_ok,
        "first_mismatch": first_mismatch,
        "segment_bytes": every,
        "verify_decode": do_seg_verify,
        "verify_end": False,
        "decoded_path": str(decoded_out) if decoded_out is not None else None,
        "decoded_bytes": decoded_bytes_written if decoded_out is not None else None,
    }


def iter_framed_segments(payload: bytes) -> Iterator[bytes]:
    """Walk the ``<u64 len><segment bytes>`` framing of a *_fullsha payload."""
    off = 0
    total = len(payload)
    while off < total:
        if off + 8 > total:
            raise ValueError(f"truncated segment header at offset {off}")
        (ln,) = struct.unpack_from("<Q", payload, off)
        off += 8
        if off + ln > total:
            raise ValueError(
                f"truncated segment at offset {off}: need {ln}, have {total - off}"
            )
        yield payload[off : off + ln]
        off += ln


def decompress_full_sha_incremental(
    model: GPT,
    payload: bytes,
    n_symbols: int,
    *,
    cfg: XsaTttConfig,
    device: torch.device,
    online_retrain_enabled: bool = True,
    progress: bool = True,
    decoded_path: str | Path | None = None,
) -> dict[str, Any]:
    """Blind decode of a segment-framed incremental bitstream (no source).

    Mirrors ``compress_full_sha_incremental`` exactly: same retrain boundaries
    on the decoded prefix, same rotary warm, same prefill+extend op sequence —
    so the model trajectory and every prob row match the encoder's.
    """
    vocab_size = int(cfg.vocab_size)
    n = int(n_symbols)
    if n <= 0:
        raise ValueError("decode needs n_symbols > 0")
    every = max(1, int(cfg.online_retrain_every))
    seed = int(cfg.seed)
    lora, optimizer = _init_online_state(model, cfg, device, online_retrain_enabled)
    _require_no_lora(lora)
    burst_state: dict = {}

    model.eval()
    engine = StaticACEngine(model, cfg, device)

    arr = np.zeros(n, dtype=_sym_dtype(vocab_size))
    hasher = hashlib.sha256()
    n_segments = (n + every - 1) // every
    xm_counts: dict[int, int] = {}
    retrain_count = 0
    decoded_n = 0
    decoded_out = Path(decoded_path) if decoded_path else None
    dec_fp = None
    t_start = time.time()

    seg_iter = enumerate(iter_framed_segments(payload))
    if progress:
        from tqdm import tqdm

        print(
            f"[AC incr] blind decode ({n:,} sym, {n_segments:,} segments, "
            f"V={vocab_size}, draft={engine.draft_kind()}, "
            f"decode={engine.decode_kind()})…",
            file=sys.stderr,
            flush=True,
        )
        seg_iter = tqdm(
            seg_iter,
            total=n_segments,
            desc="AC incr decode",
            unit="seg",
            file=sys.stderr,
            dynamic_ncols=True,
            mininterval=1.0,
            disable=False,
        )

    try:
        if decoded_out is not None:
            decoded_out.parent.mkdir(parents=True, exist_ok=True)
            dec_fp = open(decoded_out, "wb")

        for seg_i, seg_payload in seg_iter:
            start = seg_i * every
            if start >= n:
                raise ValueError(
                    f"bitstream has more segments than n_symbols={n} allows"
                )
            scheduled_steps = _online_steps_at(start, cfg, every)
            if (
                online_retrain_enabled
                and optimizer is not None
                and start > 0
                and scheduled_steps > 0
            ):
                online_retrain(
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
                model.eval()
                engine.refresh_ac_weights()
                xm_last = getattr(model, "_xm_last", None)
                if xm_last is not None and int(xm_last.get("end", -1)) == int(start):
                    chosen = int(xm_last.get("chosen", 0))
                    xm_counts[chosen] = xm_counts.get(chosen, 0) + 1
                    if progress and hasattr(seg_iter, "set_postfix_str"):
                        seg_iter.set_postfix_str(
                            _xm_share_str(xm_counts), refresh=False
                        )

            end = min(n, start + every)
            steps_before = engine.n_steps
            drafts_before = engine.n_drafts
            seg_decoded = _decode_segment_incremental(
                engine,
                arr,
                seg_payload,
                start=start,
                n_seg=end - start,
            )
            steps_delta = engine.n_steps - steps_before
            drafts_delta = engine.n_drafts - drafts_before
            seg_arr = np.asarray(
                bytearray(seg_decoded)
                if isinstance(seg_decoded, (bytes, bytearray))
                else seg_decoded,
                dtype=_sym_dtype(vocab_size),
            ).reshape(-1)
            if seg_arr.shape[0] != end - start:
                raise ValueError(
                    f"segment {seg_i}: decoded {seg_arr.shape[0]} sym, "
                    f"want {end - start}"
                )
            arr[start:end] = seg_arr
            decoded_n = end
            blob = _seg_decoded_to_bytes(seg_decoded, vocab_size=vocab_size)
            hasher.update(blob)
            if dec_fp is not None:
                dec_fp.write(blob)
                dec_fp.flush()

            log_every = max(1, (1 << 20) // every)
            n_this = end - start
            spec = n_this / max(1, steps_delta)
            draft_note = (
                f" drafts={drafts_delta}" if engine.use_draft else ""
            )
            if progress and hasattr(seg_iter, "set_postfix_str"):
                seg_iter.set_postfix_str(
                    f"sym/step={spec:.2f} W={engine.win} "
                    f"steps={steps_delta}/{n_this}{draft_note}",
                    refresh=False,
                )
            if progress and (seg_i < 4 or (seg_i + 1) % log_every == 0):
                sps = end / max(1e-9, time.time() - t_start)
                lr_now = _scheduled_replenish_lr(cfg, int(end), n_total=n)
                xm_note = f" xm[{_xm_share_str(xm_counts)}]" if xm_counts else ""
                from tqdm import tqdm as _tqdm

                _tqdm.write(
                    f"[ac-dec] pos={end / 1e6:.1f}MB seg={seg_i} "
                    f"lr={lr_now:.3e} sym/s={sps:,.0f} "
                    f"sym/step={spec:.2f} (W={engine.win} "
                    f"steps={steps_delta}/{n_this}{draft_note})"
                    + xm_note,
                    file=sys.stderr,
                )
    finally:
        if dec_fp is not None:
            dec_fp.close()

    if decoded_n != n:
        raise ValueError(f"decoded {decoded_n} symbols, want {n}")
    return {
        "n_symbols": n,
        "payload_bytes": len(payload),
        "segments": n_segments,
        "retrain_count": retrain_count,
        "decoded_sha256": hasher.hexdigest(),
        "decoded_path": str(decoded_out) if decoded_out is not None else None,
        "infer_mode": "segmented_incremental_sha",
        "decoded": None if decoded_out is not None else _symbols_for_hash(arr),
    }
