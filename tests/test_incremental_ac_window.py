"""CPU lockstep + step-count checks for the windowed incremental AC path."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
for _p in (_SRC, _SRC / "model"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from xsa_ttt.config import make_config  # noqa: E402
from xsa_ttt.incremental import (  # noqa: E402
    StaticACEngine,
    _decode_segment_incremental,
    _encode_segment_incremental,
    _segment_rows_tf,
    compile_static_extend,
)
from xsa_ttt.persistent_step import can_persistent, try_persistent_extend  # noqa: E402
from xsa_ttt.model import build_model  # noqa: E402


def _engine(
    win: int,
    *,
    block: int = 64,
    draft: bool = False,
    draft_pool: int = 128,
    draft_window: int = 32,
    decode_row: bool = False,
    encode_row: bool = False,
):
    os.environ["XSA_AC_WINDOW"] = str(win)
    os.environ["XSA_AC_GRAPH"] = "0"
    os.environ["XSA_AC_FUSED"] = "0"
    os.environ["XSA_AC_GPU_AC"] = "0"
    os.environ["XSA_AC_DRAFT"] = "1" if draft else "0"
    os.environ["XSA_AC_DRAFT_POOL"] = str(draft_pool)
    os.environ["XSA_AC_DRAFT_WINDOW"] = str(draft_window)
    os.environ["XSA_AC_DRAFT_LOCAL"] = "0"
    os.environ["XSA_AC_DECODE_ROW"] = "1" if decode_row else "0"
    os.environ["XSA_AC_ENCODE_ROW"] = "1" if encode_row else "0"
    cfg = make_config(profile="smoke")
    cfg.block_size = block
    cfg.online_retrain_every = block
    cfg.use_bf16 = False
    cfg.gradient_checkpointing = False
    device = torch.device("cpu")
    torch.manual_seed(0)
    model = build_model(cfg, device=device)
    model.eval()
    return StaticACEngine(model, cfg, device), cfg, device


def _roundtrip(
    win: int,
    data: bytes,
    *,
    block: int = 64,
    draft: bool = False,
    draft_pool: int = 128,
    draft_window: int = 32,
    decode_row: bool = False,
    encode_row: bool = False,
):
    engine, _cfg, _device = _engine(
        win,
        block=block,
        draft=draft,
        draft_pool=draft_pool,
        draft_window=draft_window,
        decode_row=decode_row,
        encode_row=encode_row,
    )
    arr = np.frombuffer(data, dtype=np.uint8).astype(np.int64)
    payload = _encode_segment_incremental(engine, arr, data, start=0)
    encode_steps = engine.n_steps
    engine.n_steps = 0
    engine.n_drafts = 0
    decoded = _decode_segment_incremental(
        engine, arr, payload, start=0, n_seg=len(data)
    )
    return bytes(decoded), encode_steps, engine.n_steps, engine.n_drafts


@pytest.mark.parametrize("win", [1, 4, 8])
def test_windowed_roundtrip_random(win: int) -> None:
    rng = np.random.default_rng(0)
    data = bytes(rng.integers(0, 256, size=48, dtype=np.uint8))
    out, enc_steps, dec_steps, drafts = _roundtrip(win, data, draft=False)
    assert out == data
    n = len(data)
    assert enc_steps == (n + win - 1) // win
    assert drafts == 0
    # Carry covers the first symbol of each window; every later symbol
    # (including the close-run that produces the next carry) is one step.
    assert dec_steps == n - 1
    assert dec_steps <= n


def test_windowed_roundtrip_mismatch_heavy() -> None:
    """Low-predictability bytes must not blow up to ~2 + (W-1) steps/window."""
    rng = np.random.default_rng(1)
    data = bytes(rng.integers(0, 256, size=64, dtype=np.uint8))
    out, enc_steps, dec_steps, drafts = _roundtrip(8, data, draft=False)
    assert out == data
    assert enc_steps == 8
    assert drafts == 0
    # Old broken argmax-chain paid ~9 steps/window (2 open + 7 misses).
    assert dec_steps == 63
    assert dec_steps < 8 * 9


def test_windowed_roundtrip_patterned() -> None:
    data = (b"abc" * 20)[:48]
    out, _enc, dec_steps, drafts = _roundtrip(4, data, draft=False)
    assert out == data
    assert drafts == 0
    assert dec_steps == 47


@pytest.mark.parametrize("win", [4, 8])
def test_decode_row_roundtrip_random(win: int) -> None:
    """W-batched encode + one-row decode must lockstep on CPU eager."""
    rng = np.random.default_rng(0)
    data = bytes(rng.integers(0, 256, size=48, dtype=np.uint8))
    out, enc_steps, dec_steps, drafts = _roundtrip(
        win, data, draft=False, decode_row=True
    )
    assert out == data
    n = len(data)
    assert enc_steps == (n + win - 1) // win
    assert drafts == 0
    assert dec_steps == n - 1


def test_decode_row_roundtrip_patterned() -> None:
    data = (b"abc" * 20)[:48]
    out, enc_steps, dec_steps, drafts = _roundtrip(
        4, data, draft=False, decode_row=True
    )
    assert out == data
    assert enc_steps == 12
    assert drafts == 0
    assert dec_steps == 47


def test_encode_row_decode_row_roundtrip() -> None:
    """Canonical one-row both sides (fallback if batched frequencies diverge)."""
    rng = np.random.default_rng(6)
    data = bytes(rng.integers(0, 256, size=40, dtype=np.uint8))
    out, enc_steps, dec_steps, drafts = _roundtrip(
        8, data, decode_row=True, encode_row=True
    )
    assert out == data
    assert drafts == 0
    assert enc_steps == len(data) - 1
    assert dec_steps == len(data) - 1


def _freq_tables(rows: np.ndarray, vocab: int) -> list[np.ndarray]:
    from arithmetic_coder_lm import _scale_cum_np

    out = []
    for i in range(rows.shape[0]):
        cum, _total = _scale_cum_np(rows[i], alphabet_size=vocab)
        out.append(np.asarray(cum))
    return out


def test_decode_row_integer_frequencies() -> None:
    """W-batched encode frequencies vs one-row decode at every position."""
    win = 4
    engine, _cfg, _device = _engine(win, decode_row=True)
    rng = np.random.default_rng(7)
    data = bytes(rng.integers(0, 256, size=32, dtype=np.uint8))
    arr = np.frombuffer(data, dtype=np.uint8).astype(np.int64)
    rows_w = _segment_rows_tf(engine, arr, arr, start=0)
    engine.n_steps = 0
    engine.reset_segment(arr, 0)
    rows_r = np.empty_like(rows_w)
    rows_r[0] = engine.carry_row()
    for i in range(len(arr) - 1):
        rows_r[i + 1] = engine.run_decode_row(int(arr[i]))
    freq_w = _freq_tables(rows_w, engine.vocab)
    freq_r = _freq_tables(rows_r, engine.vocab)
    for i, (cw, cr) in enumerate(zip(freq_w, freq_r)):
        assert np.array_equal(cw, cr), f"integer freq mismatch at pos {i}"


def test_decode_row_integer_frequencies_patterned() -> None:
    engine, _cfg, _device = _engine(8, decode_row=True)
    data = (b"\x00\xffabcXYZ" * 8)[:48]
    arr = np.frombuffer(data, dtype=np.uint8).astype(np.int64)
    rows_w = _segment_rows_tf(engine, arr, arr, start=0)
    engine.reset_segment(arr, 0)
    rows_r = np.empty_like(rows_w)
    rows_r[0] = engine.carry_row()
    for i in range(len(arr) - 1):
        rows_r[i + 1] = engine.run_decode_row(int(arr[i]))
    freq_w = _freq_tables(rows_w, engine.vocab)
    freq_r = _freq_tables(rows_r, engine.vocab)
    for i, (cw, cr) in enumerate(zip(freq_w, freq_r)):
        assert np.array_equal(cw, cr), f"integer freq mismatch at pos {i}"


def test_decode_row_frequencies_after_prefill() -> None:
    """Prefix KV must be mirrored into row_state or later rows diverge."""
    win = 4
    block = 16
    engine, _cfg, _device = _engine(win, block=block, decode_row=True)
    rng = np.random.default_rng(8)
    data = bytes(rng.integers(0, 256, size=block * 2, dtype=np.uint8))
    arr = np.frombuffer(data, dtype=np.uint8).astype(np.int64)
    start = block
    seg = arr[start : start + block]
    rows_w = _segment_rows_tf(engine, arr, seg, start=start)
    engine.reset_segment(arr, start)
    rows_r = np.empty_like(rows_w)
    rows_r[0] = engine.carry_row()
    for i in range(len(seg) - 1):
        rows_r[i + 1] = engine.run_decode_row(int(seg[i]))
    freq_w = _freq_tables(rows_w, engine.vocab)
    freq_r = _freq_tables(rows_r, engine.vocab)
    for i, (cw, cr) in enumerate(zip(freq_w, freq_r)):
        assert np.array_equal(cw, cr), f"integer freq mismatch at pos {i}"


def test_decode_row_allocates_win1_state() -> None:
    engine, _cfg, _device = _engine(4, decode_row=True)
    assert engine.use_decode_row is True
    assert engine.row_state is not None
    assert engine.row_state.win == 1
    assert engine.state.win == 4
    assert engine.decode_kind() == "row-eager"


def test_decode_row_w64_canonical_roundtrip() -> None:
    """W=64 one-row encode+decode (canonical lockstep if batched freqs diverge)."""
    rng = np.random.default_rng(9)
    data = bytes(rng.integers(0, 256, size=80, dtype=np.uint8))
    out, enc_steps, dec_steps, drafts = _roundtrip(
        64, data, block=128, decode_row=True, encode_row=True
    )
    assert out == data
    assert drafts == 0
    assert enc_steps == 79
    assert dec_steps == 79


def test_decode_row_w64_batched_freq_near_match() -> None:
    """W=64 GEMM vs one-row GEMV can differ at ``int(p*1e6+0.5)`` ties.

    A 1-count miss desyncs AC, so production lockstep uses the W-encode
    megakernel (``XSA_AC_MEGA_ENCODE=1``) or ``XSA_AC_ENCODE_ROW=1``.
    """
    engine, _cfg, _device = _engine(64, block=128, decode_row=True)
    rng = np.random.default_rng(9)
    data = bytes(rng.integers(0, 256, size=80, dtype=np.uint8))
    arr = np.frombuffer(data, dtype=np.uint8).astype(np.int64)
    rows_w = _segment_rows_tf(engine, arr, arr, start=0)
    engine.reset_segment(arr, 0)
    rows_r = np.empty_like(rows_w)
    rows_r[0] = engine.carry_row()
    for i in range(len(arr) - 1):
        rows_r[i + 1] = engine.run_decode_row(int(arr[i]))
    n_mis = 0
    max_abs = 0.0
    max_count = 0
    for i in range(len(arr)):
        max_abs = max(max_abs, float(np.max(np.abs(rows_w[i] - rows_r[i]))))
        cw = _freq_tables(rows_w[i : i + 1], engine.vocab)[0]
        cr = _freq_tables(rows_r[i : i + 1], engine.vocab)[0]
        if not np.array_equal(cw, cr):
            n_mis += 1
            max_count = max(max_count, int(np.max(np.abs(cw - cr))))
    assert max_abs < 1e-8
    assert n_mis <= 2
    assert max_count <= 1


def test_decode_row_off_keeps_single_state() -> None:
    engine, _cfg, _device = _engine(4, decode_row=False)
    assert engine.use_decode_row is False
    assert engine.row_state is None
    assert engine.decode_kind() == "window"


@pytest.mark.parametrize("win", [4, 8])
def test_sparse_draft_roundtrip(win: int) -> None:
    """Draft+verify must stay lockstep with teacher-forced encode."""
    rng = np.random.default_rng(2)
    data = bytes(rng.integers(0, 256, size=48, dtype=np.uint8))
    out, enc_steps, dec_steps, drafts = _roundtrip(win, data, draft=True)
    assert out == data
    assert enc_steps == (48 + win - 1) // win
    assert drafts > 0
    assert dec_steps <= 47


def test_sparse_draft_patterned_roundtrip() -> None:
    data = (b"abc" * 20)[:48]
    out, _enc, dec_steps, drafts = _roundtrip(4, data, draft=True)
    assert out == data
    assert drafts > 0
    assert dec_steps <= 47


def test_hca_draft_pools_long_prefix() -> None:
    """Prefix longer than one pool block must still lockstep."""
    rng = np.random.default_rng(3)
    data = bytes(rng.integers(0, 256, size=200, dtype=np.uint8))
    out, enc_steps, dec_steps, drafts = _roundtrip(
        4, data, block=256, draft=True, draft_pool=32, draft_window=32
    )
    assert out == data
    assert enc_steps == 50
    assert drafts > 0
    assert dec_steps <= 199


def test_compile_static_extend_is_a_copy() -> None:
    """Factory returns a distinct callable; eager method is unchanged."""
    engine, _cfg, _device = _engine(4, draft=False)
    eager = engine.model.static_extend
    compiled = compile_static_extend(engine.model, engine.state, mode="default")
    assert compiled is not eager
    # Bound methods are new wrappers on each access; compare the function.
    assert engine.model.static_extend.__func__ is eager.__func__
    assert engine._compiled_extend is None
    assert engine.step_kind() == "eager"


def test_compile_env_stays_eager_on_cpu() -> None:
    """XSA_AC_COMPILE is CUDA-only; CPU tests must not pay compile."""
    os.environ["XSA_AC_COMPILE"] = "1"
    try:
        engine, _cfg, _device = _engine(4, draft=False)
        assert engine._compiled_extend is None
        assert engine.step_kind() == "eager"
    finally:
        os.environ["XSA_AC_COMPILE"] = "0"


def test_static_state_marks_buffers() -> None:
    """CPU mark is a no-op; the method must still walk every buffer."""
    engine, _cfg, _device = _engine(4, draft=False)
    engine.state.mark_static_addresses()
    assert engine.state.cand_prev_raw is not None
    assert engine.state.logits is not None


def test_prepare_ac_weights_cpu_logits() -> None:
    """Dtype-clean bank views stay within a tight bound of live fp32 params."""
    engine, _cfg, _device = _engine(4, draft=False)
    model = engine.model
    rng = np.random.default_rng(4)
    data = bytes(rng.integers(0, 256, size=16, dtype=np.uint8))
    arr = np.frombuffer(data, dtype=np.uint8).astype(np.int64)
    tokens = arr[: engine.win]
    engine.reset_segment(arr, 0)
    engine.run_window(tokens, 0)
    with_views = engine.state.logits.detach().clone()
    model._ac_qo_bank = None
    model._ac_kv_bank = None
    model._ac_mlp_up = None
    model._ac_mlp_down = None
    engine.reset_segment(arr, 0)
    engine.run_window(tokens, 0)
    no_views = engine.state.logits.detach().clone()
    # CPU fp32 clones; bound is the plan's "tight bf16" ceiling.
    assert (with_views - no_views).abs().max().item() < 2e-2


def test_fused_env_stays_eager_on_cpu() -> None:
    """XSA_AC_FUSED is CUDA-only; CPU tests must not leave the eager path."""
    os.environ["XSA_AC_FUSED"] = "1"
    os.environ["XSA_AC_WINDOW"] = "4"
    os.environ["XSA_AC_GRAPH"] = "0"
    os.environ["XSA_AC_DRAFT"] = "0"
    os.environ["XSA_AC_DECODE_ROW"] = "0"
    try:
        cfg = make_config(profile="smoke")
        cfg.block_size = 64
        cfg.online_retrain_every = 64
        cfg.use_bf16 = False
        cfg.gradient_checkpointing = False
        device = torch.device("cpu")
        torch.manual_seed(0)
        model = build_model(cfg, device=device)
        model.eval()
        engine = StaticACEngine(model, cfg, device)
        assert engine._fused is False
        assert engine._graph is None
        assert engine.step_kind() == "eager"
    finally:
        os.environ["XSA_AC_FUSED"] = "0"


def test_static_extend_fused_cpu_matches_eager() -> None:
    """CPU fused falls back to the eager slot path (bit-identical)."""
    engine, _cfg, _device = _engine(4, draft=False)
    st = engine.state
    st.set_step_mode(True)
    st.token.zero_()
    st.pos.fill_(0)
    engine.model.static_extend(st)
    eager = st.logits.detach().clone()
    for sk in st.slots:
        if sk.k is not None:
            sk.k.zero_()
            sk.v.zero_()
    st.cand_prev_raw.zero_()
    engine.model.static_extend_fused(st)
    fused = st.logits.detach().clone()
    assert torch.equal(eager, fused)


def test_persistent_stays_off_on_cpu() -> None:
    engine, _cfg, _device = _engine(1, draft=False)
    st = engine.state
    st.set_step_mode(True)
    assert can_persistent(engine.model, st) is False
    assert try_persistent_extend(engine.model, st) is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
def test_fused_vs_eager_cuda_atol() -> None:
    """Fused vs eager logits on a short smoke model (bf16-Triton atol)."""
    os.environ["XSA_AC_WINDOW"] = "4"
    os.environ["XSA_AC_GRAPH"] = "0"
    os.environ["XSA_AC_FUSED"] = "0"
    os.environ["XSA_AC_DRAFT"] = "0"
    cfg = make_config(profile="smoke")
    cfg.block_size = 64
    cfg.online_retrain_every = 64
    cfg.use_bf16 = False
    cfg.gradient_checkpointing = False
    device = torch.device("cuda")
    torch.manual_seed(0)
    model = build_model(cfg, device=device)
    model.eval()
    model.prepare_ac_compute_weights(torch.float32)
    state = model.make_static_state(cfg.block_size + 8, device, win=4)
    state.set_step_mode(True)
    state.token.copy_(
        torch.arange(4, device=device, dtype=torch.long).view(1, -1)
    )
    state.pos.fill_(0)
    model.static_extend(state)
    eager = state.logits.detach().float().clone()
    for sk in state.slots:
        if sk.k is not None:
            sk.k.zero_()
            sk.v.zero_()
    state.cand_prev_raw.zero_()
    model.static_extend_fused(state)
    fused = state.logits.detach().float()
    assert (fused - eager).abs().max().item() < 0.5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
def test_persistent_vs_fused_cuda_w1() -> None:
    """W=1 persistent stack vs per-slot fused (same fused numerics family)."""
    os.environ["XSA_AC_WINDOW"] = "1"
    os.environ["XSA_AC_GRAPH"] = "0"
    os.environ["XSA_AC_FUSED"] = "0"
    os.environ["XSA_AC_DRAFT"] = "0"
    os.environ["XSA_AC_PERSISTENT"] = "0"
    cfg = make_config(profile="smoke")
    cfg.block_size = 64
    cfg.online_retrain_every = 64
    cfg.use_bf16 = False
    cfg.gradient_checkpointing = False
    device = torch.device("cuda")
    torch.manual_seed(0)
    model = build_model(cfg, device=device)
    model.eval()
    model.prepare_ac_compute_weights(torch.float32)
    state = model.make_static_state(cfg.block_size + 8, device, win=1)
    state.set_step_mode(True)
    state.token.fill_(3)
    state.pos.fill_(0)
    model.static_extend_fused(state)
    slot_fused = state.logits.detach().float().clone()
    for sk in state.slots:
        if sk.k is not None:
            sk.k.zero_()
            sk.v.zero_()
    state.cand_prev_raw.zero_()
    os.environ["XSA_AC_PERSISTENT"] = "1"
    model._persist_ws = None
    model._persist_disabled = False
    model.static_extend_fused(state)
    persist = state.logits.detach().float()
    os.environ["XSA_AC_PERSISTENT"] = "1"
    if getattr(model, "_persist_disabled", False):
        pytest.skip("persistent kernel unavailable on this GPU")
    assert (persist - slot_fused).abs().max().item() < 0.5


def test_gpu_ac_stays_off_on_cpu() -> None:
    os.environ["XSA_AC_GPU_AC"] = "1"
    try:
        engine, _cfg, _device = _engine(1, draft=False)
        assert engine._gpu_ac is False
        row_engine, _cfg, _device = _engine(4, decode_row=True)
        assert row_engine._gpu_row_ac is False
    finally:
        os.environ["XSA_AC_GPU_AC"] = "0"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
def test_gpu_ac_matches_host_coder() -> None:
    """Device WNC must match ``encode_with_probs`` / ``decode_with_probs``."""
    from arithmetic_coder_lm import decode_with_probs, encode_with_probs
    from xsa_ttt.gpu_ac import GpuRangeCoder, can_gpu_ac

    device = torch.device("cuda")
    if not can_gpu_ac(device):
        pytest.skip("Triton GPU AC unavailable")
    rng = np.random.default_rng(0)
    n, v = 64, 256
    rows = rng.random((n, v)).astype(np.float32)
    rows /= rows.sum(axis=1, keepdims=True)
    syms = rng.integers(0, v, size=n, dtype=np.int64)

    def probs_fn(i: int, _prefix):
        return rows[i]

    host = encode_with_probs(syms, probs_fn, alphabet_size=v)
    coder = GpuRangeCoder(device, max_bits=n * 48 + 4096, vocab=v)
    probs = torch.as_tensor(rows, device=device, dtype=torch.float32)
    tok = torch.as_tensor(syms, device=device, dtype=torch.int64)
    for i in range(n):
        coder.encode_symbol(probs[i], tok[i])
    gpu = coder.finish()
    assert gpu == host
    dec = GpuRangeCoder(device, max_bits=len(gpu) * 8 + 64, vocab=v)
    input_ptr = dec._in_bits.data_ptr()
    dec.begin_decode(gpu)
    assert dec._in_bits.data_ptr() == input_ptr
    out = torch.empty(n, dtype=torch.int64, device=device)
    for i in range(n):
        out[i : i + 1].copy_(dec.decode_symbol(probs[i]))
    assert np.array_equal(out.cpu().numpy(), syms)
    dec.begin_decode(gpu)
    assert dec._in_bits.data_ptr() == input_ptr
    host_dec = decode_with_probs(gpu, n, probs_fn, alphabet_size=v)
    assert bytes(host_dec) == bytes(syms.astype(np.uint8))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
@pytest.mark.parametrize(
    ("decode_k", "fused", "encode_row"),
    [(1, False, True), (4, True, True), (1, True, False)],
)
def test_gpu_decode_row_roundtrip_stays_on_device(
    decode_k: int, fused: bool, encode_row: bool
) -> None:
    """W>1 one-row decode must use the HBM coder and remain lockstep."""
    old = {
        name: os.environ.get(name)
        for name in (
            "XSA_AC_WINDOW",
            "XSA_AC_GRAPH",
            "XSA_AC_FUSED",
            "XSA_AC_GPU_AC",
            "XSA_AC_DECODE_ROW",
            "XSA_AC_ENCODE_ROW",
            "XSA_AC_DECODE_K",
            "XSA_AC_ATTN_SPLITS",
            "XSA_AC_GQA_DEDUP",
            "XSA_AC_STAGE_FUSION",
        )
    }
    try:
        os.environ["XSA_AC_WINDOW"] = "8"
        os.environ["XSA_AC_GRAPH"] = "0"
        os.environ["XSA_AC_FUSED"] = "1" if fused else "0"
        os.environ["XSA_AC_GPU_AC"] = "1"
        os.environ["XSA_AC_DECODE_ROW"] = "1"
        os.environ["XSA_AC_ENCODE_ROW"] = "1" if encode_row else "0"
        os.environ["XSA_AC_DECODE_K"] = str(decode_k)
        os.environ["XSA_AC_ATTN_SPLITS"] = "8"
        os.environ["XSA_AC_GQA_DEDUP"] = "1"
        os.environ["XSA_AC_STAGE_FUSION"] = "1"
        cfg = make_config(profile="smoke")
        cfg.block_size = 32
        cfg.online_retrain_every = 32
        cfg.use_bf16 = False
        cfg.gradient_checkpointing = False
        device = torch.device("cuda")
        torch.manual_seed(0)
        model = build_model(cfg, device=device)
        model.eval()
        engine = StaticACEngine(model, cfg, device)
        if not engine._gpu_row_ac:
            pytest.skip("GPU range coder unavailable")
        assert engine._gpu_ac is False
        rng = np.random.default_rng(12)
        data = bytes(rng.integers(0, 256, size=48, dtype=np.uint8))
        arr = np.frombuffer(data, dtype=np.uint8).astype(np.int64)
        for start in (0, 24):
            segment = data[start : start + 24]
            payload = _encode_segment_incremental(
                engine, arr, segment, start=start
            )
            decoded = _decode_segment_incremental(
                engine, arr, payload, start=start, n_seg=len(segment)
            )
            assert bytes(decoded) == segment
        if decode_k > 1:
            assert engine._gpu_row_ac is True
        if fused:
            persist_ws = getattr(model, "_persist_ws", None)
            assert persist_ws is not None
            assert getattr(persist_ws, "mega_attn", "") == "split8"
        if fused and not encode_row:
            encode_ws = getattr(model, "_encode_ws", None)
            assert encode_ws is not None
            assert getattr(encode_ws, "mega_attn", "") == "split8"
    finally:
        for name, value in old.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
def test_split_attention_w64_integer_frequencies() -> None:
    """Split W=64 encode and W=1 decode must produce identical AC tables."""
    names = (
        "XSA_AC_WINDOW",
        "XSA_AC_GRAPH",
        "XSA_AC_FUSED",
        "XSA_AC_GPU_AC",
        "XSA_AC_DECODE_ROW",
        "XSA_AC_ENCODE_ROW",
        "XSA_AC_MEGA_ENCODE",
        "XSA_AC_ATTN_SPLITS",
        "XSA_AC_GQA_DEDUP",
        "XSA_AC_STAGE_FUSION",
    )
    old = {name: os.environ.get(name) for name in names}
    try:
        os.environ["XSA_AC_WINDOW"] = "64"
        os.environ["XSA_AC_GRAPH"] = "0"
        os.environ["XSA_AC_FUSED"] = "1"
        os.environ["XSA_AC_GPU_AC"] = "0"
        os.environ["XSA_AC_DECODE_ROW"] = "1"
        os.environ["XSA_AC_ENCODE_ROW"] = "0"
        os.environ["XSA_AC_MEGA_ENCODE"] = "1"
        os.environ["XSA_AC_ATTN_SPLITS"] = "8"
        os.environ["XSA_AC_GQA_DEDUP"] = "1"
        cfg = make_config(profile="smoke")
        cfg.block_size = 128
        cfg.online_retrain_every = 128
        cfg.use_bf16 = True
        cfg.gradient_checkpointing = False
        device = torch.device("cuda")
        torch.manual_seed(0)
        model = build_model(cfg, device=device)
        model.eval()
        engine = StaticACEngine(model, cfg, device)
        rng = np.random.default_rng(13)
        arr = rng.integers(0, 256, size=80, dtype=np.uint8).astype(np.int64)
        rows_w = _segment_rows_tf(engine, arr, arr, start=0)
        engine.reset_segment(arr, 0)
        rows_r = np.empty_like(rows_w)
        rows_r[0] = engine.carry_row()
        for i in range(len(arr) - 1):
            rows_r[i + 1] = engine.run_decode_row(int(arr[i]))
        for i, (cw, cr) in enumerate(
            zip(
                _freq_tables(rows_w, engine.vocab),
                _freq_tables(rows_r, engine.vocab),
            )
        ):
            assert np.array_equal(cw, cr), f"integer freq mismatch at pos {i}"
        assert getattr(model._encode_ws, "mega_attn", "") == "split8"
        assert getattr(model._persist_ws, "mega_attn", "") == "split8"
    finally:
        for name, value in old.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
def test_split_attention_preserves_serial_bpb() -> None:
    """The changed reduction order must not materially move source BPB."""
    names = (
        "XSA_AC_WINDOW",
        "XSA_AC_GRAPH",
        "XSA_AC_FUSED",
        "XSA_AC_GPU_AC",
        "XSA_AC_DECODE_ROW",
        "XSA_AC_ENCODE_ROW",
        "XSA_AC_MEGA_ENCODE",
        "XSA_AC_ATTN_SPLITS",
        "XSA_AC_GQA_DEDUP",
        "XSA_AC_STAGE_FUSION",
    )
    old = {name: os.environ.get(name) for name in names}
    rng = np.random.default_rng(14)
    arr = rng.integers(0, 256, size=80, dtype=np.uint8).astype(np.int64)

    def rows_for(
        splits: int, dedup: int = 1, stage_fusion: int = 1
    ) -> np.ndarray:
        os.environ["XSA_AC_WINDOW"] = "64"
        os.environ["XSA_AC_GRAPH"] = "0"
        os.environ["XSA_AC_FUSED"] = "1"
        os.environ["XSA_AC_GPU_AC"] = "0"
        os.environ["XSA_AC_DECODE_ROW"] = "1"
        os.environ["XSA_AC_ENCODE_ROW"] = "0"
        os.environ["XSA_AC_MEGA_ENCODE"] = "1"
        os.environ["XSA_AC_ATTN_SPLITS"] = str(splits)
        os.environ["XSA_AC_GQA_DEDUP"] = str(dedup)
        os.environ["XSA_AC_STAGE_FUSION"] = str(stage_fusion)
        cfg = make_config(profile="smoke")
        cfg.block_size = 128
        cfg.online_retrain_every = 128
        cfg.use_bf16 = True
        cfg.gradient_checkpointing = False
        device = torch.device("cuda")
        torch.manual_seed(0)
        model = build_model(cfg, device=device)
        model.eval()
        engine = StaticACEngine(model, cfg, device)
        return _segment_rows_tf(engine, arr, arr, start=0)

    try:
        serial = rows_for(1)
        split_redundant = rows_for(8, dedup=0)
        split_unfused = rows_for(8, stage_fusion=0)
        split = rows_for(8)
        assert np.array_equal(split, split_redundant)
        for i, (fused, reference) in enumerate(
            zip(
                _freq_tables(split, 256),
                _freq_tables(split_unfused, 256),
            )
        ):
            assert np.array_equal(fused, reference), (
                f"stage-fused integer freq mismatch at pos {i}"
            )
        assert float(np.max(np.abs(split - split_unfused))) < 2e-6
        idx = np.arange(len(arr))
        serial_bpb = float(
            -np.log2(np.maximum(serial[idx, arr], 1e-30)).mean()
        )
        split_bpb = float(
            -np.log2(np.maximum(split[idx, arr], 1e-30)).mean()
        )
        assert abs(split_bpb - serial_bpb) < 5e-4
    finally:
        for name, value in old.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_static_kv_ensure_keeps_packed_view() -> None:
    """Prefill dtype/seqlen must not detach slot.k from k_pack."""
    from xsa_ttt.model import StaticKV

    max_len, n_kv, hd = 32, 2, 4
    pos = torch.zeros(1, dtype=torch.long)
    arange = torch.arange(max_len)
    arange_w = torch.arange(1)
    is_tail = arange >= (max_len - 1)
    tail_local = arange - (max_len - 1)
    sk = StaticKV(max_len, pos, arange, arange_w, is_tail, tail_local)
    k_pack = torch.randn(n_kv, max_len, hd)
    sk.k = k_pack.permute(1, 0, 2).unsqueeze(0)
    sk.v = torch.randn(n_kv, max_len, hd).permute(1, 0, 2).unsqueeze(0)
    ptr = sk.k.data_ptr()
    sk.ensure(torch.zeros(1, 8, n_kv, hd, dtype=sk.k.dtype))
    assert sk.k.data_ptr() == ptr
    assert sk.k.shape[1] == max_len
    sk.ensure(torch.zeros(1, 8, n_kv, hd, dtype=torch.float64))
    assert sk.k.data_ptr() == ptr
    assert sk.k.dtype == k_pack.dtype
