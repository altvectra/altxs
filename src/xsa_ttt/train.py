#!/usr/bin/env python3
"""Stream-lockstep pretrain + optional AC compress for the 587MB residual.

Mirrors NNCP-style stream cadence:
  walk the corpus in order → chunked TF CE (same forward as AC encode) →
  after each ``stream_pass`` optional AC; final full-corpus SHA when applicable.

On normal finish or early stop (Ctrl+C / SIGTERM): writes ``last.safetensors``,
``best.safetensors`` (if any), and ``loss_chart.png``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from xsa_ttt.chart import save_loss_chart  # noqa: E402
from xsa_ttt.checkpoint import (  # noqa: E402
    load_checkpoint_safetensors,
    save_checkpoint_safetensors,
)
from xsa_ttt.compress import (  # noqa: E402
    annotate_source_bpb,
    chunk_tf_train_loss,
    compress_bytes,
    compress_full_sha_lockstep,
    decompress_payload,
    measure_teacher_forced_bpb,
    online_retrain,
)
from xsa_ttt.config import count_parameters, make_config, should_online_retrain_at  # noqa: E402
from xsa_ttt.data import describe_data, load_symbols, resolve_data_path  # noqa: E402
from xsa_ttt.device import (  # noqa: E402
    empty_cache,
    enable_deterministic,
    resolve_device,
    synchronize,
)
from xsa_ttt.model import build_model  # noqa: E402
from xsa_ttt.ttt_lora import TTTLoRA, make_ttt_optimizer  # noqa: E402


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _ac_incremental() -> bool:
    """Shared KV-cache prob path for full-corpus AC (default ON).

    Encode and decode compute every prob row via prefill+extend, making the
    bitstream blind-decodable (--decode-payload) without the source product.
    Set XSA_AC_INCREMENTAL=0 for the legacy chunked-TF encode (fast, but its
    bitstream only verifies in-process via AC_VERIFY_DECODE — no blind decode).
    """
    return _env_flag("XSA_AC_INCREMENTAL", True)


def _fullsha_fn():
    if _ac_incremental():
        from xsa_ttt.incremental import compress_full_sha_incremental

        return compress_full_sha_incremental
    return compress_full_sha_lockstep


def _ac_verify_kwargs(
    *,
    verify_decode: bool | None = None,
    verify_end: bool | None = None,
) -> dict:
    """Full-SHA verify knobs: CLI overrides env; default encode-only."""
    return {
        "verify_decode": (
            bool(verify_decode)
            if verify_decode is not None
            else _env_flag("AC_VERIFY_DECODE", False)
        ),
        "verify_end": (
            bool(verify_end)
            if verify_end is not None
            else _env_flag("AC_VERIFY_END", False)
        ),
    }


def _ac_decoded_path(out_dir: Path, tag: str, verify_kw: dict) -> Path | None:
    """Where to stream the decompressed symbol file during AC verify."""
    if not (verify_kw.get("verify_decode") or verify_kw.get("verify_end")):
        return None
    override = os.environ.get("AC_DECODED_PATH", "").strip()
    if override:
        return Path(override)
    return out_dir / f"decoded_{tag if tag.endswith('fullsha') else tag + '_fullsha'}.bin"


def _fmt_roundtrip(ok: bool | None) -> str:
    if ok is None:
        return "n/a"
    return "OK" if ok else "FAIL"


def _append_metrics(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


# Cleared on each fresh train start so restarts don't append to stale curves.
_RUN_LOG_NAMES = (
    "metrics.jsonl",
    "loss_chart.png",
    "train_summary.json",
    "eval_tf.json",
    "ac_summary.json",
    "ac_curve.jsonl",
    "run.log",
)


def wipe_run_logs(out_dir: Path) -> list[str]:
    """Remove log/metrics artifacts from ``out_dir`` (keeps ``*.safetensors``)."""
    removed: list[str] = []
    for name in _RUN_LOG_NAMES:
        path = out_dir / name
        if path.is_file():
            path.unlink()
            removed.append(name)
    # Payload / leftover text logs from prior AC probes.
    for path in out_dir.glob("payload_*.bin"):
        path.unlink()
        removed.append(path.name)
    for path in out_dir.glob("ac_pass*.json"):
        path.unlink()
        removed.append(path.name)
    for path in out_dir.glob("*.log"):
        path.unlink()
        removed.append(path.name)
    return removed


def run_ac_measure(
    model,
    arr: np.ndarray,
    *,
    cfg,
    device: torch.device,
    out_dir: Path,
    ac_bytes: int,
    online_retrain: bool,
    progress: bool = True,
    tag: str = "ac",
    pass_idx: int | None = None,
    step: int | None = None,
    verify_decode: bool | None = None,
    verify_end: bool | None = None,
    n_symbols_total: int | None = None,
    source_bytes: int | None = None,
) -> dict:
    """AC on a prefix; returns summary (no payload / prob_cache).

    Short probes still encode+decode. Full corpus (``ac_bytes <= 0``) uses
    segmented SHA lockstep — encode-only by default (see ``AC_VERIFY_DECODE``).
    """
    n_data = int(arr.shape[0])
    ac_n = n_data if int(ac_bytes) <= 0 else min(n_data, int(ac_bytes))
    full_corpus = ac_n >= n_data
    verify_kw = _ac_verify_kwargs(
        verify_decode=verify_decode, verify_end=verify_end
    )
    full_mode = (
        "segmented_incremental_sha" if _ac_incremental() else "segmented_tf_sha"
    )
    print(
        f"[xsa_ttt] AC encode n={ac_n:,} tag={tag} "
        f"online_retrain={int(online_retrain)} "
        f"mode={full_mode if full_corpus else 'chunked_tf+memcache'} "
        f"(~{ac_n // max(1, int(cfg.online_retrain_every)):,} retrain boundaries)"
        + (
            f" verify_decode={int(verify_kw['verify_decode'])}"
            f" verify_end={int(verify_kw['verify_end'])}"
            if full_corpus
            else ""
        ),
        flush=True,
    )
    # Snapshot: online retrain during AC must not poison continued pretrain.
    out_dir.mkdir(parents=True, exist_ok=True)
    snap = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    was_training = model.training
    model.eval()
    result: dict | None = None
    try:
        if full_corpus:
            full_tag = tag if tag.endswith("fullsha") else f"{tag}_fullsha"
            decoded_path = _ac_decoded_path(out_dir, full_tag, verify_kw)
            result = _fullsha_fn()(
                model,
                arr,
                cfg=cfg,
                device=device,
                online_retrain_enabled=online_retrain,
                progress=progress,
                decoded_path=decoded_path,
                **verify_kw,
            )
            payload_path = out_dir / f"payload_{full_tag}.bin"
            payload_path.write_bytes(result["payload"])
            meta = {k: v for k, v in result.items() if k != "payload"}
            meta["ac_bytes"] = ac_n
            meta["tag"] = full_tag
            # Preserve null when encode-only (do not coerce None → False).
            if "roundtrip_ok" not in meta and "sha256_ok" in meta:
                meta["roundtrip_ok"] = meta["sha256_ok"]
            if pass_idx is not None:
                meta["pass"] = int(pass_idx)
            if step is not None:
                meta["step"] = int(step)
            meta["payload_path"] = str(payload_path)
            if result.get("decoded_path"):
                meta["decoded_path"] = result["decoded_path"]
                meta["decoded_bytes"] = result.get("decoded_bytes")
            return annotate_source_bpb(
                meta,
                n_symbols_total=n_symbols_total or ac_n,
                source_bytes=source_bytes,
            )

        slice_ = np.asarray(arr[:ac_n])
        result = compress_bytes(
            model,
            slice_,
            cfg=cfg,
            device=device,
            online_retrain_enabled=online_retrain,
            progress=progress,
        )
        # Optional: keep the drifted (post-online-retrain) weights for drift
        # analysis before the decode/finally paths restore the snapshot.
        if os.environ.get("AC_SAVE_DRIFTED", "0").lower() not in ("0", "", "false"):
            drift_path = out_dir / f"drifted_{tag}_{ac_n}.safetensors"
            save_checkpoint_safetensors(
                dict(model.state_dict()),
                drift_path,
                meta={
                    "probe": "ac_drifted",
                    "tag": tag,
                    "ac_bytes": ac_n,
                    "cfg": cfg.__dict__,
                },
            )
            print(f"[xsa_ttt] drifted weights → {drift_path}", flush=True)
        # Explicit False = encode-only probe (distill mid-train). None/True keep
        # legacy short-probe roundtrip check.
        do_decode = verify_decode is not False
        ok: bool | None = None
        if do_decode:
            # Decode from fresh weights + memcache (encode already mutated model).
            model.load_state_dict(snap, strict=True)
            model.to(device)
            model.eval()
            decoded = decompress_payload(
                model,
                result["payload"],
                ac_n,
                cfg=cfg,
                device=device,
                online_retrain_enabled=online_retrain,
                progress=progress,
                prob_cache=result.get("prob_cache"),
            )
            if int(cfg.vocab_size) <= 256:
                ok = decoded == bytes(np.asarray(slice_, dtype=np.uint8))
            else:
                ok = list(decoded) == [int(x) for x in slice_]
        payload_path = out_dir / f"payload_{tag}_{ac_n}.bin"
        payload_path.write_bytes(result["payload"])
        chunk_rows = result.pop("chunk_bpb_rows", None)
        if chunk_rows:
            curve_path = out_dir / f"ac_chunk_bpb_{tag}.jsonl"
            curve_path.write_text(
                "\n".join(json.dumps(r) for r in chunk_rows) + "\n",
                encoding="utf8",
            )
            print(f"[xsa_ttt] per-chunk rate curve → {curve_path}", flush=True)
        meta = {
            k: v for k, v in result.items() if k not in ("payload", "prob_cache")
        }
        meta["roundtrip_ok"] = ok
        meta["prob_cache_chunks"] = len(result.get("prob_cache") or {})
        meta["ac_bytes"] = ac_n
        meta["tag"] = tag
        if pass_idx is not None:
            meta["pass"] = int(pass_idx)
        if step is not None:
            meta["step"] = int(step)
        meta["payload_path"] = str(payload_path)
        return annotate_source_bpb(
            meta,
            n_symbols_total=n_symbols_total or int(arr.shape[0]),
            source_bytes=source_bytes,
        )
    finally:
        model.load_state_dict(snap, strict=True)
        model.to(device)
        if was_training:
            model.train()
        del result
        empty_cache(device)


def _fmt_ac_line(meta: dict) -> str:
    """Human line: bits/symbol + source-byte bpb + estimated full .bin."""
    parts = [f"bits/sym={float(meta['bpb']):.5f}"]
    if meta.get("source_bpb") is not None:
        parts.append(f"source_bpb={float(meta['source_bpb']):.5f}")
    parts.append(
        f"payload={int(meta['payload_bytes']):,} B / {int(meta['ac_bytes']):,} sym"
    )
    if meta.get("est_full_payload_bytes") is not None:
        est = int(meta["est_full_payload_bytes"])
        parts.append(f"est_full_bin={est:,} B ({est / 1e6:.1f} MB)")
    parts.append(f"roundtrip={_fmt_roundtrip(meta.get('roundtrip_ok'))}")
    return " ".join(parts)


def _detect_pass_offset(out_dir: Path) -> int:
    """Highest completed stream pass under ``out_dir`` (ac_curve / passN.safetensors)."""
    best = 0
    curve = out_dir / "ac_curve.jsonl"
    if curve.is_file():
        for line in curve.read_text(encoding="utf8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            p = row.get("pass")
            if isinstance(p, int):
                best = max(best, p)
    for path in out_dir.glob("pass*.safetensors"):
        stem = path.stem  # pass9 / pass9_fullsha — only plain passN
        if stem.startswith("pass") and stem[4:].isdigit():
            best = max(best, int(stem[4:]))
    return best


def stream_pretrain(
    model,
    arr: np.ndarray,
    *,
    cfg,
    device: torch.device,
    out_dir: Path,
    online_retrain_during_train: bool = True,
    ac_every_pass: bool = False,
    ac_bytes: int = 65_536,
    ac_online_retrain: bool = True,
    pass_offset: int = 0,
    start_step: int = 0,
    verify_decode: bool | None = None,
    verify_end: bool | None = None,
    n_symbols_total: int | None = None,
    source_bytes: int | None = None,
) -> dict:
    """One or more full passes over ``arr`` with stride = online_retrain_every.

    ``pass_offset`` continues numbering after a prior run (e.g. 9 → next AC is
    pass10). ``start_step`` continues the global step counter in metrics.
    """
    every = max(1, int(cfg.online_retrain_every))
    block = int(cfg.block_size)
    n = int(arr.shape[0])
    passes = max(1, int(cfg.stream_passes))
    pass_offset = max(0, int(pass_offset))
    start_step = max(0, int(start_step))
    steps_per_pass = max(1, n // every)
    max_iters = int(cfg.max_iters)
    if max_iters <= 0:
        max_iters = start_step + passes * steps_per_pass
    total_pass_target = pass_offset + passes

    # Base pretrain optimizer (full model).
    for p in model.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.pretrain_lr),
        weight_decay=float(cfg.pretrain_weight_decay),
    )

    lora = None
    lora_opt = None
    if online_retrain_during_train and (cfg.retrain_mode or "full").lower() == "lora":
        # Separate LoRA path practiced during pretrain so compress starts warm.
        torch.manual_seed(int(cfg.seed))
        lora = TTTLoRA(model, cfg).to(device)
        lora_opt = make_ttt_optimizer(lora, cfg)

    metrics_path = out_dir / "metrics.jsonl"
    chart_path = out_dir / "loss_chart.png"
    model.train()
    model.enable_looping(False)  # enable mid-run after the pretrain fraction
    step = start_step
    best_bpb = float("inf")
    best_step = start_step
    best_state: dict[str, torch.Tensor] | None = None
    last_bpb = float("nan")
    t0 = time.time()
    stream_pos = 0
    pass_idx = 0  # local completed passes in this call (0..passes)
    early_stop = False
    stop_requested = False
    ac_curve_path = out_dir / "ac_curve.jsonl"
    ac_pass_rows: list[dict] = []
    if pass_offset or start_step:
        print(
            f"[xsa_ttt] continue train: pass_offset={pass_offset} "
            f"start_step={start_step} +{passes} passes → through pass{total_pass_target}",
            flush=True,
        )

    def _request_stop(signum, _frame) -> None:  # noqa: ANN001
        nonlocal stop_requested
        stop_requested = True
        print(
            f"[train] signal {signum} at step={step}; will save safetensors + chart",
            flush=True,
        )

    prev_sigint = signal.getsignal(signal.SIGINT)
    prev_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    def _meta_for(step_i: int, train_bpb: float, *, tag: str) -> dict:
        return {
            "probe": "xsa_ttt",
            "tag": tag,
            "step": int(step_i),
            "train_bpb": float(train_bpb) if train_bpb == train_bpb else None,
            "best_train_bpb": float(best_bpb) if best_bpb < float("inf") else None,
            "best_step": int(best_step),
            "params": count_parameters(model),
            "block_size": block,
            "device": str(device),
            "data": str(getattr(cfg, "data_path", "")),
            "early_stop": early_stop,
            "cfg": cfg.__dict__,
        }

    def _write_weights() -> None:
        save_checkpoint_safetensors(
            dict(model.state_dict()),
            out_dir / "last.safetensors",
            meta=_meta_for(step, last_bpb, tag="last"),
        )
        if best_state is not None:
            save_checkpoint_safetensors(
                best_state,
                out_dir / "best.safetensors",
                meta=_meta_for(best_step, best_bpb, tag="best"),
            )

    def _finalize(reason: str) -> dict:
        model.enable_looping(True)
        try:
            _write_weights()
        except Exception as exc:  # noqa: BLE001
            print(f"[ckpt] finalize save failed: {exc}", flush=True)
        chart = save_loss_chart(
            metrics_path,
            chart_path,
            title=f"xsa_ttt ({reason})",
        )
        synchronize(device)
        empty_cache(device)
        return {
            "ok": not early_stop,
            "early_stop": early_stop,
            "reason": reason,
            "steps": step,
            "best_bpb": best_bpb if best_bpb < float("inf") else None,
            "best_step": best_step,
            "last_bpb": last_bpb if last_bpb == last_bpb else None,
            "elapsed_s": time.time() - t0,
            "checkpoint_last": str(out_dir / "last.safetensors"),
            "checkpoint_best": (
                str(out_dir / "best.safetensors") if best_state is not None else None
            ),
            "chart": str(chart) if chart else None,
            "ac_curve": str(ac_curve_path) if ac_pass_rows else None,
            "ac_passes": ac_pass_rows,
        }

    def _ac_after_pass(finished_pass: int) -> None:
        """Save pass ckpt, run AC, restore weights for the next pass."""
        tag = f"pass{finished_pass}"
        save_checkpoint_safetensors(
            dict(model.state_dict()),
            out_dir / f"{tag}.safetensors",
            meta=_meta_for(step, last_bpb, tag=tag),
        )
        _write_weights()
        print(
            f"[xsa_ttt] pass {finished_pass}/{total_pass_target} done (step={step}); "
            f"running AC…",
            flush=True,
        )
        pbar.set_description(f"xsa_ttt AC pass{finished_pass}")
        meta = run_ac_measure(
            model,
            arr,
            cfg=cfg,
            device=device,
            out_dir=out_dir,
            ac_bytes=ac_bytes,
            online_retrain=ac_online_retrain,
            progress=True,
            tag=tag,
            pass_idx=finished_pass,
            step=step,
            verify_decode=verify_decode,
            verify_end=verify_end,
            n_symbols_total=n_symbols_total,
            source_bytes=source_bytes,
        )
        meta["elapsed_s"] = time.time() - t0
        _append_metrics(ac_curve_path, meta)
        (out_dir / f"ac_{tag}.json").write_text(json.dumps(meta, indent=2) + "\n")
        row = {
            "pass": finished_pass,
            "step": step,
            "bpb": meta.get("bpb"),
            "source_bpb": meta.get("source_bpb"),
            "est_full_payload_bytes": meta.get("est_full_payload_bytes"),
            "roundtrip_ok": meta.get("roundtrip_ok"),
            "payload_bytes": meta.get("payload_bytes"),
            "ac_bytes": meta.get("ac_bytes"),
        }
        print(
            f"[xsa_ttt] pass {finished_pass} AC {_fmt_ac_line(meta)}",
            flush=True,
        )

        # Full-corpus SHA only on the final pass of this train segment.
        if finished_pass == total_pass_target:
            full_tag = f"{tag}_fullsha"
            verify_kw = _ac_verify_kwargs(
                verify_decode=verify_decode, verify_end=verify_end
            )
            print(
                f"[xsa_ttt] pass {finished_pass}: full-corpus SHA AC (final; "
                f"verify_decode={int(verify_kw['verify_decode'])} "
                f"verify_end={int(verify_kw['verify_end'])})…",
                flush=True,
            )
            pbar.set_description(f"xsa_ttt full-SHA pass{finished_pass}")
            snap = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            was_training = model.training
            model.eval()
            decoded_path = _ac_decoded_path(out_dir, full_tag, verify_kw)
            try:
                full = _fullsha_fn()(
                    model,
                    arr,
                    cfg=cfg,
                    device=device,
                    online_retrain_enabled=ac_online_retrain,
                    progress=True,
                    decoded_path=decoded_path,
                    **verify_kw,
                )
            finally:
                model.load_state_dict(snap, strict=True)
                model.to(device)
                if was_training:
                    model.train()
                empty_cache(device)
            full["pass"] = finished_pass
            full["step"] = step
            full["tag"] = full_tag
            full["elapsed_s"] = time.time() - t0
            # Drop raw payload from jsonl (can be hundreds of MB).
            full_path = out_dir / f"payload_{full_tag}.bin"
            full_path.write_bytes(full["payload"])
            full_meta = {k: v for k, v in full.items() if k != "payload"}
            full_meta["payload_path"] = str(full_path)
            if full.get("decoded_path"):
                full_meta["decoded_path"] = full["decoded_path"]
                full_meta["decoded_bytes"] = full.get("decoded_bytes")
            full_meta["ac_bytes"] = int(full_meta.get("n_bytes", arr.shape[0]))
            annotate_source_bpb(
                full_meta,
                n_symbols_total=n_symbols_total or int(arr.shape[0]),
                source_bytes=source_bytes,
            )
            _append_metrics(ac_curve_path, full_meta)
            (out_dir / f"ac_{full_tag}.json").write_text(
                json.dumps(full_meta, indent=2) + "\n"
            )
            row["full_sha_ok"] = full_meta.get("sha256_ok")
            row["full_bpb"] = full_meta.get("bpb")
            row["full_source_bpb"] = full_meta.get("source_bpb")
            row["full_payload_bytes"] = full_meta.get("payload_bytes")
            print(
                f"[xsa_ttt] pass {finished_pass} FULL SHA "
                f"{_fmt_roundtrip(full_meta.get('sha256_ok'))} "
                f"{_fmt_ac_line(full_meta)} "
                f"src_sha={str(full_meta.get('sha256', ''))[:16]}…"
                + (
                    f" dec_sha={str(full_meta.get('decoded_sha256', ''))[:16]}…"
                    if full_meta.get("decoded_sha256")
                    else ""
                ),
                flush=True,
            )

        ac_pass_rows.append(row)
        pbar.set_description("xsa_ttt train")
        model.train()

    from tqdm import tqdm

    pbar = tqdm(
        total=max_iters,
        desc="xsa_ttt train",
        unit="step",
        file=sys.stderr,
        dynamic_ncols=True,
        mininterval=0.5,
        disable=False,  # keep bar under ``tee`` (non-TTY)
    )
    try:
        while step < max_iters and pass_idx < passes:
            if stop_requested:
                early_stop = True
                break

            # Enable depth recurrence after enable_looping_at fraction.
            frac = step / max(1, max_iters)
            if frac >= float(cfg.enable_looping_at):
                model.enable_looping(True)

            chunk_start = stream_pos
            chunk_end = min(n, stream_pos + every)
            if chunk_end < 2:
                break

            opt.zero_grad(set_to_none=True)
            # Same chunked TF as AC encode: one forward over [ctx_start, chunk_end).
            loss = chunk_tf_train_loss(
                model,
                arr,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
                block_size=block,
                device=device,
                use_bf16=bool(getattr(cfg, "use_bf16", False)),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            ce = float(loss.detach().float().item())
            bpb = ce / math.log(2)
            last_bpb = bpb

            # Practice NNCP LoRA retrain at the same boundary.
            if (
                online_retrain_during_train
                and lora is not None
                and lora_opt is not None
                and should_online_retrain_at(chunk_end, every=every)
            ):
                for p in model.parameters():
                    p.requires_grad_(False)
                online_retrain(
                    model,
                    arr,
                    end=chunk_end,
                    cfg=cfg,
                    device=device,
                    optimizer=lora_opt,
                    lora=lora,
                    seed=cfg.seed,
                )
                for p in model.parameters():
                    p.requires_grad_(True)
                model.train()

            step += 1
            stream_pos = chunk_end
            wrapped = False
            if stream_pos >= n:
                stream_pos = 0
                pass_idx += 1
                wrapped = True

            if bpb < best_bpb:
                best_bpb = bpb
                best_step = step
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }

            pbar.update(1)
            abs_pass = pass_offset + pass_idx
            pbar.set_postfix(
                {
                    "bpb": f"{bpb:.3f}",
                    "best": f"{best_bpb:.3f}",
                    "ce": f"{ce:.3f}",
                    "pass": abs_pass,
                    "loop": int(model.looping_active),
                },
                refresh=False,
            )

            if step == start_step + 1 or step % 20 == 0 or wrapped or step >= max_iters:
                row = {
                    "step": step,
                    "pass": abs_pass,
                    "pos": n if wrapped else chunk_end,
                    "ce_nats": ce,
                    "bpb": bpb,
                    "looping": int(model.looping_active),
                    "elapsed_s": time.time() - t0,
                }
                _append_metrics(metrics_path, row)

            if wrapped and ac_every_pass:
                _ac_after_pass(abs_pass)
                if stop_requested:
                    early_stop = True
                    break
    except KeyboardInterrupt:
        early_stop = True
        print(
            f"[train] KeyboardInterrupt at step={step}; saving safetensors + chart",
            flush=True,
        )
        return _finalize("early_stop_interrupt")
    finally:
        pbar.close()
        signal.signal(signal.SIGINT, prev_sigint)
        signal.signal(signal.SIGTERM, prev_sigterm)

    if early_stop:
        return _finalize("early_stop_signal")
    return _finalize("completed")


_DECODE_RUNTIME_FIELDS = (
    "retrain_mode",
    "online_retrain_steps",
    "online_retrain_every",
    "online_retrain_replay",
    "ttt_lora_lr",
    "ttt_beta1",
    "ttt_beta2",
    "ttt_weight_decay",
    "ttt_replenish_ramp",
    "ttt_replenish_anneal",
    "ttt_replenish_lr_half_mb",
    "ttt_replenish_lr_min",
    "ttt_replenish_steps_min",
    "ttt_replenish_total_bytes",
    "ttt_replenish_warmup_steps",
    "ttt_replenish_accum",
    "ttt_replenish_replay_lr_scale",
    "ttt_replenish_replay_recent_mb",
    "ttt_replenish_anchor",
    "ttt_replenish_anchor_rate",
    "ttt_replenish_xm_k",
    "ttt_replenish_xm_probe_chunks",
    "ttt_replenish_xm_4x_until_bytes",
    "use_bf16",
    "gradient_checkpointing",
    "block_size",
    "seed",
)


def _overlay_env_runtime(loaded_cfg, *, profile: str) -> None:
    """Copy lockstep / replenish knobs from env onto a loaded checkpoint cfg."""
    env_cfg = make_config(profile=profile)
    for name in _DECODE_RUNTIME_FIELDS:
        setattr(loaded_cfg, name, getattr(env_cfg, name))
    if (
        (loaded_cfg.retrain_mode or "full").lower() in {"full", "replenish"}
        and int(loaded_cfg.online_retrain_steps) > 0
        and int(loaded_cfg.block_size) >= 8192
        and "XSA_GRAD_CKPT" not in os.environ
    ):
        loaded_cfg.gradient_checkpointing = True


def run_decode_payload(
    payload_path: Path,
    *,
    eval_ckpt: Path,
    profile: str,
    device: torch.device,
    out_dir: Path,
    n_bytes: int,
    decoded_path: Path | None = None,
) -> dict:
    """Blind AC-decode of a segment-framed incremental bitstream (S-side restore).

    Requires the bitstream to have been encoded with ``XSA_AC_INCREMENTAL=1``
    (shared KV-cache prob path). Replays the identical retrain trajectory on
    the decoded prefix — no source product needed.
    """
    from xsa_ttt.incremental import decompress_full_sha_incremental

    payload = Path(payload_path).read_bytes()
    model, cfg = load_checkpoint_safetensors(eval_ckpt, device=device, cfg=None)
    _overlay_env_runtime(cfg, profile=profile)
    model.gradient_checkpointing = bool(cfg.gradient_checkpointing)
    n_bytes = int(n_bytes)
    if n_bytes <= 0:
        raise ValueError("decode needs n_bytes > 0 (--ac-bytes / AC_N_SYMBOLS)")
    print(
        f"[xsa_ttt] AC blind decode payload={len(payload):,} B n={n_bytes:,} "
        f"ckpt={eval_ckpt} retrain_mode={cfg.retrain_mode} "
        f"lr={cfg.ttt_lora_lr:g} half={cfg.ttt_replenish_lr_half_mb:g}MB "
        f"lr_min={cfg.ttt_replenish_lr_min:g} out={out_dir}",
        flush=True,
    )
    online = int(cfg.online_retrain_steps) > 0
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = (
        Path(decoded_path)
        if decoded_path is not None
        else out_dir / "decoded_stream.bin"
    )
    result = decompress_full_sha_incremental(
        model,
        payload,
        n_bytes,
        cfg=cfg,
        device=device,
        online_retrain_enabled=online,
        progress=True,
        decoded_path=dest,
    )
    meta = {
        "payload_bytes": len(payload),
        "decoded_bytes": n_bytes,
        "n_symbols": n_bytes,
        "segments": result.get("segments"),
        "decoded_sha256": result.get("decoded_sha256"),
        "decoded_path": str(dest),
        "ckpt": str(eval_ckpt),
        "infer_mode": result.get("infer_mode"),
    }
    (out_dir / "decode_summary.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf8"
    )
    print(
        f"[xsa_ttt] AC decode wrote {dest} ({n_bytes:,} B) "
        f"sha256={str(meta['decoded_sha256'])[:16]}…",
        flush=True,
    )
    return meta


def _resolve_ckpt_path(path: Path, *, which: str = "last") -> Path:
    """Accept a ``.safetensors`` file or a run dir (``last``/``best``/``passN``)."""
    path = Path(path)
    if path.is_file():
        return path if path.suffix == ".safetensors" else path.with_suffix(".safetensors")
    if path.is_dir():
        which = (which or "last").strip()
        if which.endswith(".safetensors"):
            which = Path(which).stem
        cand = path / f"{which}.safetensors"
        if cand.is_file():
            return cand
        for alt_name in ("last", "best"):
            alt = path / f"{alt_name}.safetensors"
            if alt.is_file():
                return alt
        raise FileNotFoundError(
            f"no {which}.safetensors (or last/best) under run dir: {path}"
        )
    raise FileNotFoundError(f"checkpoint not found: {path}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--profile",
        default="large",
        choices=("large", "compact", "lite", "mid", "smoke"),
        help="large~32M; compact~10M; lite~3.4M; mid~1.3M; smoke=tiny",
    )
    p.add_argument("--data", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--block-size", type=int, default=None)
    p.add_argument("--max-bytes", type=int, default=None, help="truncate corpus")
    p.add_argument("--max-iters", type=int, default=None)
    p.add_argument(
        "--stream-passes",
        type=int,
        default=None,
        help="full corpus passes (overrides XSA_STREAM_PASSES / profile default)",
    )
    p.add_argument("--device", default=None)
    p.add_argument(
        "--ac",
        action="store_true",
        help="AC payload measure (after pretrain, or with --eval-only resume)",
    )
    p.add_argument(
        "--ac-every-pass",
        action="store_true",
        help="with pretrain: run AC after each stream pass (writes ac_curve.jsonl)",
    )
    p.add_argument(
        "--ac-bytes",
        type=int,
        default=65_536,
        help="AC prefix length in bytes; 0 = full corpus",
    )
    p.add_argument(
        "--tf-bytes",
        type=int,
        default=None,
        help="teacher-forced probe length (default: min(n, max(ac_bytes*4, 256KiB)))",
    )
    p.add_argument(
        "--skip-tf",
        action="store_true",
        help="skip teacher-forced probe (or set SKIP_TF=1); still runs AC if --ac",
    )
    p.add_argument("--no-online-retrain", action="store_true")
    p.add_argument(
        "--ac-verify-decode",
        action="store_true",
        help="full-SHA: per-segment decode+SHA (or set AC_VERIFY_DECODE=1)",
    )
    p.add_argument(
        "--ac-verify-end",
        action="store_true",
        help="full-SHA: encode-only then one decode pass (or AC_VERIFY_END=1)",
    )
    p.add_argument(
        "--eval-only",
        type=Path,
        default=None,
        help="skip pretrain; load .safetensors (or run dir) then TF (+ optional AC)",
    )
    p.add_argument(
        "--decode-payload",
        type=Path,
        default=None,
        help=(
            "blind AC-decode of a segment-framed *_fullsha.bin payload "
            "(no source). Requires --eval-only and --ac-bytes = symbol "
            "count, and a bitstream encoded with XSA_AC_INCREMENTAL=1 "
            "(shared KV-cache prob path). Legacy chunked-TF bitstreams "
            "will not decode — re-encode incrementally or use "
            "scripts/roundtrip_s_decode.sh (AC_VERIFY_DECODE)"
        ),
    )
    p.add_argument(
        "--decoded-path",
        type=Path,
        default=None,
        help="with --decode-payload: where to write the restored stream",
    )
    p.add_argument(
        "--ac-from-init",
        action="store_true",
        help=(
            "NNCP-style: skip pretrain/ckpt; seed+build_model then TF/AC with "
            "online retrain only (or set AC_FROM_INIT=1)"
        ),
    )
    p.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help="load .safetensors (or run dir) then continue pretrain (does not wipe logs)",
    )
    p.add_argument(
        "--pass-offset",
        type=int,
        default=None,
        help="with --init-from: start pass numbering after N (default: detect from out dir)",
    )
    p.add_argument(
        "--ckpt",
        default="last",
        help="when --eval-only/--init-from is a run dir: last|best|passN (default last)",
    )
    args = p.parse_args(argv)

    cfg = make_config(
        profile=args.profile,
        data_path=str(args.data) if args.data else None,
        seed=args.seed,
        block_size=args.block_size,
    )
    if args.max_iters is not None:
        cfg.max_iters = int(args.max_iters)
    if args.stream_passes is not None:
        cfg.stream_passes = max(1, int(args.stream_passes))
    if args.no_online_retrain:
        cfg.online_retrain_steps = 0
    if args.ac_every_pass:
        args.ac = True
    # CLI flags override env; unset CLI keeps env / encode-only default.
    ac_verify_decode = True if args.ac_verify_decode else None
    ac_verify_end = True if args.ac_verify_end else None
    ac_from_init = bool(args.ac_from_init) or _env_flag("AC_FROM_INIT", False)
    if ac_from_init:
        args.ac = True

    # Before any model build / CUDA work: seed + deterministic kernels.
    det = enable_deterministic(int(cfg.seed))
    print(
        f"[xsa_ttt] deterministic mode={det.get('mode')} seed={det.get('seed')} "
        f"algs={det.get('deterministic_algorithms')} "
        f"cudnn_det={det.get('cudnn_deterministic')} tf32={det.get('tf32')} "
        f"(COMPRESSION_DETERMINISTIC={os.environ.get('COMPRESSION_DETERMINISTIC', 'warn')})",
        flush=True,
    )

    device = resolve_device(args.device)
    if args.decode_payload is not None:
        if args.eval_only is None:
            raise SystemExit("error: --decode-payload requires --eval-only")
        n_bytes = int(args.ac_bytes)
        if n_bytes <= 0:
            n_bytes = int(os.environ.get("AC_N_SYMBOLS") or os.environ.get("AC_BYTES") or 0)
        eval_ckpt = _resolve_ckpt_path(args.eval_only, which=args.ckpt)
        out = Path(args.out or eval_ckpt.parent / "ac_decode")
        run_decode_payload(
            args.decode_payload,
            eval_ckpt=eval_ckpt,
            profile=args.profile,
            device=device,
            out_dir=out,
            n_bytes=n_bytes,
            decoded_path=args.decoded_path,
        )
        return 0
    data_path = resolve_data_path(args.data or cfg.data_path)
    arr, corpus = load_symbols(data_path, max_symbols=args.max_bytes)
    cfg.vocab_size = int(corpus.vocab_size)
    cfg.data_path = str(data_path)
    if corpus.kind == "bpe_tokens":
        n_symbols_total = int(
            corpus.meta.get("n_tokens")
            or (corpus.path.stat().st_size // 2)
        )
        source_bytes = (
            int(corpus.source_bytes) if corpus.source_bytes is not None else None
        )
    else:
        n_symbols_total = int(corpus.path.stat().st_size)
        source_bytes = n_symbols_total

    def _apply_runtime_overrides(loaded_cfg) -> None:
        if args.block_size is not None:
            loaded_cfg.block_size = int(args.block_size)
            loaded_cfg.online_retrain_every = int(args.block_size)
        if "XSA_RETRAIN_EVERY" in os.environ:
            loaded_cfg.online_retrain_every = int(os.environ["XSA_RETRAIN_EVERY"])
        if "XSA_RETRAIN_MODE" in os.environ:
            loaded_cfg.retrain_mode = os.environ["XSA_RETRAIN_MODE"].strip().lower()
        if "XSA_RETRAIN_STEPS" in os.environ:
            loaded_cfg.online_retrain_steps = max(
                0, int(os.environ["XSA_RETRAIN_STEPS"])
            )
        if "XSA_RETRAIN_REPLAY" in os.environ:
            loaded_cfg.online_retrain_replay = os.environ[
                "XSA_RETRAIN_REPLAY"
            ].strip().lower() not in {"0", "false", "no", "off"}
        if "XSA_TTT_BOOTSTRAP_SYMBOLS" in os.environ:
            loaded_cfg.ttt_bootstrap_symbols = int(
                os.environ["XSA_TTT_BOOTSTRAP_SYMBOLS"]
            )
        if "XSA_TTT_BOOTSTRAP_STEPS" in os.environ:
            loaded_cfg.ttt_bootstrap_steps = int(
                os.environ["XSA_TTT_BOOTSTRAP_STEPS"]
            )
        if "XSA_REPLENISH_RAMP" in os.environ:
            loaded_cfg.ttt_replenish_ramp = int(os.environ["XSA_REPLENISH_RAMP"])
        if "XSA_REPLENISH_ANNEAL" in os.environ:
            loaded_cfg.ttt_replenish_anneal = (
                os.environ["XSA_REPLENISH_ANNEAL"].strip().lower()
            )
        if "XSA_REPLENISH_LR_HALF_MB" in os.environ:
            loaded_cfg.ttt_replenish_lr_half_mb = float(
                os.environ["XSA_REPLENISH_LR_HALF_MB"]
            )
        if "XSA_REPLENISH_LR_MIN" in os.environ:
            loaded_cfg.ttt_replenish_lr_min = float(
                os.environ["XSA_REPLENISH_LR_MIN"]
            )
        if "XSA_REPLENISH_STEPS_MIN" in os.environ:
            loaded_cfg.ttt_replenish_steps_min = int(
                os.environ["XSA_REPLENISH_STEPS_MIN"]
            )
        if "XSA_REPLENISH_TOTAL_BYTES" in os.environ:
            loaded_cfg.ttt_replenish_total_bytes = int(
                os.environ["XSA_REPLENISH_TOTAL_BYTES"]
            )
        if "XSA_REPLENISH_WARMUP_STEPS" in os.environ:
            loaded_cfg.ttt_replenish_warmup_steps = int(
                os.environ["XSA_REPLENISH_WARMUP_STEPS"]
            )
        if "XSA_REPLENISH_ACCUM" in os.environ:
            loaded_cfg.ttt_replenish_accum = os.environ[
                "XSA_REPLENISH_ACCUM"
            ].strip().lower() not in {"0", "false", "no", "off"}
        if "XSA_REPLENISH_REPLAY_LR_SCALE" in os.environ:
            loaded_cfg.ttt_replenish_replay_lr_scale = float(
                os.environ["XSA_REPLENISH_REPLAY_LR_SCALE"]
            )
        if "XSA_REPLENISH_REPLAY_RECENT_MB" in os.environ:
            loaded_cfg.ttt_replenish_replay_recent_mb = float(
                os.environ["XSA_REPLENISH_REPLAY_RECENT_MB"]
            )
        if "XSA_REPLENISH_ANCHOR" in os.environ:
            loaded_cfg.ttt_replenish_anchor = os.environ[
                "XSA_REPLENISH_ANCHOR"
            ].strip()
        if "XSA_REPLENISH_ANCHOR_RATE" in os.environ:
            loaded_cfg.ttt_replenish_anchor_rate = float(
                os.environ["XSA_REPLENISH_ANCHOR_RATE"]
            )
        if "XSA_REPLENISH_XM_K" in os.environ:
            loaded_cfg.ttt_replenish_xm_k = int(
                os.environ["XSA_REPLENISH_XM_K"]
            )
        if "XSA_REPLENISH_XM_PROBE_CHUNKS" in os.environ:
            loaded_cfg.ttt_replenish_xm_probe_chunks = int(
                os.environ["XSA_REPLENISH_XM_PROBE_CHUNKS"]
            )
        if "XSA_REPLENISH_XM_4X_UNTIL_MB" in os.environ:
            loaded_cfg.ttt_replenish_xm_4x_until_bytes = int(
                float(os.environ["XSA_REPLENISH_XM_4X_UNTIL_MB"]) * 1024 * 1024
            )
        if "XSA_TTT_BURSTS" in os.environ:
            loaded_cfg.ttt_burst_schedule = os.environ["XSA_TTT_BURSTS"].strip()
        if "XSA_BURST_LR" in os.environ:
            loaded_cfg.ttt_burst_lr = float(os.environ["XSA_BURST_LR"])
        if "XSA_BURST_BETA1" in os.environ:
            loaded_cfg.ttt_burst_beta1 = float(os.environ["XSA_BURST_BETA1"])
        if "XSA_BURST_BETA2" in os.environ:
            loaded_cfg.ttt_burst_beta2 = float(os.environ["XSA_BURST_BETA2"])
        if "XSA_BURST_WD" in os.environ:
            loaded_cfg.ttt_burst_weight_decay = float(os.environ["XSA_BURST_WD"])
        if "XSA_RETRAIN_LR" in os.environ:
            loaded_cfg.ttt_lora_lr = float(os.environ["XSA_RETRAIN_LR"])
        elif "XSA_TTT_LR" in os.environ:
            loaded_cfg.ttt_lora_lr = float(os.environ["XSA_TTT_LR"])
        elif "XSA_TTT_LORA_LR" in os.environ:
            loaded_cfg.ttt_lora_lr = float(os.environ["XSA_TTT_LORA_LR"])
        if "XSA_TTT_WEIGHT_DECAY" in os.environ:
            loaded_cfg.ttt_weight_decay = float(
                os.environ["XSA_TTT_WEIGHT_DECAY"]
            )
        if "XSA_TTT_BETA1" in os.environ:
            loaded_cfg.ttt_beta1 = float(os.environ["XSA_TTT_BETA1"])
        if "XSA_TTT_BETA2" in os.environ:
            loaded_cfg.ttt_beta2 = float(os.environ["XSA_TTT_BETA2"])
        if "XSA_PRETRAIN_LR" in os.environ:
            loaded_cfg.pretrain_lr = float(os.environ["XSA_PRETRAIN_LR"])
        if "XSA_PRETRAIN_WD" in os.environ:
            loaded_cfg.pretrain_weight_decay = float(
                os.environ["XSA_PRETRAIN_WD"]
            )
        if "XSA_ENABLE_LOOPING_AT" in os.environ:
            loaded_cfg.enable_looping_at = float(
                os.environ["XSA_ENABLE_LOOPING_AT"]
            )
        if "XSA_GRAD_CLIP" in os.environ:
            loaded_cfg.grad_clip = float(os.environ["XSA_GRAD_CLIP"])
        if "XSA_BF16" in os.environ:
            loaded_cfg.use_bf16 = str(os.environ["XSA_BF16"]).strip().lower() not in {
                "0",
                "false",
                "no",
                "off",
            }
        if "XSA_GRAD_CKPT" in os.environ:
            loaded_cfg.gradient_checkpointing = str(
                os.environ["XSA_GRAD_CKPT"]
            ).strip().lower() not in {"0", "false", "no", "off"}
        # 16k full-weight retrain needs activation checkpointing on 80GB.
        elif (
            (loaded_cfg.retrain_mode or "full").lower() == "full"
            and int(loaded_cfg.online_retrain_steps) > 0
            and int(loaded_cfg.block_size) >= 8192
        ):
            loaded_cfg.gradient_checkpointing = True
        if args.stream_passes is not None:
            loaded_cfg.stream_passes = max(1, int(args.stream_passes))

    _apply_runtime_overrides(cfg)

    eval_ckpt: Path | None = None
    init_ckpt: Path | None = None
    if (
        sum(
            [
                args.eval_only is not None,
                args.init_from is not None,
                ac_from_init,
            ]
        )
        > 1
    ):
        raise SystemExit(
            "error: use only one of --eval-only / --init-from / --ac-from-init"
        )
    if args.eval_only is not None:
        eval_ckpt = _resolve_ckpt_path(args.eval_only, which=args.ckpt)
        out = args.out or eval_ckpt.parent
    elif args.init_from is not None:
        init_ckpt = _resolve_ckpt_path(args.init_from, which=args.ckpt)
        out = args.out or init_ckpt.parent
    else:
        out = args.out or (
            _HERE
            / "runs"
            / f"{args.profile}_{device.type}_{time.strftime('%Y%m%d_%H%M%S')}"
        )
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    print(
        f"[xsa_ttt] device={device} profile={args.profile} "
        f"{describe_data(data_path)} n={arr.shape[0]:,} "
        f"vocab={cfg.vocab_size} kind={corpus.kind} "
        f"block={cfg.block_size} every={cfg.online_retrain_every} "
        f"retrain_mode={cfg.retrain_mode} bf16={int(cfg.use_bf16)} "
        f"yarn={int(cfg.rope_yarn)} out={out}",
        flush=True,
    )

    if ac_from_init:
        lora_hint = cfg.lora_path or os.environ.get("XSA_LORA_PATH")
        print(
            f"[xsa_ttt] AC-from-init (NNCP-style): seed={cfg.seed} "
            f"no pretrain/ckpt; "
            + (
                f"distilled LoRA={lora_hint}"
                if lora_hint
                else "online retrain during AC"
            ),
            flush=True,
        )
        # Re-seed immediately before init so later CUDA setup cannot drift RNG.
        enable_deterministic(int(cfg.seed))
        model = build_model(cfg, device=device)
        nparams = count_parameters(model)
        print(
            f"[xsa_ttt] init params={nparams:,} "
            f"retrain_mode={cfg.retrain_mode} block={cfg.block_size}",
            flush=True,
        )
        if lora_hint:
            cfg.lora_path = str(lora_hint)
            if "XSA_RETRAIN_MODE" not in os.environ:
                cfg.retrain_mode = "lora"
    elif eval_ckpt is not None:
        print(f"[xsa_ttt] resume ckpt={eval_ckpt}", flush=True)
        # Prefer sibling .json cfg (exact train dims) over CLI profile defaults.
        model, cfg = load_checkpoint_safetensors(eval_ckpt, device=device, cfg=None)
        _apply_runtime_overrides(cfg)
        model.gradient_checkpointing = bool(cfg.gradient_checkpointing)
        nparams = count_parameters(model)
        print(
            f"[xsa_ttt] loaded params={nparams:,} "
            f"retrain_mode={cfg.retrain_mode} block={cfg.block_size} "
            f"grad_ckpt={int(model.gradient_checkpointing)}",
            flush=True,
        )
    else:
        pass_offset = 0
        start_step = 0
        if init_ckpt is not None:
            print(f"[xsa_ttt] init-from ckpt={init_ckpt} (continue pretrain)", flush=True)
            model, cfg = load_checkpoint_safetensors(init_ckpt, device=device, cfg=None)
            _apply_runtime_overrides(cfg)
            model.gradient_checkpointing = bool(cfg.gradient_checkpointing)
            meta_path = init_ckpt.with_suffix(".json")
            if meta_path.is_file():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf8"))
                    start_step = int(meta.get("step") or 0)
                except (json.JSONDecodeError, TypeError, ValueError):
                    start_step = 0
            if args.pass_offset is not None:
                pass_offset = max(0, int(args.pass_offset))
            else:
                pass_offset = _detect_pass_offset(out)
        else:
            wiped = wipe_run_logs(out)
            if wiped:
                print(
                    f"[xsa_ttt] wiped prior logs/metrics: {', '.join(wiped)}",
                    flush=True,
                )
            enable_deterministic(int(cfg.seed))
            model = build_model(cfg, device=device)
        nparams = count_parameters(model)
        print(
            f"[xsa_ttt] params={nparams:,} stream_passes={cfg.stream_passes} "
            f"ac_every_pass={int(args.ac_every_pass)} "
            f"pass_offset={pass_offset}",
            flush=True,
        )
        online_train = not args.no_online_retrain
        # Keep RNG aligned for any dropout / online-retrain AdamW noise.
        torch.manual_seed(int(cfg.seed) ^ (pass_offset * 0x9E3779B9))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(cfg.seed) ^ (pass_offset * 0x9E3779B9))
        train_stats = stream_pretrain(
            model,
            arr,
            cfg=cfg,
            device=device,
            out_dir=out,
            online_retrain_during_train=online_train,
            ac_every_pass=bool(args.ac_every_pass),
            ac_bytes=int(args.ac_bytes),
            ac_online_retrain=online_train and cfg.online_retrain_steps > 0,
            pass_offset=pass_offset,
            start_step=start_step,
            verify_decode=ac_verify_decode,
            verify_end=ac_verify_end,
            n_symbols_total=n_symbols_total,
            source_bytes=source_bytes,
        )
        # Drop bulky per-pass AC blobs from the JSON summary.
        summary = {
            k: v for k, v in train_stats.items() if k != "ac_passes"
        }
        if train_stats.get("ac_passes"):
            summary["ac_passes"] = train_stats["ac_passes"]
        (out / "train_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        if train_stats.get("ac_passes"):
            last_ac = train_stats["ac_passes"][-1]
            (out / "ac_summary.json").write_text(
                json.dumps(last_ac, indent=2) + "\n"
            )
            print(
                f"[xsa_ttt] AC curve: {len(train_stats['ac_passes'])} passes → "
                f"{train_stats.get('ac_curve')}",
                flush=True,
            )
        if train_stats.get("early_stop"):
            print(
                f"[xsa_ttt] early stop ({train_stats.get('reason')}); "
                f"ckpt={train_stats.get('checkpoint_last')} "
                f"chart={train_stats.get('chart')}",
                flush=True,
            )
            return 0
        eval_ckpt = out / "last.safetensors"
        # Per-pass AC already covered the plateau curve; skip duplicate final AC.
        if args.ac_every_pass:
            return 0

    online = not args.no_online_retrain and cfg.online_retrain_steps > 0
    n_data = int(arr.shape[0])
    # Skip TF when: AC-from-init, explicit SKIP_TF/--skip-tf, or eval-only+AC.
    # Contest/compare cares about AC; TF at block=16k + strict math-SDP OOMs
    # easily. Explicit TF_BYTES=N still forces a probe.
    skip_tf = args.tf_bytes is None and (
        bool(ac_from_init)
        or bool(args.skip_tf)
        or _env_flag("SKIP_TF", False)
        or (args.eval_only is not None and bool(args.ac))
    )
    if skip_tf:
        why = (
            "AC-from-init"
            if ac_from_init
            else "eval-only+AC"
            if args.eval_only is not None and bool(args.ac)
            else "SKIP_TF"
        )
        print(
            f"[xsa_ttt] skipping teacher-forced probe ({why}; "
            "set TF_BYTES=N to force)",
            flush=True,
        )
    else:
        # TF probe: never silently expand to full corpus when AC_BYTES=0.
        # Explicit TF_BYTES=0 still means full-file TF.
        if args.tf_bytes is not None:
            tf_n = (
                n_data if int(args.tf_bytes) <= 0 else min(n_data, int(args.tf_bytes))
            )
        elif int(args.ac_bytes) <= 0:
            tf_n = min(n_data, 1_048_576)
        else:
            tf_n = min(n_data, max(int(args.ac_bytes) * 4, 262_144))
        print(
            f"[xsa_ttt] teacher-forced probe n={tf_n:,} "
            f"(AC next={'full' if int(args.ac_bytes) <= 0 and args.ac else args.ac_bytes})",
            flush=True,
        )
        tf = measure_teacher_forced_bpb(
            model,
            arr,
            cfg=cfg,
            device=device,
            max_bytes=tf_n,
            online_retrain_enabled=online,
        )
        print(
            f"[xsa_ttt] teacher-forced bpb={tf['bpb']:.5f} "
            f"(n={tf['n_bytes']:,} retrains={tf['retrain_count']})",
            flush=True,
        )
        (out / "eval_tf.json").write_text(json.dumps(tf, indent=2) + "\n")
        empty_cache(device)

    if args.ac:
        empty_cache(device)
        ckpt_label: str
        if ac_from_init:
            ckpt_label = f"deterministic_init:seed={cfg.seed}"
        else:
            ckpt = (
                eval_ckpt
                if eval_ckpt is not None and eval_ckpt.is_file()
                else out / "last.safetensors"
            )
            if ckpt.is_file():
                # Fresh weights for AC (TF probe / prior steps may have mutated).
                # Must re-apply runtime overrides — sibling JSON can ship
                # gradient_checkpointing=False which OOMs 16k full retrain.
                model, cfg = load_checkpoint_safetensors(ckpt, device=device, cfg=None)
                _apply_runtime_overrides(cfg)
                model.gradient_checkpointing = bool(cfg.gradient_checkpointing)
            ckpt_label = str(ckpt)
            anneal = ""
            anneal_mode = str(
                getattr(cfg, "ttt_replenish_anneal", "exp")
            ).lower()
            if anneal_mode == "linear":
                total_override = int(
                    getattr(cfg, "ttt_replenish_total_bytes", 0)
                )
                total_label = (
                    f"{total_override:,}B" if total_override else "stream"
                )
                anneal = (
                    f" anneal=linear(1-pos/total) total={total_label}"
                    f" lr_min={cfg.ttt_replenish_lr_min:g}"
                    f" steps_min={cfg.ttt_replenish_steps_min}"
                )
            elif float(getattr(cfg, "ttt_replenish_lr_half_mb", 0.0)) > 0:
                anneal = (
                    f" anneal=exp half={cfg.ttt_replenish_lr_half_mb:g}MB"
                    f" lr_min={cfg.ttt_replenish_lr_min:g}"
                    f" steps_min={cfg.ttt_replenish_steps_min}"
                )
            extras = ""
            if bool(getattr(cfg, "ttt_replenish_accum", False)):
                extras += " accum=1(one step/boundary)"
            scale = float(getattr(cfg, "ttt_replenish_replay_lr_scale", 1.0))
            if scale != 1.0:
                extras += f" replay_lr_scale={scale:g}"
            recent = float(
                getattr(cfg, "ttt_replenish_replay_recent_mb", 0.0)
            )
            if recent > 0 and int(cfg.online_retrain_steps) > 1:
                extras += f" replay_recent={recent:g}MB"
            if str(getattr(cfg, "ttt_replenish_anchor", "") or ""):
                extras += (
                    f" anchor=on(rate="
                    f"{float(getattr(cfg, 'ttt_replenish_anchor_rate', 0.05)):g})"
                )
            warmup = int(getattr(cfg, "ttt_replenish_warmup_steps", 0))
            if warmup > 0:
                extras += f" warmup={warmup}(lr=0)"
            xm_k = int(getattr(cfg, "ttt_replenish_xm_k", 0))
            if xm_k >= 2:
                extras += (
                    f" xm=K{xm_k}(probe="
                    f"{int(getattr(cfg, 'ttt_replenish_xm_probe_chunks', 1))}"
                    f"ch"
                    + (
                        f",4x<{int(getattr(cfg, 'ttt_replenish_xm_4x_until_bytes', 0)) / (1024 * 1024):g}MB"
                        if int(getattr(cfg, "ttt_replenish_xm_4x_until_bytes", 0))
                        > 0
                        else ""
                    )
                    + ")"
                )
            print(
                f"[xsa_ttt] AC model block={cfg.block_size} "
                f"retrain_steps={cfg.online_retrain_steps} "
                f"retrain_every={cfg.online_retrain_every} "
                f"retrain_lr={cfg.ttt_lora_lr:g} "
                f"betas=({cfg.ttt_beta1:g},{cfg.ttt_beta2:g}) "
                f"grad_ckpt={int(model.gradient_checkpointing)} "
                f"looping={int(getattr(model, 'looping_active', False))}"
                + anneal
                + extras,
                flush=True,
            )
        meta = run_ac_measure(
            model,
            arr,
            cfg=cfg,
            device=device,
            out_dir=out,
            ac_bytes=int(args.ac_bytes),
            online_retrain=online,
            progress=True,
            tag="final",
            verify_decode=ac_verify_decode,
            verify_end=ac_verify_end,
            n_symbols_total=n_symbols_total,
            source_bytes=source_bytes,
        )
        meta["ckpt"] = ckpt_label
        meta["ac_from_init"] = bool(ac_from_init)
        (out / "ac_summary.json").write_text(json.dumps(meta, indent=2) + "\n")
        if int(args.ac_bytes) <= 0:
            _append_metrics(out / "ac_curve.jsonl", meta)
            (out / f"ac_{meta.get('tag', 'final_fullsha')}.json").write_text(
                json.dumps(meta, indent=2) + "\n"
            )
        print(
            f"[xsa_ttt] AC {_fmt_ac_line(meta)} "
            f"retrains={meta['retrain_count']}"
            + (
                f" sha256_ok={meta.get('sha256_ok')}"
                if "sha256_ok" in meta
                else ""
            ),
            flush=True,
        )
        # None = encode-only (no in-process verify). Only fail on explicit False.
        if meta.get("roundtrip_ok") is False:
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
