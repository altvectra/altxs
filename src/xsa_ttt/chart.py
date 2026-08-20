"""Loss / BPB chart for xsa_ttt runs."""

from __future__ import annotations

import json
from pathlib import Path


def save_loss_chart(
    metrics_jsonl: Path | str,
    out_path: Path | str,
    *,
    title: str = "xsa_ttt",
) -> Path | None:
    """Plot train CE / BPB from metrics.jsonl → PNG. Returns path or None."""
    metrics_jsonl = Path(metrics_jsonl)
    out_path = Path(out_path)
    if not metrics_jsonl.is_file():
        return None

    steps: list[int] = []
    ces: list[float] = []
    bpbs: list[float] = []
    with metrics_jsonl.open(encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "bpb" not in row or "step" not in row:
                continue
            if row.get("event"):
                continue
            steps.append(int(row["step"]))
            ces.append(float(row.get("ce_nats", row.get("ce", float("nan")))))
            bpbs.append(float(row["bpb"]))

    if not steps:
        return None

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[chart] matplotlib not available; skip loss chart", flush=True)
        return None

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(steps, bpbs, color="#1f77b4", linewidth=1.2)
    axes[0].set_ylabel("train BPB")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(title)

    axes[1].plot(steps, ces, color="#ff7f0e", linewidth=1.2)
    axes[1].set_ylabel("train CE (nats)")
    axes[1].set_xlabel("step")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[chart] wrote {out_path}", flush=True)
    return out_path
