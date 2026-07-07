"""RQ1 sequence-length crossover: where does linear attention overtake quadratic?

Performer's linear-attention advantage is ASYMPTOTIC. On the short engineered
sequences (3 passage tokens) its constant overhead makes it slower than
quadratic -- an honest negative we expect and report. The real question is where,
on the raw fixation/saccade event stream (hundreds of events per passage), the
Performer's O(N) scaling overtakes the quadratic's O(N^2). This module measures
forward-pass time (and peak GPU memory) for both a Performer and a
parameter-matched quadratic self-attention block across a sweep of sequence
lengths, and reports the empirical crossover length (or that none was reached).

Writes ``rq1_crossover.csv`` and a figure. Torch-dependent; runs on CPU or GPU.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from eco_dysformer.eval.operational import track_peak_gpu_memory
from eco_dysformer.models.blocks import SelfAttentionBlock


def _time_block(block, x, repeats: int, device) -> float:
    """Mean ms/forward, or NaN if the block OOMs (quadratic can, at large N)."""
    block.eval().to(device)
    x = x.to(device)
    is_cuda = torch.device(device).type == "cuda"
    try:
        with torch.no_grad():
            for _ in range(3):                  # warmup
                block(x)
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(repeats):
                block(x)
            if is_cuda:
                torch.cuda.synchronize()
        return (time.perf_counter() - t0) / repeats * 1000.0   # ms/forward
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            if is_cuda:
                torch.cuda.empty_cache()
            return float("nan")     # OOM -> cannot run at this length
        raise


def run_crossover(cfg, device: str | None = None,
                  batch_size: int = 4) -> pd.DataFrame:
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ge = cfg.model.gaze_encoder
    d, h, nf = ge.d_model, ge.n_heads, ge.performer_features
    lengths = list(cfg.rq1_crossover.seq_lengths)
    repeats = int(cfg.rq1_crossover.repeats)

    perf = SelfAttentionBlock("performer", d, h, dropout=0.0, n_features=nf,
                              seed=cfg.seed)
    quad = SelfAttentionBlock("quadratic", d, h, dropout=0.0, seed=cfg.seed)

    rows = []
    for L in lengths:
        x = torch.randn(batch_size, L, d)
        with track_peak_gpu_memory(dev) as m_p:
            t_perf = _time_block(perf, x, repeats, dev)
        perf_mem = m_p["peak_mb"]
        with track_peak_gpu_memory(dev) as m_q:
            t_quad = _time_block(quad, x, repeats, dev)
        quad_mem = m_q["peak_mb"]
        rows.append({
            "seq_len": L,
            "performer_ms": t_perf,
            "quadratic_ms": t_quad,
            "performer_faster": bool(t_perf < t_quad),
            "performer_peak_mb": perf_mem,
            "quadratic_peak_mb": quad_mem,
            "device": dev,
        })
    df = pd.DataFrame(rows)

    # Crossover = smallest length from which Performer stays faster.
    crossover = None
    faster = df["performer_faster"].to_numpy()
    for i in range(len(faster)):
        if faster[i] and faster[i:].all():
            crossover = int(df["seq_len"].iloc[i])
            break
    df.attrs["crossover_seq_len"] = crossover
    return df


def save_crossover(cfg, df: pd.DataFrame) -> dict:
    res_dir = Path(cfg.paths.results_dir)
    fig_dir = Path(cfg.paths.figures_dir)
    res_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(res_dir / "rq1_crossover.csv", index=False)

    crossover = df.attrs.get("crossover_seq_len")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(df["seq_len"], df["performer_ms"], "o-", label="Performer (linear)")
        ax.plot(df["seq_len"], df["quadratic_ms"], "s-", label="Quadratic (softmax)")
        ax.set_xlabel("sequence length (events)")
        ax.set_ylabel("forward time (ms)")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        title = "RQ1 attention scaling"
        if crossover is not None:
            ax.axvline(crossover, color="grey", ls="--", alpha=0.7)
            title += f" (crossover @ N={crossover})"
        else:
            title += " (no crossover in tested range)"
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "rq1_crossover.png", dpi=120)
        plt.close(fig)
    except ImportError:
        pass

    return {"crossover_seq_len": crossover,
            "csv": str(res_dir / "rq1_crossover.csv")}


if __name__ == "__main__":
    import sys
    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from eco_dysformer.config import load_config
    from eco_dysformer.seed import set_global_seed

    cfg = load_config()
    set_global_seed(cfg.seed)
    df = run_crossover(cfg)
    info = save_crossover(cfg, df)
    print(df.to_string(index=False))
    print("crossover seq len:", info["crossover_seq_len"])