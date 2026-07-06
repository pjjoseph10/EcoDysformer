"""Operational metrics: parameter count, peak GPU memory, latency, epoch time.

These are first-class reported outcomes for the project's efficiency ("Eco")
framing, not afterthoughts. Torch-dependent; GPU-memory helpers degrade cleanly
to ``None`` on CPU so the harness still runs locally.
"""
from __future__ import annotations

import time
from contextlib import contextmanager

import torch


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    ps = model.parameters()
    return sum(p.numel() for p in ps if (p.requires_grad or not trainable_only))


@contextmanager
def track_peak_gpu_memory(device):
    """Context manager yielding a dict; on exit fills ``peak_mb`` (None on CPU)."""
    out = {"peak_mb": None}
    is_cuda = torch.device(device).type == "cuda"
    if is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    try:
        yield out
    finally:
        if is_cuda:
            torch.cuda.synchronize()
            out["peak_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)


@torch.no_grad()
def measure_inference_latency(model: torch.nn.Module, inputs: tuple,
                              device, *, n_batches: int = 50,
                              warmup: int = 5) -> dict:
    """Mean/median forward latency (ms) over ``n_batches`` after warmup."""
    model.eval().to(device)
    inputs = tuple(x.to(device) if isinstance(x, torch.Tensor) else x
                   for x in inputs)
    is_cuda = torch.device(device).type == "cuda"

    for _ in range(warmup):
        model(*inputs)
    if is_cuda:
        torch.cuda.synchronize()

    times = []
    for _ in range(n_batches):
        t0 = time.perf_counter()
        model(*inputs)
        if is_cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return {
        "latency_ms_mean": float(sum(times) / len(times)),
        "latency_ms_median": float(times[len(times) // 2]),
        "n_batches": n_batches,
        "batch_size": int(inputs[0].shape[0]) if inputs else None,
    }


def operational_summary(model: torch.nn.Module, inputs: tuple, device,
                        *, latency_batches: int = 50,
                        epoch_time_s: float | None = None) -> dict:
    """Bundle param count + peak GPU mem (for a forward) + latency + epoch time."""
    with track_peak_gpu_memory(device) as mem:
        _ = measure_inference_latency(model, inputs, device,
                                      n_batches=latency_batches)
    lat = measure_inference_latency(model, inputs, device,
                                    n_batches=latency_batches)
    return {
        "param_count": count_parameters(model),
        "peak_gpu_mem_mb": mem["peak_mb"],
        "epoch_time_s": epoch_time_s,
        **lat,
    }
