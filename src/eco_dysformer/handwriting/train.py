"""Train the handwriting reversal classifier (RQ3 auxiliary branch).

Trains the lightweight CNN on the handwriting Train split and evaluates on the
Test split -- a real, reportable handwriting-domain result (accuracy / AUROC)
plus the efficiency metrics the project cares about (params, latency). The
trained model is saved so the risk-feature stage can score synthetic writing
samples. This is Kaggle-first (needs torch + the extracted images).

    python -m eco_dysformer.handwriting.train
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from eco_dysformer.eval.metrics import classification_metrics
from eco_dysformer.handwriting.data import build_loaders
from eco_dysformer.handwriting.encoder import build_handwriting_cnn


@torch.no_grad()
def _evaluate(model, loader, device) -> dict:
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        prob = torch.softmax(model(x.to(device)), dim=-1)
        ps.append(prob.cpu().numpy())
        ys.append(y.numpy())
    y_true = np.concatenate(ys)
    prob = np.concatenate(ps)
    if prob.shape[1] == 2:
        m = classification_metrics(y_true, prob[:, 1])
    else:  # multiclass -> report accuracy only
        m = {"accuracy": float((prob.argmax(1) == y_true).mean()),
             "n": int(len(y_true))}
    return m


def train_handwriting(cfg, seed: int, device=None) -> dict:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    tr, te, meta = build_loaders(cfg, seed)
    model = build_handwriting_cnn(cfg, meta["in_channels"], meta["n_classes"]).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.rq3.handwriting.train.lr)
    loss_fn = nn.CrossEntropyLoss()
    epochs = int(cfg.rq3.handwriting.train.epochs)

    epoch_times = []
    for ep in range(epochs):
        model.train()
        t0 = time.perf_counter()
        running = 0.0
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * len(y)
        dt = time.perf_counter() - t0
        epoch_times.append(dt)
        print(f"  epoch {ep+1}/{epochs}  loss={running/meta['n_train']:.4f}  {dt:.1f}s")

    test_metrics = _evaluate(model, te, device)
    n_params = sum(p.numel() for p in model.parameters())

    # inference latency (per batch) on one test batch
    xb = next(iter(te))[0].to(device)
    with torch.no_grad():
        for _ in range(3):
            model(xb)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            model(xb)
        if device.type == "cuda":
            torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - t0) / 20 * 1000

    result = {
        "meta": meta,
        "test_metrics": test_metrics,
        "param_count": int(n_params),
        "mean_epoch_time_s": float(np.mean(epoch_times)),
        "inference_latency_ms_per_batch": float(latency_ms),
        "batch_size": cfg.rq3.handwriting.train.batch_size,
        "note": ("Handwriting-domain result on a DISJOINT cohort. Reversal vs "
                 "Normal character classification; not a subject-level diagnosis."),
    }

    out_dir = Path(cfg.paths.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "rq3_handwriting_classifier.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    torch.save(model.state_dict(), out_dir / "rq3_handwriting_cnn.pt")
    return model, result


if __name__ == "__main__":
    import sys
    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from eco_dysformer.config import load_config
    from eco_dysformer.seed import set_global_seed

    cfg = load_config()
    set_global_seed(cfg.seed)
    _, res = train_handwriting(cfg, cfg.seed)
    print("\nHandwriting classifier test metrics:", res["test_metrics"])
    print("params:", res["param_count"], "| latency/batch(ms):",
          round(res["inference_latency_ms_per_batch"], 2))
