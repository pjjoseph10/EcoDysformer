"""Extract paired cross-attention weights over the interpretable passage tokens.

The fusion block's cross-attention is over the child's three passage tokens
(syllables / meaningful / pseudo), so its weights are directly interpretable:
"when encoding passage i, how much does the model attend to the complexity
context of passage j?". We average the (P x P) attention over the test children
of each outer fold, then aggregate across folds and report the mean matrix and
its across-fold standard deviation (stability).

Works only for the complexity-conditioned arm (the blind arm has no fusion
cross-attention). Requires a trained pipeline per fold.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from eco_dysformer.data.tensors import ChildArrays
from eco_dysformer.models.pipeline import FittedPipeline, resolve_device


def run_attention_extraction(cfg, arrays: ChildArrays, folds) -> dict:
    device = resolve_device(cfg)
    seed = cfg.seed
    passages = arrays.passage_names
    per_fold_mats = []

    for fold in folds:
        Xg_tr = arrays.X_gaze[fold.train_idx]
        Xl_tr = arrays.X_ling[fold.train_idx]
        y_tr = arrays.y[fold.train_idx]
        Xg_te = arrays.X_gaze[fold.test_idx]
        Xl_te = arrays.X_ling[fold.test_idx]

        pipe = FittedPipeline(cfg=cfg, seed=seed, device=device,
                              attention="performer", conditioned=True)
        pipe.fit_encoder(Xg_tr, Xl_tr, y_tr)
        mat = pipe.fusion_attention(Xg_te, Xl_te)     # (P, P)
        if mat is not None:
            per_fold_mats.append(mat)

    if not per_fold_mats:
        return {"available": False}

    stack = np.stack(per_fold_mats)                    # (folds, P, P)
    mean_mat = stack.mean(axis=0)
    std_mat = stack.std(axis=0)
    return {
        "available": True,
        "passages": passages,
        "mean_attention": mean_mat.tolist(),
        "std_attention": std_mat.tolist(),
        "across_fold_mean_std": float(std_mat.mean()),
    }


def save_attention(cfg, result: dict) -> None:
    if not result.get("available"):
        return
    res_dir = Path(cfg.paths.results_dir)
    res_dir.mkdir(parents=True, exist_ok=True)
    passages = result["passages"]
    df = pd.DataFrame(result["mean_attention"], index=passages, columns=passages)
    df.to_csv(res_dir / "fusion_attention_mean.csv")
    pd.DataFrame(result["std_attention"], index=passages, columns=passages).to_csv(
        res_dir / "fusion_attention_std.csv")
