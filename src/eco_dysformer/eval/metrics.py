"""Classification and calibration metrics (NumPy-only).

Implemented without scikit-learn so the metrics run and are unit-tested in the
bare local environment and carry no heavy dependency. Covers the RQ1/RQ2
headline metrics (accuracy, F1, AUROC) plus calibration (ECE, Brier) which the
proposal threads throughout.

AUROC uses the rank / Mann-Whitney identity; ties are handled with average
ranks. ECE is the standard equal-width binned |confidence - accuracy|.
"""
from __future__ import annotations

import numpy as np


def _binarize(y_prob, threshold: float = 0.5) -> np.ndarray:
    return (np.asarray(y_prob, dtype=float) >= threshold).astype(int)


def accuracy(y_true, y_prob, threshold: float = 0.5) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = _binarize(y_prob, threshold)
    return float((y_true == y_pred).mean())


def f1_score(y_true, y_prob, threshold: float = 0.5) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = _binarize(y_prob, threshold)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom > 0 else 0.0


def auroc(y_true, y_prob) -> float:
    """Area under the ROC curve via the rank-sum (Mann-Whitney U) identity."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")  # undefined for a single class
    order = np.argsort(y_prob, kind="mergesort")
    ranks = np.empty(len(y_prob), dtype=float)
    ranks[order] = np.arange(1, len(y_prob) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(y_prob, return_inverse=True, return_counts=True)
    sum_ranks = np.zeros(len(counts))
    np.add.at(sum_ranks, inv, ranks)
    ranks = (sum_ranks / counts)[inv]
    sum_pos = ranks[y_true == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def brier_score(y_true, y_prob) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """Equal-width ECE over ``n_bins`` confidence bins."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(y_true)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y_prob > lo) & (y_prob <= hi) if lo > 0 else (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(conf - acc)
    return float(ece)


def classification_metrics(y_true, y_prob, *, threshold: float = 0.5,
                           n_bins: int = 10) -> dict:
    """Bundle every scalar metric for one set of predictions."""
    return {
        "accuracy": accuracy(y_true, y_prob, threshold),
        "f1": f1_score(y_true, y_prob, threshold),
        "auroc": auroc(y_true, y_prob),
        "brier": brier_score(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob, n_bins),
        "n": int(len(y_true)),
        "n_pos": int(np.asarray(y_true).sum()),
    }
