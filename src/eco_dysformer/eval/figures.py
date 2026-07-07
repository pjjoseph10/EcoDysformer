"""Paper figures built from the written result artifacts.

Kept separate from the analyses so figures regenerate from the CSV/JSON outputs
without re-running the models. matplotlib only; each function no-ops cleanly if
its input artifact is missing.

    calibration_reliability  <- cv_oof_predictions.csv   (needs the CV run)
    lime_importance_bar      <- lime_pooled_importance.csv (needs the explain run)
The RQ2 gradient figure lives with its analysis in rq2_effects.make_gradient_figure.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def calibration_reliability(oof_csv: str | Path, out_path: str | Path,
                            arm: str = "performer_conditioned",
                            n_bins: int = 10) -> Path | None:
    """Reliability diagram + confidence histogram from pooled OOF predictions."""
    oof_csv, out_path = Path(oof_csv), Path(out_path)
    if not oof_csv.is_file():
        return None
    df = pd.read_csv(oof_csv)
    df = df[df["arm"] == arm]
    if df.empty:
        return None
    y = df["y_true"].to_numpy(float)
    p = df["y_prob"].to_numpy(float)

    edges = np.linspace(0, 1, n_bins + 1)
    xs, ys, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if m.sum():
            xs.append(p[m].mean())
            ys.append(y[m].mean())
            ns.append(int(m.sum()))
    ece = float(np.sum([(n / len(p)) * abs(x - yy)
                        for x, yy, n in zip(xs, ys, ns)]))

    plt = _mpl()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 6),
                                   gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot([0, 1], [0, 1], "--", color="grey", label="perfect")
    ax1.plot(xs, ys, "o-", color="#E45756", label=arm)
    ax1.set_xlabel("mean predicted P(dyslexic)")
    ax1.set_ylabel("observed fraction dyslexic")
    ax1.set_title(f"Calibration reliability (ECE={ece:.3f})")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.hist(p, bins=edges, color="#4C78A8", alpha=0.8)
    ax2.set_xlabel("predicted probability")
    ax2.set_ylabel("count")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def lime_importance_bar(pooled_csv: str | Path, out_path: str | Path,
                        top_n: int = 15) -> Path | None:
    """Horizontal bar chart of the top-N pooled LIME attributions."""
    pooled_csv, out_path = Path(pooled_csv), Path(out_path)
    if not pooled_csv.is_file():
        return None
    s = pd.read_csv(pooled_csv, index_col=0).iloc[:, 0].sort_values(ascending=False)
    s = s.head(top_n).iloc[::-1]

    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(s) + 1))
    ax.barh(range(len(s)), s.to_numpy(), color="#54A24B")
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels(s.index, fontsize=8)
    ax.set_xlabel("mean |LIME attribution| (pooled over folds)")
    ax.set_title("Pre-fusion feature attributions (original features)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def make_all_figures(cfg) -> dict:
    res, figs = Path(cfg.paths.results_dir), Path(cfg.paths.figures_dir)
    out = {}
    out["calibration"] = str(calibration_reliability(
        res / "cv_oof_predictions.csv", figs / "calibration_reliability.png") or "")
    out["lime"] = str(lime_importance_bar(
        res / "lime_pooled_importance.csv", figs / "lime_importance.png") or "")
    return out
