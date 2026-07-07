"""Paired statistics and effect sizes for small-N evaluation.

Every "matches or exceeds" claim in this project is backed here: a paired
Wilcoxon signed-rank test across outer folds, plus bootstrap confidence
intervals. RQ2's gaze-shift magnitude is quantified with Cohen's d across the
syllable -> narrative -> pseudo-text gradient. NumPy/SciPy only, so this runs and
is tested in the bare local env.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats


def benjamini_hochberg(pvalues) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted q-values for a list of p-values.

    NaN p-values pass through as NaN and are excluded from the ranking.
    """
    p = np.asarray(pvalues, dtype=float)
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    m = int(ok.sum())
    if m == 0:
        return q
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    ranked = p[order]
    adj = ranked * m / (np.arange(1, m + 1))
    # enforce monotonicity (step-up)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    q[order] = np.clip(adj, 0.0, 1.0)
    return q


def cohens_d(x, y, *, paired: bool = False) -> float:
    """Cohen's d effect size for the difference in means of ``x`` vs ``y``.

    ``paired=True`` uses the standard deviation of the paired differences
    (d_z); otherwise the pooled standard deviation.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    if paired:
        assert len(x) == len(y), "paired Cohen's d needs equal-length samples"
        diff = x - y
        sd = diff.std(ddof=1)
        return float(diff.mean() / sd) if sd > 0 else float("nan")
    nx, ny = len(x), len(y)
    pooled = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1))
                     / (nx + ny - 2))
    return float((x.mean() - y.mean()) / pooled) if pooled > 0 else float("nan")


def wilcoxon_paired(a, b, *, alternative: str = "two-sided") -> dict:
    """Paired Wilcoxon signed-rank test on per-fold scores ``a`` vs ``b``.

    Returns statistic, p-value, n, median difference, and a flag for the
    degenerate all-zero-difference case (SciPy raises there; we report cleanly).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert a.shape == b.shape, "paired test needs equal-length inputs"
    diff = a - b
    if np.allclose(diff, 0.0):
        return {"statistic": 0.0, "pvalue": 1.0, "n": int(len(a)),
                "median_diff": 0.0, "note": "all paired differences are zero"}
    try:
        res = sp_stats.wilcoxon(a, b, alternative=alternative,
                                zero_method="wilcox")
        stat, p = float(res.statistic), float(res.pvalue)
        note = ""
    except ValueError as e:
        stat, p, note = float("nan"), float("nan"), str(e)
    return {"statistic": stat, "pvalue": p, "n": int(len(a)),
            "median_diff": float(np.median(diff)), "note": note}


def bootstrap_ci(values, *, n_resamples: int = 2000, ci: float = 0.95,
                 seed: int = 0, statistic=np.mean) -> dict:
    """Percentile bootstrap CI for a statistic of a 1-D sample."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n": 0}
    rng = np.random.default_rng(seed)
    n = len(values)
    boot = np.empty(n_resamples)
    for i in range(n_resamples):
        boot[i] = statistic(values[rng.integers(0, n, n)])
    alpha = (1.0 - ci) / 2.0
    return {
        "point": float(statistic(values)),
        "lo": float(np.quantile(boot, alpha)),
        "hi": float(np.quantile(boot, 1 - alpha)),
        "n": int(n),
        "ci": ci,
    }


def paired_bootstrap_ci(a, b, *, n_resamples: int = 2000, ci: float = 0.95,
                        seed: int = 0) -> dict:
    """Bootstrap CI for the mean paired difference ``a - b`` (resample folds)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert a.shape == b.shape, "paired bootstrap needs equal-length inputs"
    diff = a - b
    out = bootstrap_ci(diff, n_resamples=n_resamples, ci=ci, seed=seed)
    out["mean_diff"] = float(np.nanmean(diff))
    return out


def compare_arms(scores_a, scores_b, *, name_a: str, name_b: str,
                 alternative: str = "two-sided", n_resamples: int = 2000,
                 ci: float = 0.95, seed: int = 0) -> dict:
    """Full paired comparison of two arms' per-fold scores (RQ1/RQ2 headline).

    Bundles the Wilcoxon test, the paired-difference bootstrap CI, and each
    arm's own bootstrap CI into one JSON-serializable record.
    """
    scores_a = list(map(float, scores_a))
    scores_b = list(map(float, scores_b))
    return {
        "arm_a": name_a,
        "arm_b": name_b,
        "scores_a": scores_a,
        "scores_b": scores_b,
        "mean_a": float(np.nanmean(scores_a)),
        "mean_b": float(np.nanmean(scores_b)),
        "wilcoxon": wilcoxon_paired(scores_a, scores_b, alternative=alternative),
        "paired_diff_ci": paired_bootstrap_ci(
            scores_a, scores_b, n_resamples=n_resamples, ci=ci, seed=seed),
        "ci_a": bootstrap_ci(scores_a, n_resamples=n_resamples, ci=ci, seed=seed),
        "ci_b": bootstrap_ci(scores_b, n_resamples=n_resamples, ci=ci, seed=seed),
    }


if __name__ == "__main__":
    # Sanity smoke test (synthetic, NOT a result): two arms where A slightly
    # exceeds B on 5 folds.
    rng = np.random.default_rng(0)
    a = 0.85 + 0.02 * rng.standard_normal(5)
    b = a - 0.01 - 0.005 * rng.standard_normal(5)
    import json
    print(json.dumps(compare_arms(a, b, name_a="A", name_b="B", seed=0), indent=2))
