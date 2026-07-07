"""RQ2: gaze-feature shift across the complexity gradient (the descriptive arm).

Because the complexity-conditioned classifier does NOT beat the complexity-blind
one (an honest null; the linguistic features are passage-level constants), RQ2's
positive evidence lives here, in the descriptive analysis of how gaze differs
between dyslexic and typical readers and how that difference behaves across the
syllable -> narrative -> pseudo-text gradient. Three complementary readings, all
from the recomputed gaze table (no model, no CV):

1. Per-passage separation -- Cohen's d + Mann-Whitney U (dyslexic vs typical) of
   each gaze feature at each passage, BH-FDR corrected across all tests.

2. Gradient interaction (the RQ2 hypothesis proper) -- per subject we fit the
   feature's slope over ranks [0,1,2], then test whether the two groups' slopes
   differ (Mann-Whitney U + Cohen's d), BH-FDR corrected across features. A
   group difference in slope IS the group x complexity interaction.

3. A gradient figure for the priority features (mean +/- SE per passage, by group).

NumPy / SciPy / pandas / matplotlib only -- runs fully in the bare local env.
Nothing here is a headline classification claim; these are legitimate descriptive
effect sizes with significance and multiple-comparison control.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eco_dysformer.eval.stats import benjamini_hochberg, cohens_d  # noqa: E402

RANKS = [0, 1, 2]
# Features the brief singles out for RQ2 (fixation / regression), reported first.
RQ2_PRIORITY = ["regression_ratio", "fix_count", "mean_fix_dur", "total_read_time_ms"]


def _mwu(dys: np.ndarray, typ: np.ndarray) -> tuple[float, float]:
    dys = dys[~np.isnan(dys)]
    typ = typ[~np.isnan(typ)]
    if len(dys) < 1 or len(typ) < 1:
        return float("nan"), float("nan")
    try:
        r = mannwhitneyu(dys, typ, alternative="two-sided")
        return float(r.statistic), float(r.pvalue)
    except ValueError:                      # e.g. all values identical
        return float("nan"), float("nan")


def per_passage_effects(gaze_df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for feat in features:
        for rank in RANKS:
            sub = gaze_df[gaze_df["complexity_rank"] == rank]
            dys = sub[sub["class_id"] == 1][feat].to_numpy(dtype=float)
            typ = sub[sub["class_id"] == 0][feat].to_numpy(dtype=float)
            u, p = _mwu(dys, typ)
            rows.append({
                "feature": feat,
                "complexity_rank": rank,
                "mean_dyslexic": float(np.nanmean(dys)),
                "mean_typical": float(np.nanmean(typ)),
                "cohens_d": cohens_d(dys, typ),
                "mannwhitney_u": u,
                "p_value": p,
                "n_dyslexic": int(np.sum(~np.isnan(dys))),
                "n_typical": int(np.sum(~np.isnan(typ))),
            })
    df = pd.DataFrame(rows)
    df["q_value_bh"] = benjamini_hochberg(df["p_value"].to_numpy())
    return df


def _subject_slope(sub: pd.DataFrame, feat: str) -> float:
    """Linear slope of ``feat`` over complexity_rank for one subject."""
    sub = sub.sort_values("complexity_rank")
    x = sub["complexity_rank"].to_numpy(dtype=float)
    y = sub[feat].to_numpy(dtype=float)
    ok = ~np.isnan(y)
    if ok.sum() < 2:
        return float("nan")
    return float(np.polyfit(x[ok], y[ok], 1)[0])


def subject_slopes(gaze_df: pd.DataFrame, feat: str) -> pd.DataFrame:
    """Per-subject slope of ``feat`` across the gradient, with class label."""
    return (gaze_df.groupby("subject_id")
            .apply(lambda g: pd.Series({
                "slope": _subject_slope(g, feat),
                "class_id": int(g["class_id"].iloc[0]),
            }), include_groups=False)
            .reset_index())


def gradient_interaction(gaze_df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for feat in features:
        sl = subject_slopes(gaze_df, feat)
        dys = sl[sl["class_id"] == 1]["slope"].to_numpy(dtype=float)
        typ = sl[sl["class_id"] == 0]["slope"].to_numpy(dtype=float)
        u, p = _mwu(dys, typ)
        rows.append({
            "feature": feat,
            "mean_slope_dyslexic": float(np.nanmean(dys)),
            "mean_slope_typical": float(np.nanmean(typ)),
            "cohens_d_slope": cohens_d(dys, typ),
            "mannwhitney_u": u,
            "p_value": p,
        })
    df = pd.DataFrame(rows)
    df["q_value_bh"] = benjamini_hochberg(df["p_value"].to_numpy())
    return df


def make_gradient_figure(gaze_df: pd.DataFrame, out_path: Path,
                         features: list[str] | None = None) -> Path | None:
    """Mean +/- SE of each priority feature across the 3 passages, by group."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    features = features or RQ2_PRIORITY
    passages = ["syllables", "meaningful", "pseudo"]
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    for ax, feat in zip(axes.ravel(), features):
        for cls, name, color in [(0, "typical", "#4C78A8"), (1, "dyslexic", "#E45756")]:
            means, ses = [], []
            for rank in RANKS:
                vals = gaze_df[(gaze_df["complexity_rank"] == rank)
                               & (gaze_df["class_id"] == cls)][feat].to_numpy(float)
                vals = vals[~np.isnan(vals)]
                means.append(vals.mean())
                ses.append(vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)
            ax.errorbar(range(3), means, yerr=ses, marker="o", capsize=3,
                        label=name, color=color)
        ax.set_xticks(range(3))
        ax.set_xticklabels(passages)
        ax.set_title(feat)
        ax.grid(alpha=0.3)
    axes[0, 0].legend()
    fig.suptitle("RQ2: gaze feature across complexity gradient (mean +/- SE)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def run(cfg) -> dict[str, pd.DataFrame]:
    from eco_dysformer.features.gaze import GAZE_FEATURE_NAMES
    gaze_path = Path(cfg.paths.features_dir) / "gaze_features.csv"
    if not gaze_path.is_file():
        raise FileNotFoundError(f"{gaze_path} missing -- run features.assemble first.")
    gaze_df = pd.read_csv(gaze_path)
    features = [f for f in GAZE_FEATURE_NAMES if f in gaze_df.columns]
    features = RQ2_PRIORITY + [f for f in features if f not in RQ2_PRIORITY]

    per_pass = per_passage_effects(gaze_df, features)
    interaction = gradient_interaction(gaze_df, features)

    out_dir = Path(cfg.paths.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_pass.to_csv(out_dir / "rq2_effects_per_passage.csv", index=False)
    interaction.to_csv(out_dir / "rq2_gradient_interaction.csv", index=False)
    fig_path = make_gradient_figure(gaze_df, Path(cfg.paths.figures_dir) / "rq2_gradient.png")
    return {"per_passage": per_pass, "interaction": interaction,
            "figure": str(fig_path) if fig_path else None}


if __name__ == "__main__":
    from eco_dysformer.config import load_config
    cfg = load_config()
    res = run(cfg)
    pd.set_option("display.width", 170)
    print("=== RQ2 per-passage (dyslexic vs typical), priority features ===")
    pp = res["per_passage"]
    cols = ["feature", "complexity_rank", "cohens_d", "p_value", "q_value_bh"]
    print(pp[pp["feature"].isin(RQ2_PRIORITY)][cols].round(4).to_string(index=False))
    print("\n=== RQ2 gradient interaction: group difference in per-subject slope ===")
    it = res["interaction"]
    icols = ["feature", "mean_slope_dyslexic", "mean_slope_typical",
             "cohens_d_slope", "p_value", "q_value_bh"]
    print(it[it["feature"].isin(RQ2_PRIORITY)][icols].round(4).to_string(index=False))
    print(f"\nfigure -> {res['figure']}")
