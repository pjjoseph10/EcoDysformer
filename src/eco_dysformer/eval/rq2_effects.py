"""RQ2 effect sizes: gaze-feature shift across the complexity gradient.

Two analyses, both from the recomputed gaze table (no model, no CV -- these are
descriptive effect sizes):

1. Per-passage separation -- Cohen's d (dyslexic vs typical, unpaired) of each
   gaze feature at each passage (syllables / meaningful / pseudo).

2. Gradient interaction -- the RQ2 hypothesis proper: does gaze shift
   *differentially* between groups as complexity rises? Per subject we fit the
   feature's slope over the ordered ranks [0,1,2], then take Cohen's d of that
   per-subject slope between groups. A non-trivial d means the groups' gaze
   trajectories diverge across the gradient.

Runs fully in the bare local env (NumPy/SciPy/pandas). Results are written to
``results_dir``; nothing is a headline claim without the paired tests in the CV
harness, but these effect sizes are legitimate descriptive outputs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eco_dysformer.eval.stats import cohens_d  # noqa: E402

RANKS = [0, 1, 2]


def per_passage_effects(gaze_df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for feat in features:
        for rank in RANKS:
            sub = gaze_df[gaze_df["complexity_rank"] == rank]
            dys = sub[sub["class_id"] == 1][feat].to_numpy(dtype=float)
            typ = sub[sub["class_id"] == 0][feat].to_numpy(dtype=float)
            rows.append({
                "feature": feat,
                "complexity_rank": rank,
                "mean_dyslexic": float(np.nanmean(dys)),
                "mean_typical": float(np.nanmean(typ)),
                "cohens_d": cohens_d(dys, typ),
                "n_dyslexic": int(np.sum(~np.isnan(dys))),
                "n_typical": int(np.sum(~np.isnan(typ))),
            })
    return pd.DataFrame(rows)


def _subject_slope(sub: pd.DataFrame, feat: str) -> float:
    """Linear slope of ``feat`` over complexity_rank for one subject."""
    sub = sub.sort_values("complexity_rank")
    x = sub["complexity_rank"].to_numpy(dtype=float)
    y = sub[feat].to_numpy(dtype=float)
    ok = ~np.isnan(y)
    if ok.sum() < 2:
        return float("nan")
    return float(np.polyfit(x[ok], y[ok], 1)[0])


def gradient_effects(gaze_df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for feat in features:
        slopes = (gaze_df.groupby("subject_id")
                  .apply(lambda g: pd.Series({
                      "slope": _subject_slope(g, feat),
                      "class_id": int(g["class_id"].iloc[0]),
                  }), include_groups=False)
                  .reset_index())
        dys = slopes[slopes["class_id"] == 1]["slope"].to_numpy(dtype=float)
        typ = slopes[slopes["class_id"] == 0]["slope"].to_numpy(dtype=float)
        rows.append({
            "feature": feat,
            "mean_slope_dyslexic": float(np.nanmean(dys)),
            "mean_slope_typical": float(np.nanmean(typ)),
            "cohens_d_slope": cohens_d(dys, typ),
        })
    return pd.DataFrame(rows)


# Features the brief singles out for RQ2 (fixation / regression), reported first.
RQ2_PRIORITY = ["regression_ratio", "fix_count", "mean_fix_dur", "total_read_time_ms"]


def run(cfg) -> dict[str, pd.DataFrame]:
    from eco_dysformer.features.gaze import GAZE_FEATURE_NAMES
    gaze_path = Path(cfg.paths.features_dir) / "gaze_features.csv"
    if not gaze_path.is_file():
        raise FileNotFoundError(
            f"{gaze_path} missing -- run features.assemble first."
        )
    gaze_df = pd.read_csv(gaze_path)
    features = [f for f in GAZE_FEATURE_NAMES if f in gaze_df.columns]
    # priority features first, rest after
    features = RQ2_PRIORITY + [f for f in features if f not in RQ2_PRIORITY]

    per_pass = per_passage_effects(gaze_df, features)
    grad = gradient_effects(gaze_df, features)

    out_dir = Path(cfg.paths.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_pass.to_csv(out_dir / "rq2_effects_per_passage.csv", index=False)
    grad.to_csv(out_dir / "rq2_effects_gradient.csv", index=False)
    return {"per_passage": per_pass, "gradient": grad}


if __name__ == "__main__":
    from eco_dysformer.config import load_config
    cfg = load_config()
    res = run(cfg)
    pd.set_option("display.width", 160)
    print("=== RQ2 per-passage Cohen's d (dyslexic vs typical), priority feats ===")
    pp = res["per_passage"]
    print(pp[pp["feature"].isin(RQ2_PRIORITY)].to_string(index=False))
    print("\n=== RQ2 gradient interaction: Cohen's d of per-subject slope ===")
    g = res["gradient"]
    print(g[g["feature"].isin(RQ2_PRIORITY)].to_string(index=False))
    print(f"\nwrote -> {Path(cfg.paths.results_dir) / 'rq2_effects_per_passage.csv'}")
