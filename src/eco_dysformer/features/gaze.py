"""Per-child, per-passage gaze feature engineering (recompute from events).

Features are computed directly from the ``*_fixations.csv`` and ``*_saccades.csv``
event files. The four features named in the brief -- fixation count, mean
fixation duration, regression ratio, total read time -- are computed here
alongside a set of literature-standard fixation/saccade descriptors that the raw
data supports (durations, amplitudes, velocities, regression sub-types, reading
rate).

Cross-check: every recomputed feature that has a direct analogue in the provided
``*_metrics.csv`` trial aggregates is compared against it (see
:func:`crosscheck_against_metrics`), so a wrong definition surfaces as a large
relative discrepancy in the feature report instead of silently corrupting the
model.

REGRESSION DEFINITION (a modeling choice, documented and testable):
    NOTE: the ``ampl_x``/``ampl_y`` columns are stored as MAGNITUDES (always >= 0;
    verified corr(ampl_x, |end_x-start_x|) == 1.0), so signed direction is
    derived from the coordinate columns: dx = end_x - start_x, dy = end_y -
    start_y. y grows downward, so reading advances with +dx (same line) or +dy
    (next line). We label each saccade:
        progressive       : dx > +dead_x and |dy| < line_tol_y
        within-line regr.  : dx < -dead_x and |dy| < line_tol_y
        return sweep       : dx < 0 and dy > +line_tol_y   (forward, line change)
        between-line regr. : dy < -line_tol_y              (look back to prev line)
    regression_count = within-line + between-line; regression_ratio =
    regression_count / n_saccades.

    This transparent geometric definition OVER-COUNTS relative to the dataset's
    published I2MC/AOI-based regression counts (which are within-line only). That
    gap is real and expected -- the cross-check reports it, and assemble.py also
    carries the dataset's ``n_regress_trial`` as ``regression_*_dataset`` columns
    so downstream can use either. dead_x and line_tol_y are parameters; TODO:
    a sensitivity sweep over them is deferred (low priority for Stage 1).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# The canonical ordered list of gaze feature names this module produces. Kept
# explicit so downstream code (feature tables, LIME) has a stable schema.
GAZE_FEATURE_NAMES: list[str] = [
    # --- the four named in the brief ---
    "fix_count",
    "mean_fix_dur",
    "regression_ratio",
    "total_read_time_ms",
    # --- extended fixation descriptors ---
    "std_fix_dur",
    "median_fix_dur",
    "total_fix_dur_ms",
    "fixation_rate_hz",          # fixations per second of reading
    "mean_fix_disp",             # mean fixation dispersion (spatial spread)
    # --- saccade descriptors ---
    "sacc_count",
    "mean_sacc_ampl",
    "std_sacc_ampl",
    "mean_sacc_dur",
    "mean_sacc_peak_vel",
    "mean_sacc_avg_vel",
    # --- regression / reading-flow descriptors ---
    "regression_count",
    "within_line_regression_count",
    "between_line_regression_count",
    "return_sweep_count",
    "progressive_count",
]


@dataclass
class RegressionParams:
    dead_x: float = 3.0        # px; ignore near-zero horizontal moves (microsaccades)
    line_tol_y: float = 15.0   # px; |ampl_y| below this = same line


def _safe_mean(x: pd.Series) -> float:
    return float(x.mean()) if len(x) else float("nan")


def _safe_std(x: pd.Series) -> float:
    return float(x.std(ddof=1)) if len(x) > 1 else 0.0


def compute_gaze_features(
    fixations: pd.DataFrame,
    saccades: pd.DataFrame,
    reg: RegressionParams | None = None,
) -> dict[str, float]:
    """Compute the gaze feature dict for one (subject, passage) trial.

    Assumes a single trial per file (asserted). Returns a dict keyed by
    :data:`GAZE_FEATURE_NAMES`.
    """
    reg = reg or RegressionParams()

    # --- structural sanity: one trial per file ---
    for name, df in (("fixations", fixations), ("saccades", saccades)):
        if "trialid" in df.columns and len(df) > 0:
            n_tr = df["trialid"].nunique()
            assert n_tr == 1, f"{name}: expected 1 trial per file, found {n_tr}"

    feat: dict[str, float] = {}

    # ---------------- fixations ----------------
    dur = pd.to_numeric(fixations["duration_ms"], errors="coerce").dropna()
    feat["fix_count"] = float(len(fixations))
    feat["mean_fix_dur"] = _safe_mean(dur)
    feat["std_fix_dur"] = _safe_std(dur)
    feat["median_fix_dur"] = float(dur.median()) if len(dur) else float("nan")
    feat["total_fix_dur_ms"] = float(dur.sum())

    if len(fixations) and {"start_ms", "end_ms"} <= set(fixations.columns):
        t0 = pd.to_numeric(fixations["start_ms"], errors="coerce").min()
        t1 = pd.to_numeric(fixations["end_ms"], errors="coerce").max()
        read_time = float(t1 - t0)
    else:
        read_time = float("nan")
    feat["total_read_time_ms"] = read_time
    feat["fixation_rate_hz"] = (
        float(len(fixations) / (read_time / 1000.0)) if read_time and read_time > 0
        else float("nan")
    )

    # fixation dispersion (spatial spread of each fixation, if provided)
    if {"disp_x", "disp_y"} <= set(fixations.columns) and len(fixations):
        dx = pd.to_numeric(fixations["disp_x"], errors="coerce")
        dy = pd.to_numeric(fixations["disp_y"], errors="coerce")
        feat["mean_fix_disp"] = float(np.hypot(dx, dy).mean())
    else:
        feat["mean_fix_disp"] = float("nan")

    # ---------------- saccades ----------------
    n_sacc = len(saccades)
    feat["sacc_count"] = float(n_sacc)
    if n_sacc:
        ampl = pd.to_numeric(saccades.get("ampl"), errors="coerce")
        feat["mean_sacc_ampl"] = _safe_mean(ampl)
        feat["std_sacc_ampl"] = _safe_std(ampl)
        feat["mean_sacc_dur"] = _safe_mean(
            pd.to_numeric(saccades["duration_ms"], errors="coerce"))
        feat["mean_sacc_peak_vel"] = _safe_mean(
            pd.to_numeric(saccades.get("peak_vel"), errors="coerce"))
        feat["mean_sacc_avg_vel"] = _safe_mean(
            pd.to_numeric(saccades.get("avg_vel"), errors="coerce"))

        # ampl_x/ampl_y are magnitudes; derive SIGNED direction from coordinates.
        dx = (pd.to_numeric(saccades["end_x"], errors="coerce")
              - pd.to_numeric(saccades["start_x"], errors="coerce")).fillna(0.0)
        dy = (pd.to_numeric(saccades["end_y"], errors="coerce")
              - pd.to_numeric(saccades["start_y"], errors="coerce")).fillna(0.0)
        same_line = dy.abs() < reg.line_tol_y
        progressive = (dx > reg.dead_x) & same_line
        within_regr = (dx < -reg.dead_x) & same_line
        return_sweep = (dx < 0) & (dy > reg.line_tol_y)
        between_regr = dy < -reg.line_tol_y

        feat["progressive_count"] = float(int(progressive.sum()))
        feat["within_line_regression_count"] = float(int(within_regr.sum()))
        feat["between_line_regression_count"] = float(int(between_regr.sum()))
        feat["return_sweep_count"] = float(int(return_sweep.sum()))
        reg_count = int(within_regr.sum()) + int(between_regr.sum())
        feat["regression_count"] = float(reg_count)
        feat["regression_ratio"] = float(reg_count / n_sacc)
    else:
        for k in ("mean_sacc_ampl", "std_sacc_ampl", "mean_sacc_dur",
                  "mean_sacc_peak_vel", "mean_sacc_avg_vel"):
            feat[k] = float("nan")
        for k in ("progressive_count", "within_line_regression_count",
                  "between_line_regression_count", "return_sweep_count",
                  "regression_count"):
            feat[k] = 0.0
        feat["regression_ratio"] = float("nan")

    # Guarantee the full schema is present and ordered.
    return {k: feat.get(k, float("nan")) for k in GAZE_FEATURE_NAMES}


def _trial_scalar(metrics: pd.DataFrame, col: str) -> float:
    """Pull a trial-level scalar from the (ROI-expanded) metrics table."""
    if col not in metrics.columns or len(metrics) == 0:
        return float("nan")
    vals = pd.to_numeric(metrics[col], errors="coerce").dropna().unique()
    return float(vals[0]) if len(vals) else float("nan")


def crosscheck_against_metrics(
    recomputed: dict[str, float], metrics: pd.DataFrame, tolerance: float = 0.02,
) -> list[dict]:
    """Compare recomputed features to dataset-provided trial aggregates.

    Returns one record per compared quantity with the two values, relative
    difference, and a pass/fail flag at ``tolerance``. Regression ratio uses a
    different definition in the dataset (progress:regress ratio, not
    regressions/saccades), so it is reported as INFO (never failed).
    """
    checks = [
        # (label, recomputed_key, metrics_col, definitional_match)
        ("fix_count", "fix_count", "n_fix_trial", True),
        ("mean_fix_dur", "mean_fix_dur", "mean_fix_dur_trial", True),
        ("total_fix_dur", "total_fix_dur_ms", "sum_fix_dur_trial", True),
        ("sacc_count", "sacc_count", "n_sacc_trial", True),
        ("mean_sacc_ampl", "mean_sacc_ampl", "mean_sacc_ampl_trial", True),
        ("mean_sacc_dur", "mean_sacc_dur", "mean_sacc_dur_trial", True),
        ("regression_count", "regression_count", "n_regress_trial", False),
    ]
    out = []
    for label, rk, mc, strict in checks:
        rv = recomputed.get(rk, float("nan"))
        mv = _trial_scalar(metrics, mc)
        denom = abs(mv) if (mv == mv and mv != 0) else float("nan")
        rel = abs(rv - mv) / denom if denom == denom else float("nan")
        out.append({
            "quantity": label,
            "recomputed": rv,
            "dataset_metrics": mv,
            "rel_diff": rel,
            "strict": strict,
            "pass": bool(rel <= tolerance) if (strict and rel == rel) else None,
        })
    return out
