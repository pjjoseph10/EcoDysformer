"""Assemble feature tables from the ETDD70 data.

Produces (written under ``config.paths.features_dir``):
    gaze_features.csv        210 rows = (subject x passage), recomputed gaze
                             features + dataset regression alternatives + label
    gaze_crosscheck.csv      recomputed-vs-metrics.csv comparison, all cells
    linguistic_features.csv  3 rows = per passage (Kaggle-first; needs engine)
    features_long.csv        gaze joined with linguistic on passage -> master
                             modeling table (210 rows), one row per (child,passage)

The gaze half runs in the bare local environment; the linguistic half requires
the Czech NLP engine and is therefore Kaggle-first. ``assemble_features`` builds
gaze unconditionally and linguistic only when ``include_linguistic=True``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from eco_dysformer.data.labels import load_labels
from eco_dysformer.data.loader import ETDD70
from eco_dysformer.data.stimuli import load_all_stimuli
from eco_dysformer.features.gaze import (
    GAZE_FEATURE_NAMES,
    compute_gaze_features,
    crosscheck_against_metrics,
)
from eco_dysformer.features.linguistic import (
    LINGUISTIC_FEATURE_NAMES,
    extract_linguistic_features,
)

ID_COLS = ["subject_id", "passage_name", "complexity_rank", "class_id", "label"]


def _dataset_regression_alts(metrics: pd.DataFrame, sacc_count: float) -> dict[str, float]:
    """Carry the dataset's published (AOI-based) regression as alt columns."""
    def scalar(col: str) -> float:
        if col not in metrics.columns or len(metrics) == 0:
            return float("nan")
        v = pd.to_numeric(metrics[col], errors="coerce").dropna().unique()
        return float(v[0]) if len(v) else float("nan")

    n_reg = scalar("n_regress_trial")
    ratio = (n_reg / sacc_count) if (sacc_count and sacc_count > 0) else float("nan")
    return {
        "regression_count_dataset": n_reg,
        "regression_ratio_dataset": ratio,
        "n_within_line_regress_dataset": scalar("n_within_line_regress_trial"),
        "n_between_line_regress_dataset": scalar("n_between_line_regress_trial"),
    }


def build_gaze_table(cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(gaze_features_df, crosscheck_df)`` for all 210 cells."""
    ds = ETDD70(cfg)
    labels = load_labels(
        cfg.paths.labels_csv,
        n_expected=cfg.dataset.n_subjects_expected,
        n_per_class=cfg.dataset.n_per_class_expected,
    )
    rank = {p["name"]: p["complexity_rank"] for p in cfg.dataset.passages}
    tol = float(cfg.features.gaze.crosscheck_tolerance)

    rows, checks = [], []
    for sid, passage in ds.iter_cells():
        fx = ds.load(sid, passage, "fixations")
        sc = ds.load(sid, passage, "saccades")
        mt = ds.load(sid, passage, "metrics")

        feats = compute_gaze_features(fx, sc)
        alts = _dataset_regression_alts(mt, feats["sacc_count"])

        for c in crosscheck_against_metrics(feats, mt, tolerance=tol):
            checks.append({"subject_id": sid, "passage_name": passage, **c})

        rows.append({
            "subject_id": sid,
            "passage_name": passage,
            "complexity_rank": rank[passage],
            "class_id": int(labels.loc[sid, "class_id"]),
            "label": labels.loc[sid, "label"],
            **feats,
            **alts,
        })

    gaze_df = pd.DataFrame(rows).sort_values(["subject_id", "complexity_rank"])
    check_df = pd.DataFrame(checks)
    _report_nans(gaze_df, GAZE_FEATURE_NAMES, "gaze")
    return gaze_df.reset_index(drop=True), check_df


def build_linguistic_table(cfg) -> pd.DataFrame:
    """Return per-passage linguistic features (3 rows). Requires the NLP engine."""
    stimuli = load_all_stimuli(cfg)
    rank = {p["name"]: p["complexity_rank"] for p in cfg.dataset.passages}

    use_emb = bool(cfg.features.linguistic.use_embedding)
    rows = []
    for name, st in stimuli.items():
        feats = extract_linguistic_features(st.text, cfg)
        row = {"passage_name": name, "complexity_rank": rank[name], **feats}
        if use_emb:
            from eco_dysformer.features.embeddings import extract_embedding
            row.update(extract_embedding(st.text, cfg))
        rows.append(row)
    ling_df = pd.DataFrame(rows).sort_values("complexity_rank").reset_index(drop=True)
    _report_nans(ling_df, LINGUISTIC_FEATURE_NAMES, "linguistic")
    return ling_df


def _report_nans(df: pd.DataFrame, cols: list[str], tag: str) -> None:
    present = [c for c in cols if c in df.columns]
    n_nan = int(df[present].isna().sum().sum())
    if n_nan:
        bad = df[present].isna().sum()
        print(f"[assemble] WARNING: {n_nan} NaN(s) in {tag} features: "
              f"{bad[bad > 0].to_dict()}")


def assemble_features(cfg, *, include_linguistic: bool = True) -> dict[str, pd.DataFrame]:
    """Build, write, and return the feature tables.

    Parameters
    ----------
    include_linguistic
        When False (bare local env), only the gaze table + cross-check are built
        and written; the linguistic/merged tables are skipped with a printed note.
    """
    out_dir = Path(cfg.paths.features_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gaze_df, check_df = build_gaze_table(cfg)
    gaze_df.to_csv(out_dir / "gaze_features.csv", index=False)
    check_df.to_csv(out_dir / "gaze_crosscheck.csv", index=False)
    tables = {"gaze": gaze_df, "crosscheck": check_df}

    # Cross-check summary: how many strict comparisons passed.
    strict = check_df[check_df["strict"] == True]  # noqa: E712
    n_pass = int((strict["pass"] == True).sum())  # noqa: E712
    print(f"[assemble] gaze cross-check: {n_pass}/{len(strict)} strict comparisons "
          f"within tolerance ({cfg.features.gaze.crosscheck_tolerance}).")

    if not include_linguistic:
        print("[assemble] linguistic features SKIPPED (include_linguistic=False). "
              "Run on Kaggle with the Czech NLP engine to produce "
              "linguistic_features.csv and features_long.csv.")
        return tables

    ling_df = build_linguistic_table(cfg)
    ling_df.to_csv(out_dir / "linguistic_features.csv", index=False)
    tables["linguistic"] = ling_df

    # Merge gaze x linguistic on passage -> master modeling table.
    long_df = gaze_df.merge(
        ling_df.drop(columns=["complexity_rank"]), on="passage_name", how="left"
    )
    assert len(long_df) == len(gaze_df), "linguistic merge changed row count"
    long_df.to_csv(out_dir / "features_long.csv", index=False)
    tables["long"] = long_df
    print(f"[assemble] wrote features_long.csv: {long_df.shape}")
    return tables


if __name__ == "__main__":
    import sys
    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from eco_dysformer.config import load_config

    _cfg = load_config()
    # Local default: gaze only (linguistic needs the Kaggle NLP engine).
    assemble_features(_cfg, include_linguistic=False)
