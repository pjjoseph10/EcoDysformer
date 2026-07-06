"""Build per-child 3-passage arrays and fold-safe standardization.

The core model unit is a child: one example = a length-3 sequence of passage
tokens ordered by complexity_rank (syllables -> meaningful -> pseudo). This module
pivots the long feature table into::

    X_gaze : (N_children, 3, F_gaze)
    X_ling : (N_children, 3, F_ling)   (or None for the complexity-blind arm)
    y      : (N_children,)             per-child label
    subjects : (N_children,)           subject_id, aligned to rows

Standardization is fit on TRAIN rows only (``ArrayScaler``) to avoid leakage, and
flat interpretable feature names ("passage__feature") are exposed for LIME.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

RANK_ORDER = [0, 1, 2]


@dataclass
class ChildArrays:
    subjects: np.ndarray            # (N,)
    X_gaze: np.ndarray              # (N, 3, F_gaze)
    X_ling: np.ndarray | None       # (N, 3, F_ling) or None
    y: np.ndarray                   # (N,)
    passage_names: list[str]        # ordered by complexity_rank
    gaze_cols: list[str]
    ling_cols: list[str]

    def flat_feature_names(self) -> list[str]:
        """Interpretable names for the flattened (LIME) view: passage__feature."""
        names = [f"{p}__{c}" for p in self.passage_names for c in self.gaze_cols]
        if self.X_ling is not None:
            names += [f"{p}__{c}" for p in self.passage_names for c in self.ling_cols]
        return names

    def flatten(self) -> np.ndarray:
        """(N, 3*F_gaze [+ 3*F_ling]) flat interpretable matrix for LIME."""
        parts = [self.X_gaze.reshape(len(self.subjects), -1)]
        if self.X_ling is not None:
            parts.append(self.X_ling.reshape(len(self.subjects), -1))
        return np.concatenate(parts, axis=1)


def build_child_arrays(df: pd.DataFrame, gaze_cols: list[str],
                       ling_cols: list[str] | None = None) -> ChildArrays:
    df = df.sort_values(["subject_id", "complexity_rank"])
    subjects = np.array(sorted(df["subject_id"].unique()))

    # Passage names in complexity order (consistent across subjects).
    order = (df.drop_duplicates("complexity_rank")
             .sort_values("complexity_rank"))
    passage_names = order["passage_name"].tolist()
    assert order["complexity_rank"].tolist() == RANK_ORDER, (
        f"expected complexity_rank {RANK_ORDER}, got {order['complexity_rank'].tolist()}"
    )

    Xg, Xl, y = [], [], []
    for sid in subjects:
        rows = df[df["subject_id"] == sid].sort_values("complexity_rank")
        assert len(rows) == 3, f"subject {sid} must have 3 passages, got {len(rows)}"
        assert rows["complexity_rank"].tolist() == RANK_ORDER
        Xg.append(rows[gaze_cols].to_numpy(dtype=float))
        if ling_cols:
            Xl.append(rows[ling_cols].to_numpy(dtype=float))
        y.append(int(rows["class_id"].iloc[0]))

    return ChildArrays(
        subjects=subjects,
        X_gaze=np.stack(Xg),
        X_ling=np.stack(Xl) if ling_cols else None,
        y=np.array(y, dtype=int),
        passage_names=passage_names,
        gaze_cols=list(gaze_cols),
        ling_cols=list(ling_cols) if ling_cols else [],
    )


class ArrayScaler:
    """Per-feature standardizer for (N, P, F) arrays, fit on TRAIN only.

    Statistics are computed over the (N_train x P) pooled rows for each feature;
    NaNs are imputed with the train mean before scaling.
    """

    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "ArrayScaler":
        flat = X.reshape(-1, X.shape[-1])
        self.mean_ = np.nanmean(flat, axis=0)
        std = np.nanstd(flat, axis=0)
        std[std == 0] = 1.0
        self.std_ = std
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None, "ArrayScaler not fit"
        Xc = X.copy()
        # impute NaNs with train mean
        inds = np.where(np.isnan(Xc))
        if inds[0].size:
            Xc[inds] = np.take(self.mean_, inds[-1])
        return (Xc - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


if __name__ == "__main__":
    # Local test: build gaze-only child arrays from the recomputed gaze table.
    import sys
    from pathlib import Path
    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from eco_dysformer.config import load_config
    from eco_dysformer.features.gaze import GAZE_FEATURE_NAMES

    cfg = load_config()
    gaze_df = pd.read_csv(Path(cfg.paths.features_dir) / "gaze_features.csv")
    arrays = build_child_arrays(gaze_df, GAZE_FEATURE_NAMES, ling_cols=None)
    print("subjects:", arrays.subjects.shape, "X_gaze:", arrays.X_gaze.shape,
          "y:", arrays.y.shape, "y_pos:", int(arrays.y.sum()))
    print("passage order:", arrays.passage_names)
    sc = ArrayScaler().fit(arrays.X_gaze[:56])
    Xt = sc.transform(arrays.X_gaze)
    print("scaled mean~0:", float(np.round(Xt[:56].reshape(-1, Xt.shape[-1]).mean(), 6)),
          "flat dims:", arrays.flatten().shape)
