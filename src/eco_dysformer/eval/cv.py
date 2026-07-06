"""Subject-level nested cross-validation with hard leakage guards.

The core model unit is per-child (a child = one example = a 3-passage sequence),
so subject-level folds are natural. But the splitter is written group-aware and
re-asserts no subject leakage regardless, so it stays correct if someone runs a
per-(child, passage) variant where a subject contributes multiple rows.

Dependency-light on purpose: implemented with NumPy only (no scikit-learn), so
it runs and is unit-tested in the bare local environment. Because every subject
carries a single label, stratified *group* k-fold reduces exactly to a
stratified k-fold over subjects, which is what we implement.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class OuterFold:
    index: int
    train_idx: np.ndarray            # row indices into the design matrix
    test_idx: np.ndarray
    inner_splits: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)


def _subject_labels(groups: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return unique subjects and their (single) label; assert label consistency."""
    subjects = np.unique(groups)
    labels = np.empty(len(subjects), dtype=int)
    for i, s in enumerate(subjects):
        ys = np.unique(y[groups == s])
        assert len(ys) == 1, (
            f"subject {s} has inconsistent labels {ys}; a child must have one label"
        )
        labels[i] = int(ys[0])
    return subjects, labels


def _assign_subject_folds(subjects: np.ndarray, labels: np.ndarray,
                          n_splits: int, seed: int) -> dict:
    """Map each subject -> fold id, stratified by class, seeded and balanced."""
    rng = np.random.default_rng(seed)
    fold_of: dict = {}
    for cls in np.unique(labels):
        members = subjects[labels == cls]
        members = members[rng.permutation(len(members))]
        # Round-robin assignment keeps folds class-balanced and near-equal size.
        for i, subj in enumerate(members):
            fold_of[subj] = i % n_splits
    return fold_of


def stratified_group_kfold(groups: np.ndarray, y: np.ndarray, n_splits: int,
                           seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(train_idx, test_idx)`` row-index pairs; groups never split."""
    groups = np.asarray(groups)
    y = np.asarray(y)
    subjects, labels = _subject_labels(groups, y)
    assert n_splits >= 2, "n_splits must be >= 2"
    assert n_splits <= min(np.bincount(labels)), (
        f"n_splits={n_splits} exceeds the smallest class's subject count "
        f"({min(np.bincount(labels))}); folds would be unstratifiable"
    )
    fold_of = _assign_subject_folds(subjects, labels, n_splits, seed)
    fold_ids = np.array([fold_of[g] for g in groups])

    splits = []
    for k in range(n_splits):
        test_mask = fold_ids == k
        test_idx = np.where(test_mask)[0]
        train_idx = np.where(~test_mask)[0]
        splits.append((train_idx, test_idx))
    return splits


def nested_cv(groups: np.ndarray, y: np.ndarray, *, outer: int, inner: int,
              seed: int) -> list[OuterFold]:
    """Build a full nested CV plan (outer = performance, inner = tuning)."""
    groups = np.asarray(groups)
    y = np.asarray(y)
    outer_splits = stratified_group_kfold(groups, y, outer, seed)

    folds: list[OuterFold] = []
    for i, (tr, te) in enumerate(outer_splits):
        # Inner CV runs on the outer-train rows only. Re-index into `tr`.
        inner_splits_local = stratified_group_kfold(
            groups[tr], y[tr], inner, seed + 1 + i
        )
        inner_splits = [(tr[a], tr[b]) for a, b in inner_splits_local]
        folds.append(OuterFold(index=i, train_idx=tr, test_idx=te,
                               inner_splits=inner_splits))
    assert_no_subject_leakage(folds, groups)
    return folds


def assert_no_subject_leakage(folds: list[OuterFold], groups: np.ndarray) -> None:
    """Hard guard: no subject appears in both train and test of any split."""
    groups = np.asarray(groups)
    for f in folds:
        tr_subj = set(groups[f.train_idx].tolist())
        te_subj = set(groups[f.test_idx].tolist())
        overlap = tr_subj & te_subj
        assert not overlap, (
            f"SUBJECT LEAKAGE in outer fold {f.index}: {sorted(overlap)} in both "
            f"train and test"
        )
        for j, (itr, ival) in enumerate(f.inner_splits):
            io, iv = set(groups[itr].tolist()), set(groups[ival].tolist())
            ov = io & iv
            assert not ov, (
                f"SUBJECT LEAKAGE in outer {f.index} inner {j}: {sorted(ov)}"
            )
            # Inner rows must be a subset of outer-train rows.
            assert io <= tr_subj and iv <= tr_subj, (
                f"inner fold {j} of outer {f.index} uses subjects outside outer-train"
            )


def fold_class_balance(folds: list[OuterFold], y: np.ndarray) -> list[dict]:
    """Per outer fold: train/test class counts (for the CV report)."""
    y = np.asarray(y)
    out = []
    for f in folds:
        out.append({
            "outer_fold": f.index,
            "n_train": int(len(f.train_idx)),
            "n_test": int(len(f.test_idx)),
            "train_pos": int(y[f.train_idx].sum()),
            "test_pos": int(y[f.test_idx].sum()),
            "n_inner": len(f.inner_splits),
        })
    return out


if __name__ == "__main__":
    # Self-test on the REAL ETDD70 subjects: build nested folds, assert no
    # leakage, print class balance. Per-child unit (one row per subject).
    import sys
    from pathlib import Path
    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from eco_dysformer.config import load_config
    from eco_dysformer.data.labels import load_labels

    cfg = load_config()
    labels = load_labels(cfg.paths.labels_csv)
    groups = labels.index.to_numpy()
    y = labels["class_id"].to_numpy()
    folds = nested_cv(groups, y, outer=cfg.eval.cv.outer_folds,
                      inner=cfg.eval.cv.inner_folds, seed=cfg.seed)
    print(f"built {len(folds)} outer folds; no subject leakage (asserted).")
    for r in fold_class_balance(folds, y):
        print("  ", r)
