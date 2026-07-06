"""The most important correctness test: no subject leakage across folds."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eco_dysformer.config import load_config
from eco_dysformer.data.labels import load_labels
from eco_dysformer.eval.cv import (
    assert_no_subject_leakage, nested_cv, stratified_group_kfold)

CFG = load_config()
LABELS = load_labels(CFG.paths.labels_csv)
GROUPS = LABELS.index.to_numpy()
Y = LABELS["class_id"].to_numpy()


def test_outer_folds_partition_all_subjects():
    splits = stratified_group_kfold(GROUPS, Y, CFG.eval.cv.outer_folds, CFG.seed)
    covered = np.concatenate([te for _, te in splits])
    assert sorted(covered.tolist()) == list(range(len(GROUPS)))  # exact partition


def test_no_subject_leakage_nested():
    folds = nested_cv(GROUPS, Y, outer=CFG.eval.cv.outer_folds,
                      inner=CFG.eval.cv.inner_folds, seed=CFG.seed)
    # explicit re-assert (nested_cv already asserts internally)
    assert_no_subject_leakage(folds, GROUPS)
    for f in folds:
        tr = set(GROUPS[f.train_idx].tolist())
        te = set(GROUPS[f.test_idx].tolist())
        assert tr.isdisjoint(te)
        for itr, iva in f.inner_splits:
            assert set(GROUPS[itr]).isdisjoint(set(GROUPS[iva]))
            assert set(GROUPS[itr]) <= tr and set(GROUPS[iva]) <= tr


def test_folds_are_class_stratified():
    folds = nested_cv(GROUPS, Y, outer=CFG.eval.cv.outer_folds,
                      inner=CFG.eval.cv.inner_folds, seed=CFG.seed)
    for f in folds:
        # 35/35 across 5 folds -> each test fold should hold ~7 positives
        assert 5 <= int(Y[f.test_idx].sum()) <= 9


def test_leakage_guard_catches_injected_leak():
    """A deliberately corrupted split must trip the guard."""
    folds = nested_cv(GROUPS, Y, outer=CFG.eval.cv.outer_folds,
                      inner=CFG.eval.cv.inner_folds, seed=CFG.seed)
    bad = folds[0]
    bad.test_idx = np.concatenate([bad.test_idx, bad.train_idx[:1]])  # inject overlap
    try:
        assert_no_subject_leakage([bad], GROUPS)
    except AssertionError:
        return
    raise AssertionError("leakage guard failed to catch an injected leak")


if __name__ == "__main__":
    from _run import run_module_tests
    raise SystemExit(run_module_tests(globals()))
