"""Stats and metrics unit tests (bare-env runnable)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eco_dysformer.eval.metrics import (
    accuracy, auroc, brier_score, expected_calibration_error, f1_score)
from eco_dysformer.eval.stats import (
    bootstrap_ci, cohens_d, compare_arms, wilcoxon_paired)


def test_cohens_d_known_value():
    # means differ by 1 pooled sd -> d ~ 1.0
    x = np.array([1, 2, 3, 4, 5], dtype=float) + 2.0
    y = np.array([1, 2, 3, 4, 5], dtype=float)
    d = cohens_d(x, y)
    assert 1.0 < d < 1.6


def test_wilcoxon_all_zero_diff():
    r = wilcoxon_paired([0.8, 0.8, 0.8], [0.8, 0.8, 0.8])
    assert r["pvalue"] == 1.0 and "zero" in r["note"]


def test_bootstrap_ci_brackets_mean():
    vals = np.array([0.80, 0.82, 0.85, 0.83, 0.81])
    ci = bootstrap_ci(vals, n_resamples=500, seed=0)
    assert ci["lo"] <= ci["point"] <= ci["hi"]


def test_compare_arms_structure():
    a = [0.9, 0.88, 0.91, 0.89, 0.9]
    b = [0.85, 0.84, 0.86, 0.85, 0.84]
    res = compare_arms(a, b, name_a="A", name_b="B", seed=0)
    assert res["mean_a"] > res["mean_b"]
    assert "wilcoxon" in res and "paired_diff_ci" in res


def test_auroc_perfect_and_random():
    assert auroc([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) == 1.0
    assert auroc([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5]) == 0.5


def test_classification_metrics_perfect():
    yt = [1, 1, 0, 0]
    yp = [0.99, 0.95, 0.02, 0.05]
    assert accuracy(yt, yp) == 1.0
    assert f1_score(yt, yp) == 1.0
    assert brier_score(yt, yp) < 0.01
    assert 0.0 <= expected_calibration_error(yt, yp) <= 0.1


if __name__ == "__main__":
    from _run import run_module_tests
    raise SystemExit(run_module_tests(globals()))
