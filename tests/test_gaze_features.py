"""Gaze feature + cross-check tests against real data (bare-env runnable)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eco_dysformer.config import load_config
from eco_dysformer.data.loader import ETDD70
from eco_dysformer.features.gaze import (
    GAZE_FEATURE_NAMES, compute_gaze_features, crosscheck_against_metrics)

CFG = load_config()
DS = ETDD70(CFG)


def test_feature_schema_complete():
    sid = DS.subjects()[0]
    feats = compute_gaze_features(DS.load(sid, "syllables", "fixations"),
                                  DS.load(sid, "syllables", "saccades"))
    assert list(feats.keys()) == GAZE_FEATURE_NAMES
    assert feats["fix_count"] > 0


def test_crosscheck_all_strict_pass():
    """Every strict recomputed-vs-metrics comparison must pass on a sample."""
    for sid in DS.subjects()[:8]:
        for passage in DS.passage_names:
            feats = compute_gaze_features(DS.load(sid, passage, "fixations"),
                                          DS.load(sid, passage, "saccades"))
            for c in crosscheck_against_metrics(feats, DS.load(sid, passage, "metrics")):
                if c["strict"]:
                    assert c["pass"] is True, (sid, passage, c)


def test_regression_uses_signed_direction():
    """With signed coordinates, at least some subjects show regressions (>0)."""
    total = 0
    for sid in DS.subjects()[:10]:
        f = compute_gaze_features(DS.load(sid, "meaningful", "fixations"),
                                  DS.load(sid, "meaningful", "saccades"))
        assert 0.0 <= f["regression_ratio"] <= 1.0
        total += f["regression_count"]
    assert total > 0, "signed-direction regressions should not be all zero"


if __name__ == "__main__":
    from _run import run_module_tests
    raise SystemExit(run_module_tests(globals()))
