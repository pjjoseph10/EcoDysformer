"""Per-child array builder + fold-safe scaler tests (bare-env runnable)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eco_dysformer.config import load_config
from eco_dysformer.data.tensors import ArrayScaler, build_child_arrays
from eco_dysformer.features.gaze import GAZE_FEATURE_NAMES

CFG = load_config()
GAZE_CSV = Path(CFG.paths.features_dir) / "gaze_features.csv"


def _load():
    assert GAZE_CSV.is_file(), "run features.assemble first to create gaze_features.csv"
    return pd.read_csv(GAZE_CSV)


def test_child_array_shapes():
    arr = build_child_arrays(_load(), GAZE_FEATURE_NAMES, ling_cols=None)
    assert arr.X_gaze.shape == (70, 3, len(GAZE_FEATURE_NAMES))
    assert arr.y.shape == (70,) and int(arr.y.sum()) == 35
    assert arr.passage_names == ["syllables", "meaningful", "pseudo"]


def test_flatten_and_names_align():
    arr = build_child_arrays(_load(), GAZE_FEATURE_NAMES, ling_cols=None)
    flat = arr.flatten()
    names = arr.flat_feature_names()
    assert flat.shape == (70, 3 * len(GAZE_FEATURE_NAMES))
    assert len(names) == flat.shape[1]
    assert names[0] == f"syllables__{GAZE_FEATURE_NAMES[0]}"


def test_scaler_fits_on_train_only():
    arr = build_child_arrays(_load(), GAZE_FEATURE_NAMES, ling_cols=None)
    train = arr.X_gaze[:56]
    sc = ArrayScaler().fit(train)
    zt = sc.transform(train).reshape(-1, train.shape[-1])
    assert np.allclose(zt.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(zt.std(axis=0), 1, atol=1e-3)


if __name__ == "__main__":
    from _run import run_module_tests
    raise SystemExit(run_module_tests(globals()))
