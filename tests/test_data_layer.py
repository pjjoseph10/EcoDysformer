"""Data-layer tests against the real ETDD70 files (bare-env runnable)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eco_dysformer.config import load_config
from eco_dysformer.data.labels import load_labels
from eco_dysformer.data.loader import ETDD70
from eco_dysformer.data.stimuli import load_all_stimuli

CFG = load_config()


def test_config_invariants():
    assert CFG.seed is not None
    assert len(CFG.dataset.passages) == 3
    assert CFG.eval.cv.group_key == "subject_id"
    assert CFG.explain.on_features == "original"


def test_labels_balance():
    labels = load_labels(CFG.paths.labels_csv)
    assert len(labels) == 70
    counts = labels["class_id"].value_counts().to_dict()
    assert counts[0] == 35 and counts[1] == 35


def test_loader_grid_complete():
    ds = ETDD70(CFG)
    assert len(ds.subjects()) == 70
    assert len(ds.manifest) == 70 * 3 * 4          # subjects x passages x types
    # every cell has all four file types
    grid = ds.manifest.groupby(["subject_id", "passage_name"])["file_type"].nunique()
    assert (grid == 4).all()


def test_labels_match_data():
    ds = ETDD70(CFG)
    labels = load_labels(CFG.paths.labels_csv)
    assert set(ds.subjects()) == set(labels.index.tolist())


def test_stimulus_text_reconstruction():
    stimuli = load_all_stimuli(CFG)
    assert set(stimuli) == {"syllables", "meaningful", "pseudo"}
    # meaningful passage is real Czech words -> known opening tokens
    assert stimuli["meaningful"].tokens[:2] == ["Malý", "Pepík"]
    for st in stimuli.values():
        assert st.n_tokens > 0 and st.text.strip()


if __name__ == "__main__":
    from _run import run_module_tests
    raise SystemExit(run_module_tests(globals()))
