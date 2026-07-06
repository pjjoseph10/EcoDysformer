"""Load the ETDD70 dyslexia labels.

`dyslexia_class_label.csv` columns (verified): ``subject_id, class_id, label``
with ``class_id`` in {0: non-dyslexic, 1: dyslexic}. 35 subjects per class.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_labels(labels_csv: str | Path, *, n_expected: int = 70,
                n_per_class: int = 35) -> pd.DataFrame:
    """Return a labels DataFrame indexed by ``subject_id`` (int).

    Columns: ``class_id`` (0/1) and ``label`` (string). Asserts the expected
    subject counts and the binary class balance so a truncated/edited label file
    is caught immediately.
    """
    labels_csv = Path(labels_csv)
    if not labels_csv.is_file():
        raise FileNotFoundError(f"labels csv not found: {labels_csv}")

    df = pd.read_csv(labels_csv)
    expected_cols = {"subject_id", "class_id", "label"}
    missing = expected_cols - set(df.columns)
    assert not missing, f"labels csv missing columns {missing}; has {list(df.columns)}"

    df["subject_id"] = df["subject_id"].astype(int)
    df["class_id"] = df["class_id"].astype(int)

    assert df["subject_id"].is_unique, "duplicate subject_id in labels csv"
    assert len(df) == n_expected, (
        f"expected {n_expected} labelled subjects, got {len(df)}"
    )
    assert set(df["class_id"].unique()) <= {0, 1}, (
        f"class_id must be binary 0/1; got {sorted(df['class_id'].unique())}"
    )
    counts = df["class_id"].value_counts().to_dict()
    assert counts.get(0) == n_per_class and counts.get(1) == n_per_class, (
        f"expected {n_per_class}/{n_per_class} class balance; got {counts}"
    )

    return df.set_index("subject_id").sort_index()


def label_vector(labels: pd.DataFrame, subject_ids) -> "pd.Series":
    """Return the ``class_id`` (0/1) for an ordered iterable of subject_ids."""
    return labels.loc[list(subject_ids), "class_id"]
