"""Inspect the ETDD70 dataset BEFORE any feature parsing.

Run this first. It prints the file tree summary, a sample of every file type,
the reconstructed Czech stimulus text for each passage, and the label balance,
then runs the loader's structural asserts. A machine-readable report is written
to ``outputs/results/dataset_inspection.json`` so the inspection itself is a
reproducible artifact and not just stdout.

    python -m eco_dysformer.data.inspect_dataset
    # or:  python src/eco_dysformer/data/inspect_dataset.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running as a plain script (adds <repo>/src to sys.path).
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Czech stimulus text contains non-latin1 glyphs; force UTF-8 stdout so printing
# works on a Windows cp1252 console.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

from eco_dysformer.config import load_config
from eco_dysformer.data.labels import load_labels
from eco_dysformer.data.loader import ETDD70
from eco_dysformer.data.stimuli import load_all_stimuli


def _rule(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def inspect(cfg=None) -> dict:
    cfg = cfg or load_config()
    report: dict = {"ok": False, "checks": {}}

    _rule("PATHS")
    for k in cfg.paths.keys():
        p = Path(cfg.paths[k])
        exists = p.exists()
        print(f"  {k:14s} {'OK ' if exists else 'MISSING'}  {p}")
        report["checks"][f"path_{k}_exists"] = exists

    # ---- structural load (runs all grid asserts) ----
    _rule("STRUCTURE (loader asserts)")
    ds = ETDD70(cfg)
    subjects = ds.subjects()
    print(f"  subjects discovered : {len(subjects)}")
    print(f"  passages            : {ds.passage_names}")
    print(f"  file types          : {ds.file_types}")
    print(f"  manifest rows       : {len(ds.manifest)} "
          f"(expect {cfg.dataset.n_subjects_expected} x "
          f"{len(ds.passage_names)} x {len(ds.file_types)} = "
          f"{cfg.dataset.n_subjects_expected * len(ds.passage_names) * len(ds.file_types)})")
    report["n_subjects"] = len(subjects)
    report["n_manifest_rows"] = int(len(ds.manifest))
    report["passages"] = ds.passage_names

    # ---- labels ----
    _rule("LABELS")
    labels = load_labels(
        cfg.paths.labels_csv,
        n_expected=cfg.dataset.n_subjects_expected,
        n_per_class=cfg.dataset.n_per_class_expected,
    )
    dist = labels["label"].value_counts().to_dict()
    print(f"  class distribution  : {dist}")
    # Every labelled subject must have data files, and vice versa.
    label_ids = set(labels.index.tolist())
    data_ids = set(subjects)
    only_labels = sorted(label_ids - data_ids)
    only_data = sorted(data_ids - label_ids)
    print(f"  labelled w/o data   : {only_labels or 'none'}")
    print(f"  data w/o label      : {only_data or 'none'}")
    assert not only_labels and not only_data, (
        "mismatch between labelled subjects and subjects with data files"
    )
    report["label_distribution"] = dist
    report["checks"]["labels_match_data"] = True

    # ---- samples of each file type ----
    _rule("SAMPLE FILES (first subject/passage, head)")
    sid0 = subjects[0]
    passage0 = ds.passage_names[0]
    samples = {}
    for ftype in ds.file_types:
        df = ds.load(sid0, passage0, ftype)
        print(f"\n  -- Subject {sid0} / {passage0} / {ftype}  "
              f"shape={df.shape}")
        print(f"     columns: {list(df.columns)}")
        samples[ftype] = {"shape": list(df.shape), "columns": list(df.columns)}
    report["sample_schemas"] = samples

    # ---- reconstructed stimulus text ----
    _rule("STIMULUS TEXT (reconstructed from ROI content)")
    stimuli = load_all_stimuli(cfg)
    report["stimuli"] = {}
    for name, st in stimuli.items():
        s = st.summary()
        print(f"\n  [{name}] {st.n_tokens} tokens, {st.n_lines} lines")
        print(f"     {s['text_preview']}")
        report["stimuli"][name] = s

    report["ok"] = True
    return report, cfg


def main() -> int:
    report, cfg = inspect()

    out_dir = Path(cfg.paths.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dataset_inspection.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    _rule("RESULT")
    print(f"  inspection OK -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
