"""Comparison table against PUBLISHED ETDD70 baselines.

Cross-paper numbers use DIFFERENT validation protocols and are APPROXIMATE; the
table marks them as such and keeps a separate, clearly-labeled section for THIS
project's own results under its own subject-level nested-CV protocol. Our rows
are filled from ``results_dir/cv_results.json`` when present, and otherwise show
``pending`` -- never a fabricated number.

Pure pandas; runs locally. Writes ``baseline_comparison.csv`` and ``.md``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _our_results(results_dir: Path) -> list[dict]:
    """Read our own arm accuracies from cv_results.json, if it exists."""
    path = results_dir / "cv_results.json"
    rows = []
    if path.is_file():
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for arm_name, arm in data.get("arms", {}).items():
            acc = arm.get("accuracy_mean")
            ci = arm.get("accuracy_ci", {})
            rows.append({
                "system": f"This work — {arm_name}",
                "accuracy": round(acc, 4) if acc is not None else "pending",
                "accuracy_ci": (f"[{ci.get('lo'):.3f}, {ci.get('hi'):.3f}]"
                                if ci.get("lo") is not None else ""),
                "validation": "subject-level nested CV (this protocol)",
                "note": "our protocol; directly comparable across our own arms",
            })
    if not rows:
        rows.append({
            "system": "This work — (Performer / quadratic / blind)",
            "accuracy": "pending",
            "accuracy_ci": "",
            "validation": "subject-level nested CV (this protocol)",
            "note": "run run_stage1 on Kaggle to populate",
        })
    return rows


def build_table(cfg) -> pd.DataFrame:
    published = [
        {
            "system": b["name"],
            "accuracy": b["accuracy"],
            "accuracy_ci": "",
            "validation": "cross-paper (protocol differs)",
            "note": b.get("note", "approximate"),
        }
        for b in cfg.baselines_published
    ]
    ours = _our_results(Path(cfg.paths.results_dir))
    df = pd.DataFrame(published + ours)
    return df


def save_table(cfg, df: pd.DataFrame) -> dict:
    res_dir = Path(cfg.paths.results_dir)
    res_dir.mkdir(parents=True, exist_ok=True)
    csv_path = res_dir / "baseline_comparison.csv"
    md_path = res_dir / "baseline_comparison.md"
    df.to_csv(csv_path, index=False)

    lines = [
        "# ETDD70 baseline comparison",
        "",
        "> **Cross-paper numbers use different validation protocols and are "
        "approximate.** They are NOT directly comparable to this project's "
        "subject-level nested-CV results. The only strictly comparable numbers "
        "are among *This work* rows, which share one protocol.",
        "",
        df.to_markdown(index=False),
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"csv": str(csv_path), "md": str(md_path)}


if __name__ == "__main__":
    import sys
    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from eco_dysformer.config import load_config

    cfg = load_config()
    df = build_table(cfg)
    info = save_table(cfg, df)
    print(df.to_string(index=False))
    print("\nwrote:", info)
