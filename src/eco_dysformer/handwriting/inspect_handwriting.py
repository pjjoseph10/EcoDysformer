"""Inspect the handwriting dataset BEFORE any parsing or modeling (RQ3, Stage 2).

Run this first on Kaggle after attaching the handwriting dataset. It makes NO
schema assumptions: it walks whatever is on disk and prints

  - the directory tree with per-directory file counts,
  - the file-extension histogram and total image count,
  - any metadata files (CSV/JSON/TXT/XLSX) with their columns + a head sample,
  - the label distribution (if organized by label folders such as
    Normal / Reversal / Corrected),
  - sample image filenames + one image's pixel size (if PIL is available), and
  - THE KEY QUESTION for RQ3: candidate subject/writer-linkage signals — metadata
    columns matching writer/subject keywords, and filename tokens that repeat
    across many images (a token shared by many files may encode a writer id).

It then writes a JSON report and prints a plain-language verdict on whether
per-writer linkage looks present, absent, or uncertain. This module DECIDES
nothing about modeling — it only reports, so the human can choose the RQ3 design.

    python -m eco_dysformer.handwriting.inspect_handwriting --root /kaggle/input/<slug>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

# Force UTF-8 stdout (labels/paths may contain non-latin1 glyphs on Windows).
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif", ".webp"}
META_EXTS = {".csv", ".json", ".txt", ".tsv", ".xlsx", ".xls", ".parquet"}
# Column-name substrings that would indicate per-writer / per-subject linkage.
LINKAGE_KEYWORDS = ("writer", "subject", "participant", "author", "user", "person",
                    "student", "child", "id", "name", "session")
# Keywords that specifically indicate a DIAGNOSIS label at some level.
DIAGNOSIS_KEYWORDS = ("dyslexi", "diagnos", "label", "class", "condition", "group",
                      "normal", "reversal", "corrected")

_TOKEN_SPLIT = re.compile(r"[_\-.\s]+")


def _rule(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def _walk(root: Path):
    """Return (per_dir_counts, ext_counter, image_paths, meta_paths)."""
    per_dir: dict[str, int] = {}
    ext_counter: Counter = Counter()
    image_paths: list[Path] = []
    meta_paths: list[Path] = []
    for p in root.rglob("*"):
        if p.is_dir():
            continue
        # Skip macOS archive junk and hidden AppleDouble files (never real data).
        if "__MACOSX" in p.parts or p.name.startswith("._") or p.name == ".DS_Store":
            continue
        ext = p.suffix.lower()
        ext_counter[ext] += 1
        rel_dir = str(p.parent.relative_to(root)) or "."
        per_dir[rel_dir] = per_dir.get(rel_dir, 0) + 1
        if ext in IMAGE_EXTS:
            image_paths.append(p)
        elif ext in META_EXTS:
            meta_paths.append(p)
    return per_dir, ext_counter, image_paths, meta_paths


def _print_tree(per_dir: dict[str, int]) -> None:
    for d in sorted(per_dir)[:60]:
        print(f"  {per_dir[d]:>8d}  files   {d}")
    if len(per_dir) > 60:
        print(f"  ... ({len(per_dir) - 60} more directories)")


def _inspect_metadata(meta_paths: list[Path], report: dict) -> None:
    import pandas as pd
    report["metadata_files"] = []
    for mp in meta_paths[:20]:
        entry = {"path": str(mp), "columns": None, "linkage_columns": [],
                 "diagnosis_columns": [], "note": ""}
        try:
            if mp.suffix.lower() in {".csv", ".tsv"}:
                sep = "\t" if mp.suffix.lower() == ".tsv" else ","
                df = pd.read_csv(mp, sep=sep, nrows=200)
            elif mp.suffix.lower() in {".xlsx", ".xls"}:
                df = pd.read_excel(mp, nrows=200)
            elif mp.suffix.lower() == ".parquet":
                df = pd.read_parquet(mp)
            else:
                text = mp.read_text(errors="replace")[:500]
                print(f"\n  -- {mp.name} (text, first 500 chars) --\n{text}")
                entry["note"] = "non-tabular; printed head"
                report["metadata_files"].append(entry)
                continue
            cols = [str(c) for c in df.columns]
            entry["columns"] = cols
            entry["linkage_columns"] = [c for c in cols
                                        if any(k in c.lower() for k in LINKAGE_KEYWORDS)]
            entry["diagnosis_columns"] = [c for c in cols
                                          if any(k in c.lower() for k in DIAGNOSIS_KEYWORDS)]
            print(f"\n  -- {mp.name}  shape~{df.shape} --")
            print(f"     columns: {cols}")
            if entry["linkage_columns"]:
                print(f"     *** possible WRITER/SUBJECT columns: {entry['linkage_columns']}")
                for c in entry["linkage_columns"]:
                    nun = int(df[c].nunique())
                    print(f"         '{c}': {nun} unique values in first {len(df)} rows"
                          f" (e.g. {list(df[c].dropna().unique()[:5])})")
            print(f"     head:\n{df.head(3).to_string(index=False)}")
        except Exception as e:  # never crash the whole inspection on one file
            entry["note"] = f"could not read: {e}"
            print(f"\n  -- {mp.name}: could not read ({e})")
        report["metadata_files"].append(entry)


def _filename_token_analysis(image_paths: list[Path], report: dict) -> None:
    """Look for filename tokens shared by many images (candidate writer ids)."""
    token_files: Counter = Counter()
    n = min(len(image_paths), 20000)
    for p in image_paths[:n]:
        toks = set(t for t in _TOKEN_SPLIT.split(p.stem) if t)
        for t in toks:
            token_files[t] += 1
    # Tokens shared by many files but not ALL (a label token would be in ~1/3;
    # a writer id in a moderate fraction; a per-image token in exactly 1).
    shared = [(t, c) for t, c in token_files.most_common(40) if 1 < c < n]
    report["filename_tokens_top"] = [{"token": t, "n_files": c} for t, c in shared[:20]]
    print(f"\n  sampled {n} image filenames. Tokens shared across many files")
    print(f"  (candidate label/writer keys — a writer id would group a moderate subset):")
    for t, c in shared[:20]:
        print(f"    {c:>7d}  '{t}'")
    if not shared:
        print("    (no repeated filename tokens found — names may be purely per-image)")


def _verdict(report: dict) -> str:
    has_link_col = any(m.get("linkage_columns") for m in report.get("metadata_files", []))
    if has_link_col:
        return ("LIKELY PRESENT: a metadata file has a writer/subject-like column. "
                "Confirm it groups multiple images per writer AND check whether a "
                "diagnosis label attaches at that level.")
    if report.get("metadata_files"):
        return ("UNCERTAIN: metadata files exist but no obvious writer/subject "
                "column. Inspect their columns above; linkage may be encoded in "
                "filenames or absent.")
    return ("LIKELY ABSENT (label-only): no metadata files and labels appear to be "
            "per character-image (folder-organized). Plan for an AGGREGATED "
            "reversal-rate PROXY, reported as a proxy — never subject-level diagnosis.")


def inspect(root: Path) -> dict:
    assert root.exists(), f"handwriting root not found: {root}"
    report: dict = {"root": str(root)}

    per_dir, ext_counter, image_paths, meta_paths = _walk(root)
    report["n_images"] = len(image_paths)
    report["n_metadata_files"] = len(meta_paths)
    report["extensions"] = dict(ext_counter)
    report["n_directories"] = len(per_dir)

    _rule("DIRECTORY TREE (files per directory)")
    _print_tree(per_dir)

    _rule("FILE EXTENSIONS")
    for ext, c in ext_counter.most_common():
        print(f"  {c:>8d}  {ext or '(no ext)'}")
    print(f"\n  total images: {len(image_paths)} | metadata files: {len(meta_paths)}")

    # Label distribution: top-level subdirectories that look like label folders.
    _rule("LABEL DISTRIBUTION (top-level folders)")
    top = Counter()
    for p in image_paths:
        rel = p.relative_to(root)
        top[rel.parts[0] if len(rel.parts) > 1 else "(root)"] += 1
    for k, c in top.most_common():
        print(f"  {c:>8d}  {k}")
    report["top_level_image_counts"] = dict(top)

    _rule("METADATA FILES (columns + head + linkage/diagnosis columns)")
    if meta_paths:
        _inspect_metadata(meta_paths, report)
    else:
        print("  (none found)")
        report["metadata_files"] = []

    _rule("SAMPLE IMAGE FILENAMES")
    for p in image_paths[:15]:
        print(f"  {p.relative_to(root)}")
    if image_paths:
        try:
            from PIL import Image
            with Image.open(image_paths[0]) as im:
                report["sample_image_size"] = list(im.size)
                report["sample_image_mode"] = im.mode
                print(f"\n  sample image size {im.size}, mode {im.mode}")
        except Exception as e:
            print(f"\n  (could not read sample image dimensions: {e})")

    _rule("SUBJECT/WRITER LINKAGE — FILENAME TOKEN ANALYSIS")
    _filename_token_analysis(image_paths, report)

    _rule("VERDICT: is per-writer subject linkage present?")
    verdict = _verdict(report)
    report["linkage_verdict"] = verdict
    print("  " + verdict)
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect the handwriting dataset (RQ3).")
    ap.add_argument("--root", required=True,
                    help="dataset root, e.g. /kaggle/input/dyslexia-handwriting-dataset")
    ap.add_argument("--out", default=None, help="where to write the JSON report")
    args = ap.parse_args()

    root = Path(args.root)
    report = inspect(root)

    out = Path(args.out) if args.out else (Path.cwd() / "handwriting_inspection.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _rule("RESULT")
    print(f"  report -> {out}")
    print("  Paste the sections above (esp. LABEL DISTRIBUTION, METADATA FILES, "
          "and VERDICT) back into the chat so we can choose the RQ3 design.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
