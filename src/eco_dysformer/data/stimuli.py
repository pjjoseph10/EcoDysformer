"""Reconstruct each passage's Czech stimulus text from the ROI files.

There is no plain-text transcript in ETDD70 -- only stimulus images plus ROI
CSVs (inside ``rois.zip``) that annotate word/line boundaries. Crucially, each
sub-line ROI row carries the token in its ``content`` column, so the full
passage text can be reconstructed in reading order (line, then part).

ROI columns (verified): ``id, stimfile, content, kind, name, x, y, width,
height, line, part, column``. ``kind`` is 'line' (content empty) or 'sub-line'
(content = the token). Tokens are real words for the meaningful passage,
syllables for T1, and orthographically-legal nonwords for T5 -- the linguistic
pipeline downstream is told which, and treats them honestly.
"""
from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_ROI_REQUIRED = {"content", "kind", "line", "part"}


@dataclass
class PassageStimulus:
    """Reconstructed stimulus for one passage."""
    name: str                # 'syllables' | 'meaningful' | 'pseudo'
    roi_file: str            # e.g. 'Meaningful_Text_rois.csv'
    tokens: list[str]        # sub-line tokens in reading order
    text: str                # tokens joined by single spaces
    n_lines: int
    n_tokens: int

    def summary(self) -> dict:
        return {
            "name": self.name,
            "roi_file": self.roi_file,
            "n_lines": self.n_lines,
            "n_tokens": self.n_tokens,
            "text_preview": (self.text[:80] + "...") if len(self.text) > 80 else self.text,
        }


def _read_roi_csv_from_zip(rois_zip: str | Path, roi_file: str) -> pd.DataFrame:
    """Read ``rois/<roi_file>`` out of ``rois.zip`` as a DataFrame."""
    rois_zip = Path(rois_zip)
    if not rois_zip.is_file():
        raise FileNotFoundError(f"rois zip not found: {rois_zip}")

    inner = f"rois/{roi_file}"
    with zipfile.ZipFile(rois_zip) as zf:
        names = set(zf.namelist())
        if inner not in names:
            # Fall back to a basename match (guards against a flat zip layout).
            candidates = [n for n in names
                          if n.endswith(roi_file) and "__MACOSX" not in n]
            if not candidates:
                raise KeyError(
                    f"'{roi_file}' not found in {rois_zip.name}; "
                    f"entries: {sorted(n for n in names if n.endswith('.csv'))}"
                )
            inner = candidates[0]
        with zf.open(inner) as fh:
            df = pd.read_csv(io.BytesIO(fh.read()))

    missing = _ROI_REQUIRED - set(df.columns)
    assert not missing, f"ROI file {roi_file} missing columns {missing}"
    return df


def reconstruct_passage(rois_zip: str | Path, name: str, roi_file: str) -> PassageStimulus:
    """Reconstruct a single passage's text from its ROI CSV."""
    df = _read_roi_csv_from_zip(rois_zip, roi_file)

    sub = df[df["kind"] == "sub-line"].copy()
    assert len(sub) > 0, f"no sub-line ROIs in {roi_file}; cannot reconstruct text"

    # Reading order: line, then part within a line.
    sub["line"] = pd.to_numeric(sub["line"], errors="coerce")
    sub["part"] = pd.to_numeric(sub["part"], errors="coerce")
    sub = sub.sort_values(["line", "part"], kind="stable")

    tokens = [str(t).strip() for t in sub["content"].tolist() if pd.notna(t)]
    # TODO(uncertain): a handful of tokens carry trailing punctuation (e.g.
    # 'ladiv.'); we keep them verbatim so the linguistic tokenizer sees the real
    # stimulus. Revisit if UDPipe/Stanza tokenization needs pre-cleaning.
    text = " ".join(tokens)

    return PassageStimulus(
        name=name,
        roi_file=roi_file,
        tokens=tokens,
        text=text,
        n_lines=int(sub["line"].nunique()),
        n_tokens=len(tokens),
    )


def load_all_stimuli(cfg) -> dict[str, PassageStimulus]:
    """Reconstruct all three passages keyed by passage name, per config."""
    rois_zip = cfg.paths.rois_zip
    out: dict[str, PassageStimulus] = {}
    for p in cfg.dataset.passages:
        out[p["name"]] = reconstruct_passage(rois_zip, p["name"], p["roi"])
    return out
