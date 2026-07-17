"""Per-child handwriting reversal-rate risk feature (RQ3).

The handwriting cohort is DISJOINT from ETDD70 and has NO writer linkage, so a
"writing sample per child" does not exist in the data. We therefore construct
SYNTHETIC writing samples: pool K character images, score each with the trained
reversal classifier, and take the mean P(Reversal) as an aggregated
reversal-rate. Images are drawn from the handwriting TEST split (unseen by the
classifier) so the score is not inflated by memorized training images.

Two assignment regimes, controlled by ``alpha``:

  alpha = 0  (HONEST control) -- the sample is drawn independently of the child.
             The feature carries NO information about the child. This is the
             truthful state of affairs for two disjoint, unlinked cohorts.

  alpha = 1  (NAIVE, class-aligned) -- the sample's Reversal/Normal mix is skewed
             by the child's CLASS. This DELIBERATELY RECONSTRUCTS the flaw in
             disjoint-cohort fusion: pairing samples from a matching class
             manufactures a cross-modal correlation that is a cohort artifact,
             not genuine subject-level signal.

             *** This is a NEGATIVE EXAMPLE, never a proposed method. It leaks
             label information on purpose, exactly as implicit class-aligned
             fusion does in prior work. It must never be reported as a result of
             the proposed model. ***

The resulting feature is a NOISY label proxy at alpha=1 (the classifier is
imperfect and the draw is stochastic), which makes it a realistic reconstruction
rather than a trivial label copy.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Pure-numpy core (no torch) -- unit-testable without the images.
# --------------------------------------------------------------------------- #
def assign_reversal_rates(rev_probs: np.ndarray, char_is_reversal: np.ndarray,
                          child_classes: np.ndarray, *, chars_per_sample: int,
                          alpha: float, delta: float, rng) -> np.ndarray:
    """Build one synthetic writing sample per child; return its reversal-rate.

    Parameters
    ----------
    rev_probs
        (n_images,) classifier P(Reversal) for each image in the source pool.
    char_is_reversal
        (n_images,) bool, the image's TRUE label (Reversal=True) -- used only to
        compose the draw, never fed to the model.
    child_classes
        (n_children,) 0/1 ETDD70 labels.
    alpha
        0 = random draw (honest); 1 = fully class-aligned (naive flaw).
    delta
        Mix skew at alpha=1 (e.g. 0.35 -> 85% / 15% reversal chars).

    Returns
    -------
    (n_children,) mean P(Reversal) over each child's K sampled characters.
    """
    assert rev_probs.shape == char_is_reversal.shape
    rev_idx = np.where(char_is_reversal)[0]
    nor_idx = np.where(~char_is_reversal)[0]
    assert len(rev_idx) and len(nor_idx), "pool needs both Reversal and Normal images"

    out = np.empty(len(child_classes), dtype=float)
    for i, c in enumerate(child_classes):
        sign = 1.0 if int(c) == 1 else -1.0
        p_rev = float(np.clip(0.5 + alpha * delta * sign, 0.0, 1.0))
        n_rev = int(rng.binomial(chars_per_sample, p_rev))
        picks = np.concatenate([
            rng.choice(rev_idx, size=n_rev, replace=True),
            rng.choice(nor_idx, size=chars_per_sample - n_rev, replace=True),
        ])
        out[i] = float(rev_probs[picks].mean())
    return out


# --------------------------------------------------------------------------- #
# Torch-dependent scoring of the image pool (Kaggle).
# --------------------------------------------------------------------------- #
def score_pool(cfg, model, seed: int, device=None):
    """Score every image in the source split; return (rev_probs, is_reversal)."""
    import torch
    from torch.utils.data import DataLoader

    from eco_dysformer.handwriting.data import HandwritingDataset, discover_split

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hw = cfg.rq3.handwriting
    class_map = hw.classes.to_dict() if hasattr(hw.classes, "to_dict") else dict(hw.classes)
    items = discover_split(Path(hw.data_root), cfg.rq3.synthetic_sample.source_split,
                           class_map)
    ds = HandwritingDataset(items, image_size=hw.image_size, grayscale=hw.grayscale)
    loader = DataLoader(ds, batch_size=hw.train.batch_size, shuffle=False,
                        num_workers=hw.train.num_workers)

    model.eval().to(device)
    probs, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            probs.append(model.reversal_prob(x.to(device)).cpu().numpy())
            labels.append(y.numpy())
    rev_probs = np.concatenate(probs)
    y = np.concatenate(labels)
    reversal_label = class_map.get("Reversal", 1)
    return rev_probs, (y == reversal_label)


def build_risk_features(cfg, model, subjects: np.ndarray, child_classes: np.ndarray,
                        seed: int, device=None) -> pd.DataFrame:
    """Return a per-child table with the aligned (naive) and random (honest) feature."""
    ss = cfg.rq3.synthetic_sample
    rev_probs, is_rev = score_pool(cfg, model, seed, device)

    def make(alpha, tag):
        rng = np.random.default_rng(seed + ss.seed_offset + int(alpha * 1000))
        return assign_reversal_rates(
            rev_probs, is_rev, child_classes, chars_per_sample=ss.chars_per_sample,
            alpha=alpha, delta=ss.class_alignment_delta, rng=rng)

    df = pd.DataFrame({
        "subject_id": subjects,
        "class_id": child_classes,
        "reversal_rate_aligned": make(cfg.rq3.fusion.alignment_alpha_naive, "aligned"),
        "reversal_rate_random": make(cfg.rq3.fusion.alignment_alpha_honest, "random"),
    })
    df.attrs["pool_size"] = int(len(rev_probs))
    return df


def save_risk_features(cfg, df: pd.DataFrame) -> Path:
    out = Path(cfg.paths.features_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "rq3_handwriting_risk.csv"
    df.to_csv(path, index=False)

    # Sanity summary: how strongly does each feature track the child's class?
    summary = {}
    for col in ("reversal_rate_aligned", "reversal_rate_random"):
        d = df[df.class_id == 1][col].to_numpy()
        t = df[df.class_id == 0][col].to_numpy()
        summary[col] = {"mean_dyslexic": float(d.mean()), "mean_typical": float(t.mean()),
                        "point_biserial_r": float(np.corrcoef(df[col], df.class_id)[0, 1])}
    summary["note"] = ("aligned = DELIBERATE flaw reconstruction (class-leaking); "
                       "random = honest, uninformative. A large |r| for 'aligned' and "
                       "~0 for 'random' is the expected, intended contrast.")
    with open(Path(cfg.paths.results_dir) / "rq3_risk_feature_summary.json", "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return path
