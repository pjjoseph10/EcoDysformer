"""LIME explanations on ORIGINAL interpretable features, with fold stability.

For each outer fold we train the complexity-conditioned pipeline on outer-train,
build a LIME tabular explainer whose feature space is the flat, interpretable
"passage__feature" matrix (gaze + linguistic, in original units -- never PCA),
and explain each outer-test child through ``pipeline.predict_proba_flat``. We
aggregate per-feature mean |attribution| within a fold, then report:

    - top-k features by mean |attribution| (pooled across folds)
    - attribution stability across folds: mean pairwise Jaccard of the top-k
      feature sets, and mean pairwise Spearman of the full importance vectors
    - a face-validity check: fraction of the pooled top-5 features that are
      literature-documented biomarkers (fixation / regression / reading-time /
      dwell terms). This is interpretability face-validity, NOT a clinical claim.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from eco_dysformer.data.tensors import ChildArrays
from eco_dysformer.models.pipeline import FittedPipeline, resolve_device

# Substrings marking literature-documented oculomotor dyslexia biomarkers.
BIOMARKER_TERMS = ("regression", "fix_count", "fix_dur", "read_time", "dwell",
                   "fixation", "sacc")


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 1.0


def _explain_fold(pipe: FittedPipeline, train_flat, test_flat, feature_names,
                  n_samples: int, seed: int) -> dict[str, float]:
    from lime.lime_tabular import LimeTabularExplainer

    explainer = LimeTabularExplainer(
        training_data=train_flat,
        feature_names=feature_names,
        class_names=["non-dyslexic", "dyslexic"],
        mode="classification",
        discretize_continuous=True,
        random_state=seed,
    )
    n_feat = len(feature_names)
    abs_acc = np.zeros(n_feat)
    for row in test_flat:
        exp = explainer.explain_instance(
            row, pipe.predict_proba_flat, num_features=n_feat,
            num_samples=n_samples, labels=(1,),
        )
        for idx, w in exp.as_map()[1]:
            abs_acc[idx] += abs(w)
    abs_acc /= max(len(test_flat), 1)
    return {feature_names[i]: float(abs_acc[i]) for i in range(n_feat)}


def run_lime_stability(cfg, arrays: ChildArrays, folds, *, top_k: int = 5) -> dict:
    device = resolve_device(cfg)
    seed = cfg.seed
    feature_names = arrays.flat_feature_names()
    flat_all = arrays.flatten()

    per_fold_importance = []
    for fold in folds:
        Xg_tr = arrays.X_gaze[fold.train_idx]
        Xl_tr = arrays.X_ling[fold.train_idx] if arrays.X_ling is not None else None
        y_tr = arrays.y[fold.train_idx]

        pipe = FittedPipeline(cfg=cfg, seed=seed, device=device,
                              attention="performer", conditioned=True)
        pipe.fit(Xg_tr, Xl_tr, y_tr)

        imp = _explain_fold(pipe, flat_all[fold.train_idx], flat_all[fold.test_idx],
                            feature_names, cfg.explain.lime_samples, seed)
        per_fold_importance.append(imp)

    imp_df = pd.DataFrame(per_fold_importance).fillna(0.0)   # (folds, features)
    pooled = imp_df.mean(axis=0).sort_values(ascending=False)
    top_features = pooled.head(top_k).index.tolist()

    # Stability: pairwise Jaccard of per-fold top-k, pairwise Spearman of vectors.
    fold_topsets = [set(imp_df.loc[i].sort_values(ascending=False).head(top_k).index)
                    for i in imp_df.index]
    jaccards = [_jaccard(a, b) for a, b in combinations(fold_topsets, 2)]
    spearmans = [spearmanr(imp_df.loc[i], imp_df.loc[j]).correlation
                 for i, j in combinations(imp_df.index, 2)]

    n_bio = sum(any(t in f for t in BIOMARKER_TERMS) for f in top_features)
    return {
        "top_features": top_features,
        "pooled_importance": pooled.to_dict(),
        "stability_mean_jaccard_topk": float(np.mean(jaccards)) if jaccards else 1.0,
        "stability_mean_spearman": float(np.nanmean(spearmans)) if spearmans else 1.0,
        "biomarker_facevalidity_top5": float(n_bio / max(len(top_features), 1)),
        "top_k": top_k,
        "note": "interpretability face-validity only; not a clinical claim",
    }


def save_lime(cfg, result: dict) -> None:
    res_dir = Path(cfg.paths.results_dir)
    res_dir.mkdir(parents=True, exist_ok=True)
    with open(res_dir / "lime_stability.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    pd.Series(result["pooled_importance"]).sort_values(ascending=False).to_csv(
        res_dir / "lime_pooled_importance.csv", header=["mean_abs_attribution"])