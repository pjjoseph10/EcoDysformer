"""Subject-level nested cross-validation across all Stage-1 arms.

Arms:
    performer_conditioned  -- RQ2 core & RQ1 core (Performer, complexity-conditioned)
    quadratic_conditioned  -- RQ1 baseline (parameter-matched quadratic attention)
    performer_blind        -- RQ2 contrast (complexity-blind, gaze-only)

Protocol (per arm):
    outer loop = performance estimate; inner loop = LightGBM hyperparameter
    selection (num_leaves grid) on the frozen embeddings. The neural encoder is
    trained once per (inner/outer) train partition and reused across the grid.
    Scaling is fit on train only; subject-level folds are asserted leak-free.

Headline comparisons use a paired Wilcoxon signed-rank test across the outer
folds plus bootstrap CIs (never single-split point estimates). Operational
metrics (params, latency, epoch time, peak GPU memory) are recorded per arm.

Torch + LightGBM -> Kaggle-first. Writes ``cv_results.json`` and per-fold CSVs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from eco_dysformer.data.tensors import ChildArrays
from eco_dysformer.eval.cv import assert_no_subject_leakage, fold_class_balance, nested_cv
from eco_dysformer.eval.metrics import classification_metrics
from eco_dysformer.eval.operational import measure_inference_latency, track_peak_gpu_memory
from eco_dysformer.eval.stats import bootstrap_ci, compare_arms
from eco_dysformer.models.build import assert_param_matched, build_model
from eco_dysformer.models.pipeline import FittedPipeline, resolve_device

# Inner-loop LightGBM hyperparameter grid (nested selection).
NUM_LEAVES_GRID = [7, 15, 31]

ARMS = {
    "performer_conditioned": dict(attention="performer", conditioned=True),
    "quadratic_conditioned": dict(attention="quadratic", conditioned=True),
    "performer_blind":        dict(attention="performer", conditioned=False),
}


def _slice(arrays: ChildArrays, idx: np.ndarray):
    Xg = arrays.X_gaze[idx]
    Xl = arrays.X_ling[idx] if arrays.X_ling is not None else None
    return Xg, Xl, arrays.y[idx]


def _select_num_leaves(arrays, fold, arm_kw, cfg, device, seed) -> int:
    """Inner CV: pick num_leaves by mean AUROC over the inner folds."""
    scores = {nl: [] for nl in NUM_LEAVES_GRID}
    for itr, ival in fold.inner_splits:
        Xg_tr, Xl_tr, y_tr = _slice(arrays, itr)
        Xg_va, Xl_va, y_va = _slice(arrays, ival)
        pipe = FittedPipeline(cfg=cfg, seed=seed, device=device, **arm_kw)
        emb_tr = pipe.fit_encoder(Xg_tr, Xl_tr, y_tr)
        for nl in NUM_LEAVES_GRID:
            pipe.fit_head(emb_tr, y_tr, lgbm_overrides={"num_leaves": nl})
            p = pipe.predict_proba(Xg_va, Xl_va)
            m = classification_metrics(y_va, p)
            scores[nl].append(m["auroc"] if m["auroc"] == m["auroc"] else 0.5)
    mean_scores = {nl: float(np.mean(v)) for nl, v in scores.items()}
    return max(mean_scores, key=mean_scores.get)


def run_arm(name: str, arrays: ChildArrays, folds, cfg, device, seed) -> dict:
    arm_kw = ARMS[name]
    per_fold, selected, epoch_times, oof = [], [], [], []
    for fold in folds:
        nl = _select_num_leaves(arrays, fold, arm_kw, cfg, device, seed)
        selected.append(nl)

        Xg_tr, Xl_tr, y_tr = _slice(arrays, fold.train_idx)
        Xg_te, Xl_te, y_te = _slice(arrays, fold.test_idx)
        pipe = FittedPipeline(cfg=cfg, seed=seed, device=device, **arm_kw)
        pipe.fit(Xg_tr, Xl_tr, y_tr, lgbm_overrides={"num_leaves": nl})
        epoch_times.append(pipe.epoch_time_s)

        proba = pipe.predict_proba(Xg_te, Xl_te)
        m = classification_metrics(y_te, proba)
        m["outer_fold"] = fold.index
        m["num_leaves"] = nl
        per_fold.append(m)

        # Out-of-fold predictions (each child predicted exactly once, on the fold
        # where it is held out) -- feeds the calibration reliability diagram.
        for s, yt, yp in zip(arrays.subjects[fold.test_idx], y_te, proba):
            oof.append({"arm": name, "outer_fold": fold.index,
                        "subject_id": int(s), "y_true": int(yt), "y_prob": float(yp)})

    df = pd.DataFrame(per_fold)
    def summ(col):
        return bootstrap_ci(df[col].to_numpy(), n_resamples=cfg.eval.bootstrap.n_resamples,
                            ci=cfg.eval.bootstrap.ci, seed=seed)
    return {
        "name": name,
        "per_fold": per_fold,
        "oof": oof,
        "accuracy_mean": float(df["accuracy"].mean()),
        "accuracy_ci": {k: summ("accuracy")[k] for k in ("lo", "hi")},
        "f1_mean": float(df["f1"].mean()),
        "auroc_mean": float(df["auroc"].mean()),
        "ece_mean": float(df["ece"].mean()),
        "brier_mean": float(df["brier"].mean()),
        "selected_num_leaves": selected,
        "epoch_time_s_mean": float(np.mean([e for e in epoch_times if e is not None])),
    }


def _operational(cfg, arrays, device, seed) -> dict:
    """Params / latency / peak memory for the Performer vs quadratic arms."""
    in_gaze = arrays.X_gaze.shape[-1]
    in_ling = arrays.X_ling.shape[-1] if arrays.X_ling is not None else 1
    B, P = 8, arrays.X_gaze.shape[1]
    gaze = torch.randn(B, P, in_gaze, device=device)
    ling = torch.randn(B, P, in_ling, device=device)

    out = {}
    perf = build_model(cfg, in_gaze=in_gaze, in_ling=in_ling, attention="performer",
                       conditioned=True, seed=seed).to(device)
    quad = build_model(cfg, in_gaze=in_gaze, in_ling=in_ling, attention="quadratic",
                       conditioned=True, seed=seed).to(device)
    assert_param_matched(perf, quad, cfg.model.param_match_tolerance)
    for tag, model in (("performer", perf), ("quadratic", quad)):
        with track_peak_gpu_memory(device) as mem:
            lat = measure_inference_latency(model, (gaze, ling), device,
                                            n_batches=cfg.eval.operational.latency_batches)
        out[tag] = {
            "param_count": sum(p.numel() for p in model.parameters()),
            "peak_gpu_mem_mb": mem["peak_mb"],
            **lat,
        }
    return out


def run_nested_cv(cfg, arrays_conditioned: ChildArrays,
                  arrays_blind: ChildArrays) -> dict:
    device = resolve_device(cfg)
    seed = cfg.seed
    subjects = arrays_conditioned.subjects
    y = arrays_conditioned.y
    folds = nested_cv(subjects, y, outer=cfg.eval.cv.outer_folds,
                      inner=cfg.eval.cv.inner_folds, seed=seed)
    assert_no_subject_leakage(folds, subjects)

    results = {
        "seed": seed,
        "device": str(device),
        "n_outer": cfg.eval.cv.outer_folds,
        "n_inner": cfg.eval.cv.inner_folds,
        "leakage_checked": True,
        "fold_class_balance": fold_class_balance(folds, y),
        "arms": {},
        "comparisons": {},
    }

    arm_arrays = {
        "performer_conditioned": arrays_conditioned,
        "quadratic_conditioned": arrays_conditioned,
        "performer_blind": arrays_blind,
    }
    for name in ARMS:
        results["arms"][name] = run_arm(name, arm_arrays[name], folds, cfg,
                                        device, seed)

    def fold_vec(arm, metric):
        return [f[metric] for f in results["arms"][arm]["per_fold"]]

    # RQ1: Performer vs parameter-matched quadratic (same conditioned features).
    results["comparisons"]["RQ1_performer_vs_quadratic"] = {
        m: compare_arms(fold_vec("performer_conditioned", m),
                        fold_vec("quadratic_conditioned", m),
                        name_a="performer_conditioned", name_b="quadratic_conditioned",
                        alternative=cfg.eval.wilcoxon.alternative,
                        n_resamples=cfg.eval.bootstrap.n_resamples,
                        ci=cfg.eval.bootstrap.ci, seed=seed)
        for m in ("accuracy", "auroc", "f1")
    }
    # RQ2: complexity-conditioned vs complexity-blind (both Performer).
    results["comparisons"]["RQ2_conditioned_vs_blind"] = {
        m: compare_arms(fold_vec("performer_conditioned", m),
                        fold_vec("performer_blind", m),
                        name_a="performer_conditioned", name_b="performer_blind",
                        alternative=cfg.eval.wilcoxon.alternative,
                        n_resamples=cfg.eval.bootstrap.n_resamples,
                        ci=cfg.eval.bootstrap.ci, seed=seed)
        for m in ("accuracy", "auroc", "f1")
    }

    results["operational"] = _operational(cfg, arrays_conditioned, device, seed)
    return results


def save_results(cfg, results: dict) -> dict:
    res_dir = Path(cfg.paths.results_dir)
    res_dir.mkdir(parents=True, exist_ok=True)

    # Pull the bulky out-of-fold predictions out of the JSON into their own CSV.
    oof_rows = []
    for arm, r in results["arms"].items():
        oof_rows.extend(r.pop("oof", []))
    if oof_rows:
        pd.DataFrame(oof_rows).to_csv(res_dir / "cv_oof_predictions.csv", index=False)

    with open(res_dir / "cv_results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    rows = []
    for arm, r in results["arms"].items():
        for f in r["per_fold"]:
            rows.append({"arm": arm, **f})
    pd.DataFrame(rows).to_csv(res_dir / "cv_per_fold.csv", index=False)
    return {"json": str(res_dir / "cv_results.json"),
            "csv": str(res_dir / "cv_per_fold.csv"),
            "oof_csv": str(res_dir / "cv_oof_predictions.csv")}
