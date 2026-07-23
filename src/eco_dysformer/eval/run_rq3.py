"""RQ3: naive vs honest fusion of a DISJOINT-cohort handwriting signal.

Arms (all inside the SAME subject-level nested CV as Stage 1):

  gaze_only        Stage-1 core (gaze + linguistic). Reference accuracy.
  naive_aligned    Naive joint attention + CLASS-ALIGNED handwriting feature.
                   Deliberately reconstructs the disjoint-cohort flaw -> expected
                   to look inflated. A NEGATIVE EXAMPLE, never a proposed method.
  naive_random     Naive joint attention + RANDOM handwriting feature. The control
                   that proves any naive_aligned gain is a cohort ARTIFACT: the
                   handwriting cohort shares no subjects with ETDD70, so an
                   honestly-assigned feature carries no per-child information.
  honest_aligned   Honest calibrated late fusion (isotonic on the handwriting risk
                   + a decision-level meta-classifier over the core score), given
                   the same class-aligned feature.
  honest_random    Same, with the random feature.

Headline RQ3 numbers:
  accuracy(naive_aligned) - accuracy(gaze_only)      = manufactured gain
  accuracy(naive_aligned) - accuracy(naive_random)   = the cohort artifact
  ECE(naive_*) - ECE(honest_*)                       = the calibration cost

No leakage: the isotonic calibrator and the meta-classifier are fit on the
outer-train fold only, and the core scores that train the meta-classifier are
out-of-fold predictions from the inner splits.

NOTE (documented simplification): num_leaves is fixed to 7 for every RQ3 arm,
inherited from Stage 1 where the inner loop selected 7 unanimously (30/30
selections). The outer loop still provides the performance estimate and
subject-level folds are preserved.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from eco_dysformer.data.tensors import ChildArrays
from eco_dysformer.eval.cv import assert_no_subject_leakage, nested_cv
from eco_dysformer.eval.metrics import classification_metrics
from eco_dysformer.eval.stats import compare_arms
from eco_dysformer.models.pipeline import FittedPipeline, resolve_device
from eco_dysformer.models.rq3_fusion import RQ3NaivePipeline

NUM_LEAVES_RQ3 = 7
HW_COLS = {"aligned": "reversal_rate_aligned", "random": "reversal_rate_random"}


def _fit_core(cfg, seed, device, Xg, Xl, y) -> FittedPipeline:
    p = FittedPipeline(cfg=cfg, seed=seed, device=device,
                       attention="performer", conditioned=True)
    p.fit(Xg, Xl, y, lgbm_overrides={"num_leaves": NUM_LEAVES_RQ3})
    return p


def _pack(per_fold: list[dict], name: str) -> dict:
    df = pd.DataFrame(per_fold)
    return {
        "name": name,
        "per_fold": per_fold,
        "accuracy_mean": float(df["accuracy"].mean()),
        "f1_mean": float(df["f1"].mean()),
        "auroc_mean": float(df["auroc"].mean()),
        "ece_mean": float(df["ece"].mean()),
        "brier_mean": float(df["brier"].mean()),
    }


def arm_gaze_only(arrays: ChildArrays, folds, cfg, device, seed) -> dict:
    per_fold = []
    for fold in folds:
        tr, te = fold.train_idx, fold.test_idx
        p = _fit_core(cfg, seed, device, arrays.X_gaze[tr], arrays.X_ling[tr], arrays.y[tr])
        proba = p.predict_proba(arrays.X_gaze[te], arrays.X_ling[te])
        m = classification_metrics(arrays.y[te], proba)
        m["outer_fold"] = fold.index
        per_fold.append(m)
    return _pack(per_fold, "gaze_only")


def arm_naive(arrays: ChildArrays, hw: np.ndarray, folds, cfg, device, seed,
              name: str) -> dict:
    per_fold = []
    for fold in folds:
        tr, te = fold.train_idx, fold.test_idx
        p = RQ3NaivePipeline(cfg=cfg, attention="performer", seed=seed, device=device)
        p.fit(arrays.X_gaze[tr], arrays.X_ling[tr], hw[tr], arrays.y[tr],
              lgbm_overrides={"num_leaves": NUM_LEAVES_RQ3})
        proba = p.predict_proba(arrays.X_gaze[te], arrays.X_ling[te], hw[te])
        m = classification_metrics(arrays.y[te], proba)
        m["outer_fold"] = fold.index
        per_fold.append(m)
    return _pack(per_fold, name)


def arm_honest(arrays: ChildArrays, hw: np.ndarray, folds, cfg, device, seed,
               name: str) -> dict:
    """Calibrated late fusion: isotonic(handwriting) + meta over the core score."""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    per_fold = []
    for fold in folds:
        tr, te = fold.train_idx, fold.test_idx

        # 1. Out-of-fold core scores on the outer-train rows (inner splits).
        pos = {int(g): i for i, g in enumerate(tr)}
        core_oof = np.full(len(tr), np.nan)
        for itr, ival in fold.inner_splits:
            pi = _fit_core(cfg, seed, device, arrays.X_gaze[itr], arrays.X_ling[itr],
                           arrays.y[itr])
            pv = pi.predict_proba(arrays.X_gaze[ival], arrays.X_ling[ival])
            for j, rid in enumerate(ival):
                core_oof[pos[int(rid)]] = pv[j]
        assert not np.isnan(core_oof).any(), "inner folds must cover all outer-train rows"

        # 2. Calibrate the handwriting risk INDEPENDENTLY, on train only.
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(hw[tr].ravel(), arrays.y[tr])
        hw_cal_tr = iso.predict(hw[tr].ravel())
        hw_cal_te = iso.predict(hw[te].ravel())

        # 3. Decision-level meta-classifier (no joint subject-level learning claim).
        meta = LogisticRegression(max_iter=1000)
        meta.fit(np.column_stack([core_oof, hw_cal_tr]), arrays.y[tr])

        # 4. Refit the core on the full outer-train, then fuse on test.
        p_full = _fit_core(cfg, seed, device, arrays.X_gaze[tr], arrays.X_ling[tr],
                           arrays.y[tr])
        core_te = p_full.predict_proba(arrays.X_gaze[te], arrays.X_ling[te])
        proba = meta.predict_proba(np.column_stack([core_te, hw_cal_te]))[:, 1]

        m = classification_metrics(arrays.y[te], proba)
        m["outer_fold"] = fold.index
        per_fold.append(m)
    return _pack(per_fold, name)


def load_handwriting_risk(cfg, subjects: np.ndarray) -> dict[str, np.ndarray]:
    """Load the per-child risk features, aligned to ``subjects`` order."""
    path = Path(cfg.paths.features_dir) / "rq3_handwriting_risk.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} missing -- run the handwriting risk-feature step first "
            "(handwriting.train then handwriting.risk_feature).")
    df = pd.read_csv(path).set_index("subject_id")
    missing = set(subjects.tolist()) - set(df.index.tolist())
    assert not missing, f"risk features missing for subjects {sorted(missing)[:5]}"
    df = df.loc[subjects]
    return {k: df[col].to_numpy(dtype=float).reshape(-1, 1) for k, col in HW_COLS.items()}


def run_rq3(cfg, arrays: ChildArrays) -> dict:
    device = resolve_device(cfg)
    seed = cfg.seed
    folds = nested_cv(arrays.subjects, arrays.y, outer=cfg.eval.cv.outer_folds,
                      inner=cfg.eval.cv.inner_folds, seed=seed)
    assert_no_subject_leakage(folds, arrays.subjects)
    hw = load_handwriting_risk(cfg, arrays.subjects)

    results = {
        "seed": seed, "device": str(device), "n_outer": cfg.eval.cv.outer_folds,
        "leakage_checked": True,
        "num_leaves_fixed": NUM_LEAVES_RQ3,
        "disclaimer": (
            "The handwriting cohort is DISJOINT from ETDD70 and has no writer "
            "linkage. The *_aligned arms use a deliberately class-aligned feature "
            "that reconstructs the flaw in prior disjoint-cohort fusion; they are "
            "negative examples, NOT proposed methods."),
        "arms": {},
    }
    print("[rq3] gaze_only ...")
    results["arms"]["gaze_only"] = arm_gaze_only(arrays, folds, cfg, device, seed)
    for tag in ("aligned", "random"):
        print(f"[rq3] naive_{tag} ...")
        results["arms"][f"naive_{tag}"] = arm_naive(
            arrays, hw[tag], folds, cfg, device, seed, f"naive_{tag}")
        print(f"[rq3] honest_{tag} ...")
        results["arms"][f"honest_{tag}"] = arm_honest(
            arrays, hw[tag], folds, cfg, device, seed, f"honest_{tag}")

    def vec(arm, metric):
        return [f[metric] for f in results["arms"][arm]["per_fold"]]

    def cmp(a, b, metric):
        return compare_arms(vec(a, metric), vec(b, metric), name_a=a, name_b=b,
                            alternative=cfg.eval.wilcoxon.alternative,
                            n_resamples=cfg.eval.bootstrap.n_resamples,
                            ci=cfg.eval.bootstrap.ci, seed=seed)

    results["headline"] = {
        "manufactured_gain_accuracy": cmp("naive_aligned", "gaze_only", "accuracy"),
        "cohort_artifact_accuracy": cmp("naive_aligned", "naive_random", "accuracy"),
        "naive_vs_honest_accuracy": cmp("naive_aligned", "honest_aligned", "accuracy"),
        "naive_vs_honest_ece": cmp("naive_aligned", "honest_aligned", "ece"),
        "honest_random_vs_gaze_only_accuracy": cmp("honest_random", "gaze_only", "accuracy"),
    }
    return results


def run_alignment_sweep(cfg, arrays: ChildArrays, model) -> pd.DataFrame:
    """Dose-response: naive & honest accuracy/ECE vs alignment strength (delta).

    The image pool is scored ONCE; each delta only re-mixes the synthetic
    writing-sample draw, so this is far cheaper than re-scoring per delta. The
    reference arms (gaze_only, and the honest random control) are delta-free.
    """
    from eco_dysformer.handwriting.risk_feature import assign_reversal_rates, score_pool

    device = resolve_device(cfg)
    seed = cfg.seed
    folds = nested_cv(arrays.subjects, arrays.y, outer=cfg.eval.cv.outer_folds,
                      inner=cfg.eval.cv.inner_folds, seed=seed)
    assert_no_subject_leakage(folds, arrays.subjects)

    rev_probs, is_rev = score_pool(cfg, model, seed, device)
    ss = cfg.rq3.synthetic_sample

    gaze = arm_gaze_only(arrays, folds, cfg, device, seed)
    rows = [{"delta": None, "r_with_class": 0.0, "arm": "gaze_only",
             "accuracy": gaze["accuracy_mean"], "ece": gaze["ece_mean"],
             "auroc": gaze["auroc_mean"]}]

    for delta in cfg.rq3.sweep.deltas:
        rng = np.random.default_rng(seed + ss.seed_offset + int(round(delta * 1000)))
        feat = assign_reversal_rates(
            rev_probs, is_rev, arrays.y, chars_per_sample=ss.chars_per_sample,
            alpha=1.0, delta=float(delta), rng=rng).reshape(-1, 1)
        r = float(np.corrcoef(feat.ravel(), arrays.y)[0, 1])
        na = arm_naive(arrays, feat, folds, cfg, device, seed, f"naive_d{delta}")
        ho = arm_honest(arrays, feat, folds, cfg, device, seed, f"honest_d{delta}")
        for arm, res in (("naive", na), ("honest", ho)):
            rows.append({"delta": float(delta), "r_with_class": r, "arm": arm,
                         "accuracy": res["accuracy_mean"], "ece": res["ece_mean"],
                         "auroc": res["auroc_mean"]})
        print(f"  delta={delta:.2f}  r={r:+.3f}  "
              f"naive acc={na['accuracy_mean']:.3f}/ece={na['ece_mean']:.3f}  "
              f"honest acc={ho['accuracy_mean']:.3f}/ece={ho['ece_mean']:.3f}")
    return pd.DataFrame(rows)


def save_alignment_sweep(cfg, df: pd.DataFrame) -> dict:
    res_dir = Path(cfg.paths.results_dir)
    fig_dir = Path(cfg.paths.figures_dir)
    res_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(res_dir / "rq3_alignment_sweep.csv", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sweep = df[df["delta"].notna()]
        gaze_acc = float(df[df["arm"] == "gaze_only"]["accuracy"].iloc[0])
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
        for arm, color in (("naive", "#E45756"), ("honest", "#4C78A8")):
            d = sweep[sweep["arm"] == arm]
            ax1.plot(d["r_with_class"], d["accuracy"], "o-", color=color, label=arm)
            ax2.plot(d["r_with_class"], d["ece"], "o-", color=color, label=arm)
        ax1.axhline(gaze_acc, ls="--", color="grey", label="gaze-only")
        ax1.set_xlabel("feature-class alignment r"); ax1.set_ylabel("accuracy")
        ax1.set_title("Manufactured accuracy vs alignment"); ax1.legend(); ax1.grid(alpha=.3)
        ax2.set_xlabel("feature-class alignment r"); ax2.set_ylabel("ECE")
        ax2.set_title("Calibration vs alignment"); ax2.legend(); ax2.grid(alpha=.3)
        fig.suptitle("RQ3 dose-response: disjoint-cohort artifact vs alignment strength")
        fig.tight_layout()
        fig.savefig(fig_dir / "rq3_alignment_sweep.png", dpi=120)
        plt.close(fig)
    except ImportError:
        pass
    return {"csv": str(res_dir / "rq3_alignment_sweep.csv")}


def save_rq3(cfg, results: dict) -> dict:
    res_dir = Path(cfg.paths.results_dir)
    res_dir.mkdir(parents=True, exist_ok=True)
    with open(res_dir / "rq3_results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    rows = []
    for arm, r in results["arms"].items():
        for f in r["per_fold"]:
            rows.append({"arm": arm, **f})
    pd.DataFrame(rows).to_csv(res_dir / "rq3_per_fold.csv", index=False)

    summary = pd.DataFrame([
        {"arm": a, "accuracy": r["accuracy_mean"], "f1": r["f1_mean"],
         "auroc": r["auroc_mean"], "ece": r["ece_mean"], "brier": r["brier_mean"]}
        for a, r in results["arms"].items()])
    summary.to_csv(res_dir / "rq3_summary.csv", index=False)
    return {"json": str(res_dir / "rq3_results.json"),
            "summary": str(res_dir / "rq3_summary.csv")}
