"""Stage-2 entry point: RQ3 handwriting branch + naive-vs-honest fusion.

Pipeline:
  1. handwriting  -- train the reversal classifier on the (disjoint) handwriting
                     cohort              -> rq3_handwriting_classifier.json / .pt
  2. risk_feature -- synthetic writing samples -> per-child reversal-rate, in a
                     CLASS-ALIGNED (naive flaw) and a RANDOM (honest) variant
                                          -> rq3_handwriting_risk.csv
  3. rq3          -- all fusion arms under subject-level nested CV
                                          -> rq3_results.json / rq3_summary.csv

Prerequisites (Kaggle):
  * the handwriting images extracted to ``rq3.handwriting.data_root``
  * ``features_long.csv`` present in ``paths.features_dir`` (run run_stage1, or
    copy it from the archived stage1_outputs/features/)

    python -m eco_dysformer.run_stage2
    python -m eco_dysformer.run_stage2 --skip-handwriting-train   # reuse saved .pt
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

# LightGBM's sklearn wrapper sets feature names even for plain arrays, so newer
# scikit-learn emits a spurious "X does not have valid feature names" warning on
# every predict. We pass numpy in identical column order at fit and predict, so
# predictions are unaffected -- silence the noise, not a real problem.
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

from eco_dysformer.config import load_config
from eco_dysformer.seed import set_global_seed


def _banner(msg: str) -> None:
    print("\n" + "#" * 78 + f"\n# {msg}\n" + "#" * 78)


def _build_arrays(cfg):
    import pandas as pd
    from eco_dysformer.data.tensors import build_child_arrays
    from eco_dysformer.features.gaze import GAZE_FEATURE_NAMES
    from eco_dysformer.features.linguistic import LINGUISTIC_FEATURE_NAMES

    path = Path(cfg.paths.features_dir) / "features_long.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} missing -- run run_stage1 (features stage) first, or copy it "
            "from the archived stage1_outputs/features/features_long.csv")
    df = pd.read_csv(path)
    ling_cols = [c for c in LINGUISTIC_FEATURE_NAMES if c in df.columns]
    ling_cols += [c for c in df.columns if c.startswith("ling_emb_")]
    return build_child_arrays(df, GAZE_FEATURE_NAMES, ling_cols=ling_cols)


def _load_handwriting_model(cfg):
    import torch
    from eco_dysformer.handwriting.encoder import build_handwriting_cnn

    hw = cfg.rq3.handwriting
    class_map = hw.classes.to_dict() if hasattr(hw.classes, "to_dict") else dict(hw.classes)
    n_classes = len(set(class_map.values())) + (1 if hw.include_corrected and
                                                "Corrected" not in class_map else 0)
    model = build_handwriting_cnn(cfg, 1 if hw.grayscale else 3, n_classes)
    ckpt = Path(cfg.paths.results_dir) / "rq3_handwriting_cnn.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"{ckpt} missing -- run without --skip-handwriting-train")
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return model


def main() -> int:
    ap = argparse.ArgumentParser(description="Eco-Dysformer v2 Stage-2 (RQ3) runner")
    ap.add_argument("--config", default=None)
    ap.add_argument("--skip-handwriting-train", action="store_true",
                    help="reuse the saved rq3_handwriting_cnn.pt")
    ap.add_argument("--sweep", action="store_true",
                    help="also run the alignment-strength dose-response sweep")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg.seed)
    summary: dict = {"seed": cfg.seed, "stages": {}}

    # 1. handwriting classifier -------------------------------------------------
    _banner("1/3 handwriting reversal classifier (disjoint cohort)")
    if args.skip_handwriting_train:
        model = _load_handwriting_model(cfg)
        print("  reused saved checkpoint")
        summary["stages"]["handwriting"] = {"status": "reused"}
    else:
        from eco_dysformer.handwriting.train import train_handwriting
        model, hw_res = train_handwriting(cfg, cfg.seed)
        print("  test metrics:", hw_res["test_metrics"])
        summary["stages"]["handwriting"] = {
            "status": "ok", "test_accuracy": hw_res["test_metrics"]["accuracy"],
            "param_count": hw_res["param_count"]}

    # 2. per-child risk feature -------------------------------------------------
    _banner("2/3 per-child reversal-rate risk feature (aligned vs random)")
    from eco_dysformer.handwriting.risk_feature import build_risk_features, save_risk_features
    arrays = _build_arrays(cfg)
    risk = build_risk_features(cfg, model, arrays.subjects, arrays.y, cfg.seed)
    path = save_risk_features(cfg, risk)
    for col in ("reversal_rate_aligned", "reversal_rate_random"):
        import numpy as np
        r = float(np.corrcoef(risk[col], risk["class_id"])[0, 1])
        print(f"  {col:26s} point-biserial r with class = {r:+.3f}")
    print(f"  -> {path}")
    summary["stages"]["risk_feature"] = {"status": "ok", "csv": str(path)}

    # 3. RQ3 fusion ablation ----------------------------------------------------
    _banner("3/3 RQ3 fusion ablation (subject-level nested CV)")
    from eco_dysformer.eval.run_rq3 import run_rq3, save_rq3
    results = run_rq3(cfg, arrays)
    info = save_rq3(cfg, results)
    print("\n  per-arm accuracy / ECE:")
    for arm, r in results["arms"].items():
        print(f"    {arm:16s} acc={r['accuracy_mean']:.3f}  auroc={r['auroc_mean']:.3f}"
              f"  ece={r['ece_mean']:.3f}")
    h = results["headline"]
    print("\n  HEADLINE:")
    print(f"    manufactured gain (naive_aligned - gaze_only) : "
          f"{h['manufactured_gain_accuracy']['paired_diff_ci']['mean_diff']:+.3f}")
    print(f"    cohort artifact  (naive_aligned - naive_random): "
          f"{h['cohort_artifact_accuracy']['paired_diff_ci']['mean_diff']:+.3f}")
    print(f"    ECE cost         (naive_aligned - honest_aligned): "
          f"{h['naive_vs_honest_ece']['paired_diff_ci']['mean_diff']:+.3f}")
    summary["stages"]["rq3"] = {"status": "ok", **info}

    # 4. (optional) alignment-strength dose-response sweep ----------------------
    if args.sweep:
        _banner("4/4 RQ3 alignment-strength sweep (dose-response)")
        from eco_dysformer.eval.run_rq3 import run_alignment_sweep, save_alignment_sweep
        sweep_df = run_alignment_sweep(cfg, arrays, model)
        sinfo = save_alignment_sweep(cfg, sweep_df)
        summary["stages"]["rq3_sweep"] = {"status": "ok", **sinfo}

    out = Path(cfg.paths.results_dir) / "stage2_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    _banner("summary")
    print(f"Stage-2 summary -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
