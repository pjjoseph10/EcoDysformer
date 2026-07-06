"""Stage-1 entry point: produce the full CORE result set and comparison table.

Stages (in order):
    1. inspect      dataset structure + asserts            -> dataset_inspection.json
    2. features     recompute gaze (+ cross-check) & Czech linguistic features
    3. rq2_effects  Cohen's d of gaze shift across gradient -> rq2_effects_*.csv
    4. model        subject-level nested CV across arms     -> cv_results.json
    5. crossover    RQ1 Performer-vs-quadratic scaling      -> rq1_crossover.csv/png
    6. explain      LIME (original features) + attention    -> lime_stability.json ...
    7. baselines    published-baseline comparison table     -> baseline_comparison.*
    8. summary      one roll-up                             -> stage1_summary.json

Usage:
    # Full run (Kaggle GPU, Internet for the Czech model download):
    python -m eco_dysformer.run_stage1

    # Local bare env (no torch/NLP): runs only the stages that work locally
    # (inspect, gaze features, RQ2 effects, baseline table) and skips the rest
    # with a clear note. No fabricated numbers are ever produced.
    python -m eco_dysformer.run_stage1 --local
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

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
    """Build conditioned (gaze+ling) and blind (gaze-only) child arrays."""
    import pandas as pd
    from eco_dysformer.data.tensors import build_child_arrays
    from eco_dysformer.features.gaze import GAZE_FEATURE_NAMES
    from eco_dysformer.features.linguistic import LINGUISTIC_FEATURE_NAMES

    long_path = Path(cfg.paths.features_dir) / "features_long.csv"
    if not long_path.is_file():
        raise FileNotFoundError(
            f"{long_path} missing -- run the features stage with linguistic "
            "features (needs the Czech NLP engine; Kaggle)."
        )
    df = pd.read_csv(long_path)
    ling_cols = [c for c in LINGUISTIC_FEATURE_NAMES if c in df.columns]
    ling_cols += [c for c in df.columns if c.startswith("ling_emb_")]
    conditioned = build_child_arrays(df, GAZE_FEATURE_NAMES, ling_cols=ling_cols)
    blind = build_child_arrays(df, GAZE_FEATURE_NAMES, ling_cols=None)
    return conditioned, blind


def main() -> int:
    ap = argparse.ArgumentParser(description="Eco-Dysformer v2 Stage-1 runner")
    ap.add_argument("--config", default=None, help="path to stage1.yaml")
    ap.add_argument("--local", action="store_true",
                    help="run only stages that work in the bare local env")
    ap.add_argument("--skip-explain", action="store_true",
                    help="skip the (slow) LIME/attention explainability stage")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg.seed)
    summary: dict = {"seed": cfg.seed, "local_mode": args.local, "stages": {}}

    def mark(stage: str, status: str, **extra):
        summary["stages"][stage] = {"status": status, **extra}
        print(f"[stage:{stage}] {status}")

    # 1. inspect --------------------------------------------------------------
    _banner("1/8 dataset inspection")
    from eco_dysformer.data.inspect_dataset import inspect
    report, _ = inspect(cfg)
    mark("inspect", "ok", n_subjects=report["n_subjects"])

    # 2. features -------------------------------------------------------------
    _banner("2/8 feature engineering")
    from eco_dysformer.features.assemble import assemble_features
    include_ling = not args.local
    try:
        tables = assemble_features(cfg, include_linguistic=include_ling)
        mark("features", "ok",
             gaze_rows=len(tables["gaze"]),
             linguistic="built" if "linguistic" in tables else "skipped")
    except Exception as e:  # linguistic engine unavailable, etc.
        traceback.print_exc()
        mark("features", "partial", error=str(e))

    # 3. rq2 effect sizes -----------------------------------------------------
    _banner("3/8 RQ2 effect sizes (gaze shift across gradient)")
    from eco_dysformer.eval.rq2_effects import run as run_rq2
    run_rq2(cfg)
    mark("rq2_effects", "ok")

    # 7. baselines table (built early; refreshed after model stage) -----------
    from eco_dysformer.eval.baselines_table import build_table, save_table

    if args.local:
        _banner("model / crossover / explain SKIPPED (--local)")
        mark("model", "skipped", reason="needs torch+lightgbm (Kaggle)")
        mark("crossover", "skipped", reason="needs torch (Kaggle)")
        mark("explain", "skipped", reason="needs trained pipeline (Kaggle)")
        save_table(cfg, build_table(cfg))
        mark("baselines", "ok")
        return _finish(cfg, summary)

    # 4. model: nested CV -----------------------------------------------------
    _banner("4/8 subject-level nested cross-validation")
    from eco_dysformer.eval.run_cv import run_nested_cv, save_results
    cond, blind = _build_arrays(cfg)
    cv_results = run_nested_cv(cfg, cond, blind)
    save_results(cfg, cv_results)
    mark("model", "ok",
         performer_acc=cv_results["arms"]["performer_conditioned"]["accuracy_mean"])

    # 5. RQ1 crossover --------------------------------------------------------
    _banner("5/8 RQ1 sequence-length crossover")
    from eco_dysformer.eval.rq1_crossover import run_crossover, save_crossover
    cross_df = run_crossover(cfg)
    cross_info = save_crossover(cfg, cross_df)
    mark("crossover", "ok", crossover_seq_len=cross_info["crossover_seq_len"])

    # 6. explainability -------------------------------------------------------
    if args.skip_explain:
        mark("explain", "skipped", reason="--skip-explain")
    else:
        _banner("6/8 explainability (LIME on original features + attention)")
        from eco_dysformer.eval.cv import nested_cv
        from eco_dysformer.explain.attention_extract import (
            run_attention_extraction, save_attention)
        from eco_dysformer.explain.lime_explain import run_lime_stability, save_lime
        folds = nested_cv(cond.subjects, cond.y, outer=cfg.eval.cv.outer_folds,
                          inner=cfg.eval.cv.inner_folds, seed=cfg.seed)
        lime_res = run_lime_stability(cfg, cond, folds)
        save_lime(cfg, lime_res)
        attn_res = run_attention_extraction(cfg, cond, folds)
        save_attention(cfg, attn_res)
        mark("explain", "ok",
             top_features=lime_res["top_features"],
             mean_jaccard=lime_res["stability_mean_jaccard_topk"])

    # 7. baselines (now populated with our results) ---------------------------
    _banner("7/8 baseline comparison table")
    save_table(cfg, build_table(cfg))
    mark("baselines", "ok")

    return _finish(cfg, summary)


def _finish(cfg, summary: dict) -> int:
    _banner("8/8 summary")
    out = Path(cfg.paths.results_dir) / "stage1_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f"\nStage-1 summary -> {out}")
    for s, r in summary["stages"].items():
        print(f"  {s:12s} {r['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
