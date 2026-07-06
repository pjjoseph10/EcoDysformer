"""End-to-end model smoke test (Kaggle: torch+lightgbm). Skips in the bare env.

Fast synthetic check that the neural pipeline wires up correctly: encoder trains,
LightGBM head fits, probabilities are in range, the Performer and quadratic arms
are parameter-matched, the blind arm runs without linguistic input, the flat LIME
entry point returns a 2-column proba, and fusion attention is capturable. Uses
random inputs with tiny epochs -- this is a wiring test, NOT a result.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import torch  # noqa: F401
    import lightgbm  # noqa: F401
    HAVE_STACK = True
except ImportError:
    HAVE_STACK = False

try:
    import pytest  # type: ignore
    pytestmark = pytest.mark.skipif(
        not HAVE_STACK, reason="torch/lightgbm not installed (run on Kaggle)")
except ImportError:
    pytest = None  # bare-env __main__ path handles the skip itself

if HAVE_STACK:
    from eco_dysformer.config import load_config
    from eco_dysformer.models.build import assert_param_matched, build_model
    from eco_dysformer.models.pipeline import FittedPipeline, resolve_device


def _cfg_fast():
    cfg = load_config()
    # shrink epochs for a fast smoke; mutate the underlying dict in place
    cfg.model.head._data["aux_epochs"] = 3  # type: ignore[attr-defined]
    return cfg


def _synth(n=24, P=3, Fg=20, Fl=11, seed=0):
    rng = np.random.default_rng(seed)
    Xg = rng.standard_normal((n, P, Fg))
    Xl = rng.standard_normal((n, P, Fl))
    y = (rng.random(n) > 0.5).astype(int)
    return Xg, Xl, y


def test_param_match_performer_quadratic():
    cfg = load_config()
    perf = build_model(cfg, in_gaze=20, in_ling=11, attention="performer",
                       conditioned=True, seed=cfg.seed)
    quad = build_model(cfg, in_gaze=20, in_ling=11, attention="quadratic",
                       conditioned=True, seed=cfg.seed)
    assert_param_matched(perf, quad, cfg.model.param_match_tolerance)


def test_conditioned_pipeline_fit_predict():
    cfg = _cfg_fast()
    dev = resolve_device(cfg)
    Xg, Xl, y = _synth()
    pipe = FittedPipeline(cfg=cfg, seed=cfg.seed, device=dev,
                          attention="performer", conditioned=True)
    pipe.fit(Xg[:16], Xl[:16], y[:16])
    p = pipe.predict_proba(Xg[16:], Xl[16:])
    assert p.shape == (8,) and np.all((p >= 0) & (p <= 1))
    # LIME entry point
    flat = np.concatenate([Xg[16:].reshape(8, -1), Xl[16:].reshape(8, -1)], axis=1)
    p2 = pipe.predict_proba_flat(flat)
    assert p2.shape == (8, 2) and np.allclose(p2.sum(axis=1), 1.0)
    # attention capturable (3x3)
    attn = pipe.fusion_attention(Xg[16:], Xl[16:])
    assert attn.shape == (3, 3)


def test_blind_pipeline_runs_without_linguistic():
    cfg = _cfg_fast()
    dev = resolve_device(cfg)
    Xg, _, y = _synth()
    pipe = FittedPipeline(cfg=cfg, seed=cfg.seed, device=dev,
                          attention="performer", conditioned=False)
    pipe.fit(Xg[:16], None, y[:16])
    p = pipe.predict_proba(Xg[16:], None)
    assert p.shape == (8,) and np.all((p >= 0) & (p <= 1))
    assert pipe.fusion_attention(Xg[16:], None) is None


if __name__ == "__main__":
    if not HAVE_STACK:
        print("SKIP: torch/lightgbm not installed (this smoke test targets Kaggle).")
        raise SystemExit(0)
    from _run import run_module_tests
    raise SystemExit(run_module_tests(globals()))
