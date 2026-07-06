"""Classification heads.

Two heads, by design:
    AuxHead        -- a tiny MLP on the per-child fused vector, trained end-to-end
                      with BCE to give the neural encoders a learning signal.
    LightGBMHead   -- the PRIMARY downstream classifier (proposal 5.3/6.5). It is
                      fit on the *frozen* fused embeddings extracted from the
                      trained encoders, per outer-train fold.
The two-stage split (neural representation -> gradient-boosted classifier) is
deliberate and matches the proposal.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from eco_dysformer.seed import lightgbm_seed_params


class AuxHead(nn.Module):
    def __init__(self, d_model: int, hidden: int | None = None, dropout: float = 0.1):
        super().__init__()
        hidden = hidden or d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)                 # (B,) logits


class LightGBMHead:
    """Thin wrapper around ``lightgbm.LGBMClassifier`` with pinned seeds."""

    def __init__(self, cfg, seed: int):
        try:
            import lightgbm as lgb
        except ImportError as e:  # pragma: no cover - env dependent
            raise ImportError(
                "LightGBM is required for the downstream head. `pip install "
                "lightgbm` (preinstalled on Kaggle)."
            ) from e
        lc = cfg.model.lightgbm
        params = dict(
            n_estimators=lc.n_estimators,
            learning_rate=lc.learning_rate,
            num_leaves=lc.num_leaves,
            max_depth=lc.max_depth,
            subsample=lc.subsample,
            colsample_bytree=lc.colsample_bytree,
            reg_lambda=lc.reg_lambda,
            min_child_samples=lc.min_child_samples,
            objective="binary",
            n_jobs=1,
            verbose=-1,
        )
        params.update(lightgbm_seed_params(seed))
        self.model = lgb.LGBMClassifier(**params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LightGBMHead":
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.model.feature_importances_
