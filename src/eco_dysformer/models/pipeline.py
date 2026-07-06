"""End-to-end fitted pipeline: neural encoders -> frozen embedding -> LightGBM.

A ``FittedPipeline`` bundles, for one training partition:
    - fold-safe ``ArrayScaler`` (fit on train only) for gaze and linguistic feats
    - the trained ``EcoDysformer`` encoder (Performer or quadratic; conditioned or
      blind), trained end-to-end with the auxiliary BCE head
    - the LightGBM head fit on the FROZEN per-child embeddings

The neural encode and the LightGBM fit are separated so nested inner-loop tuning
can reuse one set of embeddings across a LightGBM hyperparameter grid instead of
retraining the encoder each time.

``predict_proba_flat`` accepts the flat, UNSCALED, interpretable feature matrix
(passage x feature) and runs the whole pipeline -- this is the entry point LIME
perturbs, keeping attributions in original-feature units (never PCA).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from eco_dysformer.data.tensors import ArrayScaler
from eco_dysformer.models.build import build_model
from eco_dysformer.models.heads import LightGBMHead


def resolve_device(cfg) -> torch.device:
    d = cfg.train.device
    if d == "auto":
        d = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(d)


# NOTE on augmentation: time-warp / segment-permutation (config) apply to the RAW
# event-stream regime; on the short 3-passage standardized feature vectors used by
# the core classifier, Gaussian feature jitter (applied inline in fit_encoder) is
# the meaningful augmentation.


@dataclass
class FittedPipeline:
    cfg: object
    attention: str
    conditioned: bool
    seed: int
    device: torch.device
    model: nn.Module | None = None
    gaze_scaler: ArrayScaler = field(default_factory=ArrayScaler)
    ling_scaler: ArrayScaler | None = None
    head: LightGBMHead | None = None
    epoch_time_s: float | None = None
    in_gaze: int = 0
    in_ling: int = 0
    n_passages: int = 3

    # ---------------------------------------------------------------- encoder
    def fit_encoder(self, Xg: np.ndarray, Xl: np.ndarray | None,
                    y: np.ndarray) -> np.ndarray:
        """Fit scalers + train the neural encoder. Returns train embeddings."""
        cfg = self.cfg
        self.in_gaze = Xg.shape[-1]
        self.n_passages = Xg.shape[1]
        Xg_s = self.gaze_scaler.fit_transform(Xg)
        if self.conditioned:
            assert Xl is not None, "conditioned pipeline requires linguistic feats"
            self.ling_scaler = ArrayScaler()
            Xl_s = self.ling_scaler.fit_transform(Xl)
            self.in_ling = Xl.shape[-1]
        else:
            Xl_s = None
            self.in_ling = Xl.shape[-1] if Xl is not None else 1

        torch.manual_seed(self.seed)
        self.model = build_model(cfg, in_gaze=self.in_gaze, in_ling=self.in_ling,
                                 attention=self.attention,
                                 conditioned=self.conditioned,
                                 seed=self.seed).to(self.device)

        gaze_t = torch.tensor(Xg_s, dtype=torch.float32, device=self.device)
        ling_t = (torch.tensor(Xl_s, dtype=torch.float32, device=self.device)
                  if Xl_s is not None else None)
        y_t = torch.tensor(y, dtype=torch.float32, device=self.device)

        opt = torch.optim.Adam(self.model.parameters(), lr=cfg.model.head.aux_lr,
                               weight_decay=cfg.model.head.aux_weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        aug = cfg.train.augmentation
        jitter = float(aug.jitter_std) if aug.enabled else 0.0

        self.model.train()
        epochs = int(cfg.model.head.aux_epochs)
        t0 = time.perf_counter()
        for _ in range(epochs):
            opt.zero_grad()
            g_in = gaze_t
            if jitter > 0:
                noise = torch.empty_like(g_in.cpu()).normal_(0.0, jitter, generator=gen)
                g_in = g_in + noise.to(self.device)
            logit, _ = self.model(g_in, ling_t)
            loss = loss_fn(logit, y_t)
            loss.backward()
            opt.step()
        self.epoch_time_s = (time.perf_counter() - t0) / max(epochs, 1)

        return self.embed(Xg, Xl)

    # ------------------------------------------------------------------- head
    def fit_head(self, emb: np.ndarray, y: np.ndarray,
                 lgbm_overrides: dict | None = None) -> "FittedPipeline":
        self.head = LightGBMHead(self.cfg, self.seed)
        if lgbm_overrides:
            self.head.model.set_params(**lgbm_overrides)
        self.head.fit(emb, y)
        return self

    def fit(self, Xg, Xl, y, lgbm_overrides: dict | None = None) -> "FittedPipeline":
        emb = self.fit_encoder(Xg, Xl, y)
        return self.fit_head(emb, y, lgbm_overrides)

    # -------------------------------------------------------------- inference
    @torch.no_grad()
    def embed(self, Xg: np.ndarray, Xl: np.ndarray | None) -> np.ndarray:
        self.model.eval()
        Xg_s = self.gaze_scaler.transform(Xg)
        gaze_t = torch.tensor(Xg_s, dtype=torch.float32, device=self.device)
        ling_t = None
        if self.conditioned:
            Xl_s = self.ling_scaler.transform(Xl)
            ling_t = torch.tensor(Xl_s, dtype=torch.float32, device=self.device)
        emb = self.model.encode(gaze_t, ling_t)
        return emb.cpu().numpy()

    def predict_proba(self, Xg: np.ndarray, Xl: np.ndarray | None) -> np.ndarray:
        assert self.head is not None, "pipeline head not fit"
        return self.head.predict_proba(self.embed(Xg, Xl))

    def predict_proba_flat(self, flat: np.ndarray) -> np.ndarray:
        """LIME entry point: flat UNSCALED (passage x feature) -> P(dyslexic).

        Returns an (N, 2) array [P(neg), P(pos)] as LIME expects.
        """
        flat = np.atleast_2d(flat)
        n = flat.shape[0]
        gaze_dim = self.n_passages * self.in_gaze
        Xg = flat[:, :gaze_dim].reshape(n, self.n_passages, self.in_gaze)
        Xl = None
        if self.conditioned:
            Xl = flat[:, gaze_dim:].reshape(n, self.n_passages, self.in_ling)
        p1 = self.predict_proba(Xg, Xl)
        return np.column_stack([1.0 - p1, p1])

    # ----------------------------------------------------------- explainability
    @torch.no_grad()
    def fusion_attention(self, Xg: np.ndarray, Xl: np.ndarray | None) -> np.ndarray | None:
        """Return mean passage-to-passage fusion attention (P, P), or None if blind."""
        if not self.conditioned:
            return None
        self.model.eval()
        cross = self.model.fusion.cross.attn
        cross.capture = True
        try:
            _ = self.embed(Xg, Xl)
            w = cross.last_attn        # (B, h, P, P)
            if w is None:
                return None
            return w.mean(dim=(0, 1)).cpu().numpy()
        finally:
            cross.capture = False
            cross.last_attn = None
