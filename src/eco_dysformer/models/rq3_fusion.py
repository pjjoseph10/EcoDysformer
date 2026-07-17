"""RQ3 fusion models: NAIVE joint cross-attention over a disjoint cohort.

``NaiveJointModel`` pools gaze, linguistic AND handwriting features into ONE
joint attention block (3 gaze tokens + 3 linguistic tokens + 1 handwriting
token = 7 tokens), which is exactly how prior comparable work fuses modalities
drawn from disjoint cohorts. It is implemented faithfully so its behaviour can be
measured -- it is the NEGATIVE example in the RQ3 ablation, contrasted against the
honest calibrated late fusion in ``eval/run_rq3.py``.

*** No claim of subject-level joint learning is made here. The handwriting cohort
shares no subjects with ETDD70 and has no writer linkage, so the handwriting token
carries no genuine per-child information. Any accuracy this arm gains over the
gaze-only baseline is a cohort artifact, which is the point of the experiment. ***

Contrast with the Stage-1 core (``models/build.EcoDysformer``), whose paired
cross-attention is defensible precisely because gaze and linguistic complexity ARE
genuinely paired within each child.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from eco_dysformer.data.tensors import ArrayScaler
from eco_dysformer.models.blocks import SelfAttentionBlock
from eco_dysformer.models.gaze_encoder import GazeEncoder
from eco_dysformer.models.heads import AuxHead, LightGBMHead
from eco_dysformer.models.linguistic_encoder import LinguisticEncoder


class NaiveJointModel(nn.Module):
    """Gaze + linguistic + handwriting pooled into one joint attention block."""

    def __init__(self, *, in_gaze: int, in_ling: int, in_hw: int, cfg,
                 attention: str, seed: int, n_passages: int = 3):
        super().__init__()
        m = cfg.model
        d = m.gaze_encoder.d_model
        nf = m.gaze_encoder.performer_features
        drop = m.gaze_encoder.dropout

        self.gaze_encoder = GazeEncoder(
            in_dim=in_gaze, d_model=d, n_heads=m.gaze_encoder.n_heads,
            n_layers=m.gaze_encoder.n_layers, attention=attention, dropout=drop,
            n_features=nf, seed=seed, input_mode="features", n_passages=n_passages)
        self.linguistic_encoder = LinguisticEncoder(
            in_dim=in_ling, d_model=m.linguistic_encoder.d_model,
            n_layers=m.linguistic_encoder.n_layers, attention=attention,
            n_heads=m.fusion.n_heads, dropout=m.linguistic_encoder.dropout,
            n_features=nf, seed=seed, n_passages=n_passages)
        # The disjoint-cohort handwriting feature enters as a single extra token.
        self.hw_proj = nn.Sequential(nn.Linear(in_hw, d), nn.LayerNorm(d))
        self.joint_blocks = nn.ModuleList([
            SelfAttentionBlock(attention, d, m.fusion.n_heads, m.fusion.dropout,
                               n_features=nf, seed=seed + 300 + i)
            for i in range(m.gaze_encoder.n_layers)
        ])
        self.norm = nn.LayerNorm(d)
        self.aux_head = AuxHead(d, dropout=drop)
        self.embedding_dim = d

    def encode(self, gaze: torch.Tensor, ling: torch.Tensor,
               hw: torch.Tensor) -> torch.Tensor:
        g = self.gaze_encoder(gaze)               # (B, 3, d)
        l = self.linguistic_encoder(ling)         # (B, 3, d)
        h = self.hw_proj(hw).unsqueeze(1)         # (B, 1, d)
        seq = torch.cat([g, l, h], dim=1)         # (B, 7, d)  <- one joint block
        for blk in self.joint_blocks:
            seq = blk(seq)
        return self.norm(seq).mean(dim=1)         # (B, d)

    def forward(self, gaze, ling, hw):
        emb = self.encode(gaze, ling, hw)
        return self.aux_head(emb), emb


@dataclass
class RQ3NaivePipeline:
    """Naive-joint-fusion pipeline: encoders -> frozen embedding -> LightGBM."""
    cfg: object
    attention: str
    seed: int
    device: torch.device
    model: nn.Module | None = None
    gaze_scaler: ArrayScaler = field(default_factory=ArrayScaler)
    ling_scaler: ArrayScaler = field(default_factory=ArrayScaler)
    hw_scaler: ArrayScaler = field(default_factory=ArrayScaler)
    head: LightGBMHead | None = None
    epoch_time_s: float | None = None

    def fit_encoder(self, Xg, Xl, Xh, y) -> np.ndarray:
        cfg = self.cfg
        Xg_s = self.gaze_scaler.fit_transform(Xg)
        Xl_s = self.ling_scaler.fit_transform(Xl)
        Xh_s = self.hw_scaler.fit_transform(Xh)          # (N, F_hw)

        torch.manual_seed(self.seed)
        self.model = NaiveJointModel(
            in_gaze=Xg.shape[-1], in_ling=Xl.shape[-1], in_hw=Xh.shape[-1],
            cfg=cfg, attention=self.attention, seed=self.seed).to(self.device)

        t = lambda a: torch.tensor(a, dtype=torch.float32, device=self.device)
        gaze_t, ling_t, hw_t, y_t = t(Xg_s), t(Xl_s), t(Xh_s), t(y)

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
            logit, _ = self.model(g_in, ling_t, hw_t)
            loss_fn(logit, y_t).backward()
            opt.step()
        self.epoch_time_s = (time.perf_counter() - t0) / max(epochs, 1)
        return self.embed(Xg, Xl, Xh)

    def fit_head(self, emb, y, lgbm_overrides: dict | None = None):
        self.head = LightGBMHead(self.cfg, self.seed)
        if lgbm_overrides:
            self.head.model.set_params(**lgbm_overrides)
        self.head.fit(emb, y)
        return self

    def fit(self, Xg, Xl, Xh, y, lgbm_overrides: dict | None = None):
        return self.fit_head(self.fit_encoder(Xg, Xl, Xh, y), y, lgbm_overrides)

    @torch.no_grad()
    def embed(self, Xg, Xl, Xh) -> np.ndarray:
        self.model.eval()
        t = lambda a: torch.tensor(a, dtype=torch.float32, device=self.device)
        emb = self.model.encode(t(self.gaze_scaler.transform(Xg)),
                                t(self.ling_scaler.transform(Xl)),
                                t(self.hw_scaler.transform(Xh)))
        return emb.cpu().numpy()

    def predict_proba(self, Xg, Xl, Xh) -> np.ndarray:
        assert self.head is not None, "pipeline head not fit"
        return self.head.predict_proba(self.embed(Xg, Xl, Xh))
