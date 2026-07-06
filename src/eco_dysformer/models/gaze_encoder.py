"""Gaze encoder (Performer or quadratic).

Two input modes:
    'features' -- the CORE classification path. Input is the short engineered
        per-passage gaze feature vector: (B, P=3, F_gaze). Each passage is
        projected to d_model, given a learned passage/complexity position code,
        and passed through self-attention blocks over the 3 passage tokens.
        Returns per-passage embeddings (B, 3, d).
    'events'   -- the RQ1 crossover path. Input is the raw fixation/saccade event
        stream (B, L, E) with L up to hundreds. Projected and self-attended over
        the L events, then masked-mean-pooled to (B, d). This is the regime where
        linear vs quadratic attention actually differs; used by rq1_crossover and
        available as an alternative gaze embedding source.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from eco_dysformer.models.blocks import SelfAttentionBlock


class GazeEncoder(nn.Module):
    def __init__(self, in_dim: int, d_model: int, n_heads: int, n_layers: int,
                 attention: str, dropout: float = 0.1, n_features: int = 64,
                 seed: int = 1337, input_mode: str = "features",
                 n_passages: int = 3):
        super().__init__()
        self.input_mode = input_mode
        self.d_model = d_model
        self.input_proj = nn.Linear(in_dim, d_model)
        if input_mode == "features":
            # learned position code per passage (complexity slot)
            self.pos = nn.Parameter(torch.zeros(1, n_passages, d_model))
            nn.init.normal_(self.pos, std=0.02)
        else:
            self.pos = None
        self.blocks = nn.ModuleList([
            SelfAttentionBlock(attention, d_model, n_heads, dropout,
                               n_features=n_features, seed=seed + i)
            for i in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """features mode -> (B, P, d); events mode -> (B, d) pooled."""
        h = self.input_proj(x)
        if self.input_mode == "features":
            h = h + self.pos
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        if self.input_mode == "features":
            return h                                   # (B, P, d)
        # events: masked mean pool over the event axis
        if mask is not None:
            m = mask.unsqueeze(-1).to(h.dtype)         # (B, L, 1)
            return (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        return h.mean(dim=1)                            # (B, d)
