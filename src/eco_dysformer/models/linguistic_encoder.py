"""Linguistic-complexity encoder.

Input is the per-passage linguistic feature vector (B, P=3, F_ling) -- the RQ2
conditioning signal (identical across children; 3 distinct values). Projects to
d_model, adds a passage position code, and applies self-attention blocks over the
three passage tokens, returning (B, 3, d) to pair with the gaze embeddings in the
cross-attention fusion.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from eco_dysformer.models.blocks import SelfAttentionBlock


class LinguisticEncoder(nn.Module):
    def __init__(self, in_dim: int, d_model: int, n_layers: int, attention: str,
                 n_heads: int = 4, dropout: float = 0.1, n_features: int = 64,
                 seed: int = 1337, n_passages: int = 3):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, n_passages, d_model))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([
            SelfAttentionBlock(attention, d_model, n_heads, dropout,
                               n_features=n_features, seed=seed + 100 + i)
            for i in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x) + self.pos
        for blk in self.blocks:
            h = blk(h)
        return self.norm(h)                            # (B, P, d)
