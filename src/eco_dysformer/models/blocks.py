"""Pre-norm transformer blocks built on the swappable attention.

A ``SelfAttentionBlock`` and a ``CrossAttentionBlock``, each parameterized by the
attention ``kind`` so the whole model can be flipped between Performer and
quadratic from one config switch while staying parameter-matched.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from eco_dysformer.models.attention import build_attention


def _ffn(d_model: int, mult: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_model, mult * d_model),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(mult * d_model, d_model),
    )


class SelfAttentionBlock(nn.Module):
    def __init__(self, kind: str, d_model: int, n_heads: int, dropout: float = 0.1,
                 ffn_mult: int = 2, n_features: int = 64, seed: int = 1337):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = build_attention(kind, d_model, n_heads, dropout,
                                    n_features=n_features, seed=seed)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = _ffn(d_model, ffn_mult, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.drop(self.attn(h))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class CrossAttentionBlock(nn.Module):
    """Query stream attends to a key/value stream (the paired fusion primitive)."""

    def __init__(self, kind: str, d_model: int, n_heads: int, dropout: float = 0.1,
                 ffn_mult: int = 2, n_features: int = 64, seed: int = 1337):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = build_attention(kind, d_model, n_heads, dropout,
                                    n_features=n_features, seed=seed)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = _ffn(d_model, ffn_mult, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(x_q)
        kv = self.norm_kv(x_kv)
        x = x_q + self.drop(self.attn(q, kv, kv))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x
