"""Paired cross-attention fusion over a child's three passage tokens.

The gaze token stream (B, 3, d) attends to the linguistic-complexity token stream
(B, 3, d). Because every (gaze, linguistic) pair really did come from the same
child reading that specific passage, this cross-attention is over *genuinely
paired* examples -- the methodological point of the whole project (contrast with
the disjoint-cohort fusion the project explicitly avoids).

Output is mean-pooled over the three passages to a single per-child vector
(B, d), which the auxiliary head trains on and the LightGBM head classifies.

The complexity-BLIND baseline reuses this module with ``conditioned=False``: the
linguistic stream is ignored and the gaze tokens are pooled directly, so the two
arms differ only in whether complexity conditioning is present.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from eco_dysformer.models.blocks import CrossAttentionBlock


class PairedCrossAttentionFusion(nn.Module):
    def __init__(self, d_model: int, n_heads: int, attention: str,
                 dropout: float = 0.1, n_features: int = 64, seed: int = 1337,
                 conditioned: bool = True):
        super().__init__()
        self.conditioned = conditioned
        if conditioned:
            self.cross = CrossAttentionBlock(attention, d_model, n_heads, dropout,
                                             n_features=n_features, seed=seed + 200)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, gaze_tokens: torch.Tensor,
                ling_tokens: torch.Tensor | None) -> torch.Tensor:
        if self.conditioned:
            assert ling_tokens is not None, (
                "conditioned fusion requires linguistic tokens"
            )
            fused = self.cross(gaze_tokens, ling_tokens)   # (B, P, d)
        else:
            fused = gaze_tokens
        pooled = self.norm(fused).mean(dim=1)              # (B, d) per-child vector
        return pooled
