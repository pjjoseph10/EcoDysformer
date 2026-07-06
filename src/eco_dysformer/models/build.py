"""Assemble the full Eco-Dysformer model and enforce RQ1 parameter-matching.

``EcoDysformer`` wires the gaze encoder, the optional linguistic encoder, and the
paired cross-attention fusion into a module that maps a child's per-passage
feature tensors to (logit, per-child embedding). The embedding is what the
LightGBM head consumes; the logit trains the encoders via the auxiliary head.

Two switches drive every Stage-1 arm:
    attention  : 'performer' (RQ1 core) | 'quadratic' (RQ1 baseline)
    conditioned: True  (complexity-conditioned, RQ2 core)
                 False (complexity-blind gaze-only baseline, RQ2 contrast)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from eco_dysformer.models.attention import learnable_param_count
from eco_dysformer.models.fusion import PairedCrossAttentionFusion
from eco_dysformer.models.gaze_encoder import GazeEncoder
from eco_dysformer.models.heads import AuxHead
from eco_dysformer.models.linguistic_encoder import LinguisticEncoder


class EcoDysformer(nn.Module):
    def __init__(self, *, in_gaze: int, in_ling: int, cfg, attention: str,
                 conditioned: bool, seed: int, n_passages: int = 3):
        super().__init__()
        self.conditioned = conditioned
        self.attention = attention
        m = cfg.model
        d = m.gaze_encoder.d_model
        nf = m.gaze_encoder.performer_features

        self.gaze_encoder = GazeEncoder(
            in_dim=in_gaze, d_model=d, n_heads=m.gaze_encoder.n_heads,
            n_layers=m.gaze_encoder.n_layers, attention=attention,
            dropout=m.gaze_encoder.dropout, n_features=nf, seed=seed,
            input_mode="features", n_passages=n_passages,
        )
        if conditioned:
            self.linguistic_encoder = LinguisticEncoder(
                in_dim=in_ling, d_model=m.linguistic_encoder.d_model,
                n_layers=m.linguistic_encoder.n_layers, attention=attention,
                n_heads=m.fusion.n_heads, dropout=m.linguistic_encoder.dropout,
                n_features=nf, seed=seed, n_passages=n_passages,
            )
            assert m.linguistic_encoder.d_model == d, (
                "gaze and linguistic d_model must match for fusion"
            )
        else:
            self.linguistic_encoder = None

        self.fusion = PairedCrossAttentionFusion(
            d_model=d, n_heads=m.fusion.n_heads, attention=attention,
            dropout=m.fusion.dropout, n_features=nf, seed=seed,
            conditioned=conditioned,
        )
        self.aux_head = AuxHead(d, dropout=m.gaze_encoder.dropout)
        self.embedding_dim = d

    def encode(self, gaze_feats: torch.Tensor,
               ling_feats: torch.Tensor | None) -> torch.Tensor:
        """Return the per-child fused embedding (B, d)."""
        gaze_tokens = self.gaze_encoder(gaze_feats)            # (B, P, d)
        ling_tokens = None
        if self.conditioned:
            assert ling_feats is not None, "conditioned model needs ling_feats"
            ling_tokens = self.linguistic_encoder(ling_feats)  # (B, P, d)
        return self.fusion(gaze_tokens, ling_tokens)           # (B, d)

    def forward(self, gaze_feats, ling_feats=None):
        emb = self.encode(gaze_feats, ling_feats)
        logit = self.aux_head(emb)
        return logit, emb


def build_model(cfg, *, in_gaze: int, in_ling: int, attention: str,
                conditioned: bool, seed: int) -> EcoDysformer:
    return EcoDysformer(in_gaze=in_gaze, in_ling=in_ling, cfg=cfg,
                        attention=attention, conditioned=conditioned, seed=seed)


def assert_param_matched(model_a: nn.Module, model_b: nn.Module,
                         tol: float = 0.02) -> None:
    """RQ1 guard: the Performer and quadratic arms must be parameter-matched."""
    pa = learnable_param_count(model_a)
    pb = learnable_param_count(model_b)
    rel = abs(pa - pb) / max(pa, pb)
    assert rel <= tol, (
        f"RQ1 arms are NOT parameter-matched: {pa} vs {pb} (rel diff {rel:.4f} "
        f"> tol {tol}). The comparison must isolate attention complexity, not "
        f"capacity."
    )


if __name__ == "__main__":
    # CPU smoke test (skipped cleanly if torch is unavailable): build both arms,
    # check param-match, run one forward pass. Uses random inputs -- NOT a result.
    import sys
    from pathlib import Path
    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from eco_dysformer.config import load_config

    cfg = load_config()
    B, P, Fg, Fl = 4, 3, 20, 11
    gaze = torch.randn(B, P, Fg)
    ling = torch.randn(B, P, Fl)

    perf = build_model(cfg, in_gaze=Fg, in_ling=Fl, attention="performer",
                       conditioned=True, seed=cfg.seed)
    quad = build_model(cfg, in_gaze=Fg, in_ling=Fl, attention="quadratic",
                       conditioned=True, seed=cfg.seed)
    assert_param_matched(perf, quad, cfg.model.param_match_tolerance)
    logit, emb = perf(gaze, ling)
    print(f"performer params: {learnable_param_count(perf)}")
    print(f"quadratic params: {learnable_param_count(quad)}")
    print(f"forward ok: logit {tuple(logit.shape)}, emb {tuple(emb.shape)}")
    blind = build_model(cfg, in_gaze=Fg, in_ling=Fl, attention="performer",
                        conditioned=False, seed=cfg.seed)
    lo2, em2 = blind(gaze, None)
    print(f"blind forward ok: logit {tuple(lo2.shape)}, emb {tuple(em2.shape)}")
