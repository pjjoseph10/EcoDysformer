"""Swappable attention: quadratic (softmax) vs Performer (FAVOR+ linear).

This is the crux of RQ1. Both variants share identical *learned* parameters
(q/k/v/out projections with the same shapes), so a model built with one vs the
other is parameter-matched by construction -- the only difference is how the
attention itself is computed:

    QuadraticAttention : materializes the N x N score matrix -> O(N^2) time/mem.
    PerformerAttention : FAVOR+ positive random features -> O(N * m * d), linear
                         in sequence length N (m = number of random features).

The random-feature matrix is a fixed, seeded, non-trainable buffer (so it adds
zero learnable parameters and keeps runs deterministic). Both support self- and
cross-attention via the same ``forward(query, key, value)`` signature, which the
paired cross-attention fusion relies on.

RQ1 caveat baked in: on short sequences (e.g. the 3 passage tokens) Performer's
constant overhead makes it SLOWER than quadratic; the crossover only appears on
the long raw event streams. ``rq1_crossover`` measures where.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _MHABase(nn.Module):
    """Shared projections / head reshaping for both attention variants."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        # Explainability hook: when True, `last_attn` holds the (B,h,Nq,Nk)
        # attention weight matrix from the most recent forward pass. For the
        # Performer this is the implicit normalized kernel phi(q)phi(k)^T -- only
        # meaningful/cheap for small N (e.g. the 3-passage fusion).
        self.capture = False
        self.last_attn: torch.Tensor | None = None

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        return x.view(b, n, self.n_heads, self.d_head).transpose(1, 2)  # (b,h,n,dh)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        b, h, n, dh = x.shape
        return x.transpose(1, 2).contiguous().view(b, n, h * dh)

    def forward(self, query, key=None, value=None):
        key = query if key is None else key
        value = key if value is None else value
        q = self._split(self.q_proj(query))
        k = self._split(self.k_proj(key))
        v = self._split(self.v_proj(value))
        ctx = self._attend(q, k, v)          # (b,h,nq,dh)
        return self.out_proj(self._merge(ctx))

    def _attend(self, q, k, v):  # pragma: no cover - overridden
        raise NotImplementedError


class QuadraticAttention(_MHABase):
    """Standard scaled dot-product (softmax) attention -- O(N^2)."""

    def _attend(self, q, k, v):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = F.softmax(scores, dim=-1)
        if self.capture:
            self.last_attn = attn.detach()
        return torch.matmul(self.dropout(attn), v)


class PerformerAttention(_MHABase):
    """FAVOR+ kernelized linear attention -- O(N * m * d).

    Positive random features approximate the softmax kernel:
        phi(x) = exp(omega . x' - ||x'||^2 / 2) / sqrt(m),   x' = x * d_head^-1/4
    with a per-row max-subtraction for numerical stability. omega is a fixed
    seeded Gaussian buffer (non-trainable).
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0,
                 n_features: int = 64, seed: int = 1337):
        super().__init__(d_model, n_heads, dropout)
        self.n_features = n_features
        g = torch.Generator().manual_seed(seed)
        # (n_heads, d_head, m) orthogonal-ish Gaussian projection, fixed buffer.
        omega = torch.randn(n_heads, self.d_head, n_features, generator=g)
        self.register_buffer("omega", omega, persistent=True)

    def _phi(self, x: torch.Tensor, is_query: bool) -> torch.Tensor:
        # x: (b,h,n,dh). scale improves the softmax approximation.
        x = x * (self.d_head ** -0.25)
        proj = torch.einsum("bhnd,hdm->bhnm", x, self.omega)   # (b,h,n,m)
        sq = (x ** 2).sum(dim=-1, keepdim=True) * 0.5           # (b,h,n,1)
        # FAVOR+ stabilization: for QUERIES a per-query max (dim=-1) that cancels
        # in the num/den ratio; for KEYS a SINGLE constant across all keys (max
        # over key+feature axes) so it also cancels. A per-key max would rescale
        # each key non-uniformly and distort attention (verified ~1.5x worse).
        if is_query:
            stab = torch.amax(proj, dim=-1, keepdim=True).detach()
        else:
            stab = torch.amax(proj, dim=(-2, -1), keepdim=True).detach()
        phi = torch.exp(proj - sq - stab) + 1e-6
        return phi / math.sqrt(self.n_features)

    def _attend(self, q, k, v):
        qf = self._phi(q, is_query=True)                        # (b,h,nq,m)
        kf = self._phi(k, is_query=False)                       # (b,h,nk,m)
        kv = torch.einsum("bhnm,bhnd->bhmd", kf, v)             # (b,h,m,dh)
        z = kf.sum(dim=2)                                       # (b,h,m)
        num = torch.einsum("bhnm,bhmd->bhnd", qf, kv)           # (b,h,nq,dh)
        den = torch.einsum("bhnm,bhm->bhn", qf, z).unsqueeze(-1) + 1e-6
        if self.capture:
            # implicit normalized attention matrix (cheap only for small N)
            w = torch.einsum("bhqm,bhkm->bhqk", qf, kf)
            self.last_attn = (w / (w.sum(dim=-1, keepdim=True) + 1e-6)).detach()
        return self.dropout(num / den)


def build_attention(kind: str, d_model: int, n_heads: int, dropout: float = 0.0,
                    n_features: int = 64, seed: int = 1337) -> _MHABase:
    """Factory: ``kind`` in {'performer', 'quadratic'} -> matched attention module."""
    kind = kind.lower()
    if kind == "quadratic":
        return QuadraticAttention(d_model, n_heads, dropout)
    if kind == "performer":
        return PerformerAttention(d_model, n_heads, dropout,
                                  n_features=n_features, seed=seed)
    raise ValueError(f"unknown attention kind {kind!r}; use performer|quadratic")


def learnable_param_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)