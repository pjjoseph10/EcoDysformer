"""Lightweight CNN for 28x28 handwriting character images (RQ3).

Deliberately NOT a LeViT/ViT: the images are 28x28 (MNIST-scale), so a small
convolutional net is faster, smaller, and consistent with the project's
efficiency ("Eco") framing. Two conv blocks (channels from config) -> pooled
embedding -> linear classifier. ``embed`` exposes the pre-logit embedding;
``reversal_prob`` gives P(Reversal) for the risk-feature stage.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class HandwritingCNN(nn.Module):
    def __init__(self, in_channels: int = 1, channels=(16, 32), embed_dim: int = 64,
                 n_classes: int = 2, image_size: int = 28, dropout: float = 0.1,
                 reversal_index: int = 1):
        super().__init__()
        self.reversal_index = reversal_index
        c1, c2 = channels
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, padding=1), nn.BatchNorm2d(c1), nn.ReLU(),
            nn.MaxPool2d(2),                                    # 28 -> 14
            nn.Conv2d(c1, c2, 3, padding=1), nn.BatchNorm2d(c2), nn.ReLU(),
            nn.MaxPool2d(2),                                    # 14 -> 7
        )
        feat = c2 * (image_size // 4) * (image_size // 4)
        self.embed_fc = nn.Sequential(
            nn.Flatten(), nn.Linear(feat, embed_dim), nn.ReLU(), nn.Dropout(dropout))
        self.classifier = nn.Linear(embed_dim, n_classes)
        self.embedding_dim = embed_dim

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed_fc(self.features(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.embed(x))

    @torch.no_grad()
    def reversal_prob(self, x: torch.Tensor) -> torch.Tensor:
        """P(Reversal) per image (softmax over the reversal class index)."""
        return torch.softmax(self.forward(x), dim=-1)[:, self.reversal_index]


def build_handwriting_cnn(cfg, in_channels: int, n_classes: int) -> HandwritingCNN:
    enc = cfg.rq3.handwriting.encoder
    return HandwritingCNN(
        in_channels=in_channels, channels=tuple(enc.channels), embed_dim=enc.embed_dim,
        n_classes=n_classes, image_size=cfg.rq3.handwriting.image_size,
        dropout=enc.dropout, reversal_index=1)
