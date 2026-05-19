"""Vanilla E-GraphSAGE baseline (no SDE, no anti-concentration)."""
from __future__ import annotations
import torch.nn as nn
from ..models.egraphsage import EGraphSAGE


class EGraphSAGEBaseline(nn.Module):
    def __init__(self, feature_dim: int = 83, num_classes: int = 34, hidden: int = 128):
        super().__init__()
        self.encoder = EGraphSAGE(edge_features=feature_dim, hidden_dim=hidden)
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x):
        return self.head(self.encoder(x))

    forward_mc = forward                                # parity with SODE-Guard API
