"""Lipschitz-bounded MLP baseline: the certifiable encoder with a
spectral-normalised linear head and no SDE. Its logits are L-Lipschitz with
L = L_h * ||W||, so the Lipschitz-margin certificate of Tsuzuku et al. (2018)
gives the l2 radius M(x) / (sqrt(2) L)."""
from __future__ import annotations
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm

from ..models.egraphsage import EGraphSAGE, spectral_norm_exact


class LipschitzMLP(nn.Module):
    def __init__(self, feature_dim: int = 83, num_classes: int = 34, hidden: int = 128):
        super().__init__()
        self.encoder = EGraphSAGE(edge_features=feature_dim, hidden_dim=hidden, lipschitz=True)
        self.head = spectral_norm(nn.Linear(hidden, num_classes))

    def forward(self, x):
        return self.head(self.encoder(x))

    def lipschitz_constant(self) -> float:
        was = self.training
        self.eval()
        try:
            return self.encoder.lipschitz_bound() * spectral_norm_exact(self.head.weight)
        finally:
            self.train(was)
