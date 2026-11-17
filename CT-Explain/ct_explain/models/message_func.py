"""Edge-conditioned message function g_φ.

Paper: g_φ : ℝ^{d_h} × ℝ^{d_e} → ℝ^{d_h} computes the contribution of a
neighbour u's current hidden state and the edge feature e_{vu} to the
incoming message before attention-weighted aggregation in Equation 1.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class EdgeConditionedMessage(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        dropout: float = 0.0,
        mlp_hidden: Sequence[int] = (256,),
        activation: type[nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = hidden_dim + edge_dim
        for h in mlp_hidden:
            layers += [nn.Linear(in_dim, h), activation(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, hidden_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, h_u: torch.Tensor, e_vu: torch.Tensor) -> torch.Tensor:
        """h_u: (E, d_h), e_vu: (E, d_e) → m: (E, d_h)."""
        return self.net(torch.cat([h_u, e_vu], dim=-1))
