"""Neural-ODE dynamics function f_θ used in Equation 1.

The ODE closure wraps the current graph snapshot and a pre-computed
neighbour-message tensor so that the solver (`torchdiffeq.odeint`) can invoke
f_θ without reshaping data at every step. The closure is intentionally
stateless with respect to parameters — all trainable weights live on the
`NeuralODEFunc` module itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn

from ct_explain.models.message_func import EdgeConditionedMessage
from ct_explain.models.temporal_attention import TemporalMultiHeadAttention


@dataclass
class GraphSnapshot:
    """Static data needed by f_θ during one ODE integration window."""

    edge_index: torch.Tensor          # (2, E)
    edge_features: torch.Tensor       # (E, d_e)
    delta_t: torch.Tensor             # (E,)  (t_query − t_{vu})
    num_nodes: int


class NeuralODEFunc(nn.Module):
    """Dynamics
        dh_v/dt = f_θ(h_v(t), ⊕_{u∈N(v,t)} α_{vu}(t) · g_φ(h_u(t), e_{vu}), t)

    parameterised by an MLP over [h_v ∥ aggregated_message ∥ t_embed].
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        num_heads: int = 4,
        time_dim: int = 32,
        mlp_hidden: Sequence[int] = (256, 256),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        self.message = EdgeConditionedMessage(hidden_dim, edge_dim, dropout=dropout)
        self.attention = TemporalMultiHeadAttention(
            hidden_dim, num_heads=num_heads, time_dim=time_dim, dropout=dropout
        )

        in_dim = 2 * hidden_dim + time_dim
        layers: list[nn.Module] = []
        for h in mlp_hidden:
            layers += [nn.Linear(in_dim, h), nn.SiLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, hidden_dim))
        self.net = nn.Sequential(*layers)

        self.snapshot: Optional[GraphSnapshot] = None
        # Cached attention per solver step (useful for explainability).
        self.last_attention: Optional[torch.Tensor] = None

    def bind(self, snapshot: GraphSnapshot) -> None:
        self.snapshot = snapshot
        self.last_attention = None

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:  # noqa: D401
        if self.snapshot is None:
            raise RuntimeError("Call NeuralODEFunc.bind(snapshot) before odeint.")

        snap = self.snapshot
        src = snap.edge_index[1]
        tgt_h = h                                         # all target nodes
        src_h = h[src]                                    # (E, d_h)
        messages = self.message(src_h, snap.edge_features)  # (E, d_h)

        aggregated, attn = self.attention(
            h_v=tgt_h,
            h_u=src_h,
            delta_t=snap.delta_t,
            edge_index=snap.edge_index,
            messages=messages,
            num_targets=snap.num_nodes,
        )
        self.last_attention = attn

        # Time features for explicit t-dependence (global scalar broadcast).
        t_vec = t.expand(snap.num_nodes, 1) if t.dim() == 0 else t.view(-1, 1)
        t_feat = self.attention.time_encoding(t_vec.squeeze(-1))
        dh = self.net(torch.cat([h, aggregated, t_feat], dim=-1))
        return dh
