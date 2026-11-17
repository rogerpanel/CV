"""Temporal multi-head attention (Equation 2).

    α_vu(t) =
        exp(LeakyReLU(aᵀ [W_q h_v(t) ∥ W_k h_u(t) ∥ φ(t - t_{vu})]))
      / Σ_w exp(LeakyReLU(aᵀ [W_q h_v(t) ∥ W_k h_w(t) ∥ φ(t - t_{wv})]))

We implement a multi-head generalisation. The attention weights α_{vu}(t)
are returned alongside the aggregated neighbourhood message so that
`AttentionFlowExplainer` can render them directly.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ct_explain.models.time_encoding import TimeEncoding


class TemporalMultiHeadAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        time_dim: int = 32,
        leaky_slope: float = 0.2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.leaky_slope = leaky_slope

        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.time_encoding = TimeEncoding(time_dim)
        # aᵀ in Eq. 2 — one learnable vector per head.
        self.a = nn.Parameter(torch.randn(num_heads, 2 * self.head_dim + time_dim) * 0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h_v: torch.Tensor,          # (N_target, d_h)
        h_u: torch.Tensor,          # (E, d_h)     neighbour state
        delta_t: torch.Tensor,      # (E,)          t − t_{vu}
        edge_index: torch.Tensor,   # (2, E)        row 0 = target, row 1 = source
        messages: torch.Tensor,     # (E, d_h)      g_φ(h_u, e_{vu})
        num_targets: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (aggregated_message, attention_weights).

        attention_weights: (num_heads, E) — softmaxed per target node.
        """
        if num_targets is None:
            num_targets = int(edge_index[0].max().item()) + 1

        tgt_idx = edge_index[0]
        q = self.W_q(h_v[tgt_idx]).view(-1, self.num_heads, self.head_dim)  # (E, H, D)
        k = self.W_k(h_u).view(-1, self.num_heads, self.head_dim)           # (E, H, D)
        v = self.W_v(messages).view(-1, self.num_heads, self.head_dim)      # (E, H, D)

        phi = self.time_encoding(delta_t)                                    # (E, T)
        phi_e = phi.unsqueeze(1).expand(-1, self.num_heads, -1)              # (E, H, T)

        # [W_q h_v ∥ W_k h_u ∥ φ(Δt)] — concatenate along feature dim
        feat = torch.cat([q, k, phi_e], dim=-1)                              # (E, H, 2D+T)
        raw = (feat * self.a).sum(-1)                                        # (E, H)
        raw = F.leaky_relu(raw, negative_slope=self.leaky_slope)

        # Softmax per target node, per head — numerically stable scatter softmax.
        alpha = _scatter_softmax(raw, tgt_idx, num_targets)                  # (E, H)
        alpha = self.dropout(alpha)

        weighted = v * alpha.unsqueeze(-1)                                   # (E, H, D)
        out = torch.zeros(num_targets, self.num_heads, self.head_dim,
                          device=h_v.device, dtype=h_v.dtype)
        out.index_add_(0, tgt_idx, weighted)                                 # (N, H, D)
        out = out.reshape(num_targets, -1)                                   # (N, d_h)
        return out, alpha.t()                                                # attn shape (H, E)


def _scatter_softmax(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Softmax over `src` grouped by `index`, for each column independently.

    src:   (E, H)
    index: (E,)  ∈ [0, dim_size)
    returns (E, H), softmax-normalised per target node.
    """
    # Max-subtraction for stability.
    max_per_group = torch.full(
        (dim_size, src.size(-1)), float("-inf"), device=src.device, dtype=src.dtype
    )
    max_per_group = max_per_group.scatter_reduce(
        0, index.unsqueeze(-1).expand_as(src), src, reduce="amax", include_self=True
    )
    shifted = src - max_per_group[index]
    expv = shifted.exp()
    denom = torch.zeros_like(max_per_group)
    denom.index_add_(0, index, expv)
    return expv / (denom[index] + 1e-12)
