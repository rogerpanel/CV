"""E-GraphSAGE edge encoder.

Lo et al. (2022), "E-GraphSAGE: A Graph Neural Network Based Intrusion
Detection System for IoT". Flow records are treated as edges of a bipartite
graph between source and destination endpoints; the encoder produces 128-d
edge embeddings via gated residual aggregation.

Without an explicit graph topology (tabular CSV flows) the encoder reduces to a
per-flow residual MLP over the 83-dim feature vector.

``lipschitz=True`` builds the certifiable variant required by the mean-predictor
certificate: every linear layer is spectral-normalised, the input-dependent gate
is replaced by a fixed convex combination, and LayerNorm is removed (LayerNorm
has no global Lipschitz bound). ``lipschitz_bound()`` then returns a sound upper
bound on the encoder's l2 Lipschitz constant. In graph mode the bound holds with
the other flows of the micro-graph held fixed.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm

try:
    from torch_geometric.nn import SAGEConv
    _HAS_PYG = True
except ImportError:                                          # pragma: no cover
    _HAS_PYG = False

GELU_LIPSCHITZ = 1.1289  # max_x |d/dx GELU(x)|, attained at x = sqrt(2)
FIXED_GATE = 0.5


def spectral_norm_exact(weight: torch.Tensor) -> float:
    return float(torch.linalg.matrix_norm(weight.detach().float(), ord=2))


class _AttentionGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, h: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        alpha = torch.sigmoid(self.score(h))
        return alpha * h + (1 - alpha) * residual


class EGraphSAGE(nn.Module):
    def __init__(self, edge_features: int = 83, hidden_dim: int = 128,
                 num_layers: int = 3, dropout: float = 0.10,
                 lipschitz: bool = False):
        super().__init__()
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.lipschitz = lipschitz
        sn = spectral_norm if lipschitz else (lambda m: m)
        self.input_proj = sn(nn.Linear(edge_features, hidden_dim))

        self.layers = nn.ModuleList()
        self.gates = nn.ModuleList()
        for _ in range(num_layers):
            if _HAS_PYG and not lipschitz:
                self.layers.append(SAGEConv(hidden_dim, hidden_dim, aggr="mean"))
            else:
                self.layers.append(sn(nn.Linear(hidden_dim, hidden_dim)))
            self.gates.append(nn.Identity() if lipschitz else _AttentionGate(hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.Identity() if lipschitz else nn.LayerNorm(hidden_dim)

    def forward(self, edge_attr: torch.Tensor,
                edge_index: torch.Tensor | None = None,
                node_feat: torch.Tensor | None = None) -> torch.Tensor:
        h = F.gelu(self.input_proj(edge_attr))
        for layer, gate in zip(self.layers, self.gates):
            if _HAS_PYG and not self.lipschitz and edge_index is not None and node_feat is not None:
                node_out = layer(node_feat, edge_index)
                src, dst = edge_index
                h_new = F.gelu(node_out[src] + node_out[dst])
            else:
                h_new = F.gelu(layer(h))
            if self.lipschitz:
                h = FIXED_GATE * h_new + (1 - FIXED_GATE) * h
            else:
                h = gate(h_new, h)
            h = self.dropout(h)
        return self.norm(h)

    def lipschitz_bound(self) -> float:
        """Upper bound on sup_x ||h(x+d) - h(x)|| / ||d|| (eval mode)."""
        if not self.lipschitz:
            raise ValueError("Encoder was built with lipschitz=False (LayerNorm and "
                             "input-dependent gates have no global Lipschitz bound).")
        bound = GELU_LIPSCHITZ * spectral_norm_exact(self.input_proj.weight)
        for layer in self.layers:
            bound *= FIXED_GATE * GELU_LIPSCHITZ * spectral_norm_exact(layer.weight) + (1 - FIXED_GATE)
        return bound
