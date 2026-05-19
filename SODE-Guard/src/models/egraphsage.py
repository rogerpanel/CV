"""E-GraphSAGE edge encoder.

Lopez-Martin et al. (2020), "E-GraphSAGE: A Graph Neural Network Based Intrusion
Detection System for IoT". We treat flow records as **edges** of a bipartite
graph between source and destination endpoints. The encoder produces 128-d
edge embeddings via attention-gated residual aggregation.

In the absence of an explicit graph topology (e.g. tabular CSV flows), the
encoder degenerates to a feed-forward MLP over the 83-dim feature vector,
which is the standard "tabular-flow" mode used by the manuscript's Table 2.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import SAGEConv
    _HAS_PYG = True
except ImportError:                                          # pragma: no cover
    _HAS_PYG = False


class _AttentionGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, h: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        alpha = torch.sigmoid(self.score(h))
        return alpha * h + (1 - alpha) * residual


class EGraphSAGE(nn.Module):
    def __init__(self, edge_features: int = 83, hidden_dim: int = 128,
                 num_layers: int = 3, dropout: float = 0.10):
        super().__init__()
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(edge_features, hidden_dim)

        self.layers = nn.ModuleList()
        self.gates = nn.ModuleList()
        for _ in range(num_layers):
            if _HAS_PYG:
                self.layers.append(SAGEConv(hidden_dim, hidden_dim, aggr="mean"))
            else:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.gates.append(_AttentionGate(hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, edge_attr: torch.Tensor,
                edge_index: torch.Tensor | None = None,
                node_feat: torch.Tensor | None = None) -> torch.Tensor:
        h = F.gelu(self.input_proj(edge_attr))
        for layer, gate in zip(self.layers, self.gates):
            if _HAS_PYG and edge_index is not None and node_feat is not None:
                # Standard PyG forward; we then return edge embeddings as the
                # concatenation of (source, dest) node states.
                node_out = layer(node_feat, edge_index)
                src, dst = edge_index
                h_new = F.gelu(node_out[src] + node_out[dst])
            else:
                h_new = F.gelu(layer(h))
            h = gate(h_new, h)
            h = self.dropout(h)
        return self.norm(h)
