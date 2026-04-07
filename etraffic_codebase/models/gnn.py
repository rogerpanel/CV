"""
Graph Neural Network models for encrypted traffic topology analysis

Implements GNN architectures for detecting coordinated multi-flow attacks
(e.g., DDoS, lateral movement) across encrypted connections.

Models:
- GraphSAGENet: GraphSAGE-based model (96.8% on UNSW-NB15)
- GATNet: Graph Attention Network variant

References:
    Paper Section 3.3 - Graph Neural Network Branch
    Lin et al. (2023) - E-GRACL achieving 96.8% accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .base import BaseModel

try:
    from torch_geometric.nn import SAGEConv, GATConv, global_mean_pool, global_max_pool
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False


class GlobalAttentionPooling(nn.Module):
    """
    Global attention pooling for graph-level representations.

    Learns to weight node features based on their relevance
    for classification tasks.
    """

    def __init__(self, in_channels: int):
        super(GlobalAttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.Tanh(),
            nn.Linear(in_channels // 2, 1)
        )

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        attn_weights = self.attention(x)
        attn_weights = torch.softmax(attn_weights, dim=0)
        weighted = x * attn_weights

        # Scatter sum by batch
        unique_batches = torch.unique(batch)
        pooled = torch.zeros(
            len(unique_batches), x.size(1), device=x.device
        )
        for i, b in enumerate(unique_batches):
            mask = (batch == b)
            pooled[i] = weighted[mask].sum(dim=0)

        return pooled


class GraphSAGENet(BaseModel):
    """
    GraphSAGE-based model for encrypted traffic topology analysis.

    Captures inter-flow relationships and coordinated attack patterns
    across multiple encrypted connections.

    Architecture:
    1. Multiple GraphSAGE convolution layers
    2. Batch normalization + dropout
    3. Global pooling (mean + max + attention)
    4. Classification head

    Performance: 96.8% accuracy on UNSW-NB15

    Reference: Paper Section 3.3, Lin et al. (2023) E-GRACL
    """

    def __init__(self, input_dim: int = 88, num_classes: int = 6,
                 hidden_channels: int = 128, num_layers: int = 3,
                 dropout: float = 0.2, use_attention_pool: bool = True):
        super(GraphSAGENet, self).__init__()

        if not HAS_TORCH_GEOMETRIC:
            raise ImportError(
                "torch_geometric is required for GNN models. "
                "Install with: pip install torch-geometric"
            )

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.use_attention_pool = use_attention_pool

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # First layer
        self.convs.append(SAGEConv(input_dim, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.dropout = nn.Dropout(dropout)

        if use_attention_pool:
            self.attn_pool = GlobalAttentionPooling(hidden_channels)
            pool_dim = hidden_channels * 3  # mean + max + attention
        else:
            pool_dim = hidden_channels * 2  # mean + max

        self.classifier = nn.Sequential(
            nn.Linear(pool_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Node features (num_nodes, input_dim)
            edge_index: Graph connectivity (2, num_edges)
            batch: Batch assignment vector (num_nodes,)

        Returns:
            Class logits (batch_size, num_classes)
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Graph convolution layers
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x, inplace=True)
            x = self.dropout(x)

        # Global pooling
        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)

        if self.use_attention_pool:
            attn_pool = self.attn_pool(x, batch)
            graph_feat = torch.cat([mean_pool, max_pool, attn_pool], dim=-1)
        else:
            graph_feat = torch.cat([mean_pool, max_pool], dim=-1)

        # Classification
        logits = self.classifier(graph_feat)
        return logits

    def get_config(self) -> dict:
        return {
            'input_dim': self.input_dim,
            'num_classes': self.num_classes
        }


class GATNet(BaseModel):
    """
    Graph Attention Network for encrypted traffic analysis.

    Uses attention mechanisms to weight neighboring node importance,
    offering improved performance on heterogeneous graph structures.
    """

    def __init__(self, input_dim: int = 88, num_classes: int = 6,
                 hidden_channels: int = 128, num_layers: int = 3,
                 heads: int = 4, dropout: float = 0.2):
        super(GATNet, self).__init__()

        if not HAS_TORCH_GEOMETRIC:
            raise ImportError(
                "torch_geometric is required for GNN models. "
                "Install with: pip install torch-geometric"
            )

        self.input_dim = input_dim
        self.num_classes = num_classes

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(GATConv(input_dim, hidden_channels, heads=heads))
        self.bns.append(nn.BatchNorm1d(hidden_channels * heads))

        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels * heads, hidden_channels, heads=heads)
            )
            self.bns.append(nn.BatchNorm1d(hidden_channels * heads))

        # Final layer: single head
        self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=1))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.elu(x, inplace=True)
            x = self.dropout(x)

        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        graph_feat = torch.cat([mean_pool, max_pool], dim=-1)

        logits = self.classifier(graph_feat)
        return logits

    def get_config(self) -> dict:
        return {
            'input_dim': self.input_dim,
            'num_classes': self.num_classes
        }
