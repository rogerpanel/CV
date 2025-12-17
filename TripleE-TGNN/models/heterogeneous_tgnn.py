"""Heterogeneous Temporal GNN for cross-granularity integration."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class HeterogeneousTemporalGNN(nn.Module):
    """
    Integrates service, trace, and node embeddings through heterogeneous graph.

    Node types: services, traces, pods
    Edge types: service-call, trace-span, pod-deploy, pod-comm
    """

    def __init__(self, embedding_dim: int = 512, hidden_dim: int = 256, num_layers: int = 2):
        super().__init__()

        self.num_layers = num_layers

        # Type-specific message passing (simplified - full version uses HeteroConv)
        self.service_layers = nn.ModuleList([
            nn.Linear(embedding_dim, hidden_dim) for _ in range(num_layers)
        ])

        self.trace_layers = nn.ModuleList([
            nn.Linear(embedding_dim, hidden_dim) for _ in range(num_layers)
        ])

        self.node_layers = nn.ModuleList([
            nn.Linear(embedding_dim, hidden_dim) for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, embedding_dim)

    def forward(
        self,
        hetero_batch,
        service_emb: torch.Tensor,
        trace_emb: torch.Tensor,
        node_emb: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Integrate embeddings through heterogeneous graph.

        Args:
            hetero_batch: PyG HeteroData batch
            service_emb, trace_emb, node_emb: Initial embeddings

        Returns:
            Dictionary with updated embeddings
        """
        h_service = service_emb
        h_trace = trace_emb
        h_node = node_emb

        for layer in range(self.num_layers):
            # Message passing (simplified - use actual graph structure in full version)
            h_service = torch.relu(self.service_layers[layer](h_service))
            h_trace = torch.relu(self.trace_layers[layer](h_trace))
            h_node = torch.relu(self.node_layers[layer](h_node))

        # Project back to embedding dimension
        h_service = self.output_proj(h_service)
        h_trace = self.output_proj(h_trace)
        h_node = self.output_proj(h_node)

        return {
            'service': h_service,
            'trace': h_trace,
            'node': h_node
        }
