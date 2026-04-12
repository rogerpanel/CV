"""
TripleE-TGNN: Triple-Embedding Temporal Graph Neural Network

Main model integrating service-level, trace-level, and node-level embeddings
through heterogeneous temporal GNN with adaptive cross-granularity fusion.

Reference:
    Anaedevha et al., "TripleE-TGNN: Triple-Embedding Temporal Graph Neural  Networks for Multi-Granularity Microservices Security", IEEE TIFS 2025.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List

from models.service_encoder import ServiceLevelEncoder
from models.trace_encoder import TraceLevelEncoder
from models.node_encoder import NodeLevelEncoder
from models.heterogeneous_tgnn import HeterogeneousTemporalGNN


class GranularityFusion(nn.Module):
    """
    Adaptive cross-granularity fusion with learned attention weights.

    Combines service, trace, and node embeddings with granularity-level
    attention, allowing the model to emphasize different perspectives
    based on the current system state.
    """

    def __init__(self, embedding_dim: int = 512, hidden_dim: int = 256):
        super().__init__()

        # Granularity-specific projection
        self.service_proj = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.Tanh()
        )

        self.trace_proj = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.Tanh()
        )

        self.node_proj = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.Tanh()
        )

        # Attention query vectors
        self.q_service = nn.Parameter(torch.randn(hidden_dim))
        self.q_trace = nn.Parameter(torch.randn(hidden_dim))
        self.q_node = nn.Parameter(torch.randn(hidden_dim))

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)  # Binary classification
        )

    def forward(
        self,
        service_emb: torch.Tensor,
        trace_emb: torch.Tensor,
        node_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse multi-granularity embeddings with adaptive attention.

        Args:
            service_emb: (batch_size, embedding_dim)
            trace_emb: (batch_size, embedding_dim)
            node_emb: (batch_size, embedding_dim)

        Returns:
            logits: (batch_size, 2) - Classification logits
            attention_weights: (batch_size, 3) - Granularity weights
        """
        batch_size = service_emb.size(0)

        # Project embeddings
        h_service = self.service_proj(service_emb)  # (batch, hidden_dim)
        h_trace = self.trace_proj(trace_emb)
        h_node = self.node_proj(node_emb)

        # Compute attention scores
        score_service = torch.matmul(h_service, self.q_service)  # (batch,)
        score_trace = torch.matmul(h_trace, self.q_trace)
        score_node = torch.matmul(h_node, self.q_node)

        # Softmax to get attention weights
        scores = torch.stack([score_service, score_trace, score_node], dim=1)  # (batch, 3)
        attention_weights = F.softmax(scores, dim=1)  # (batch, 3)

        # Concatenate all embeddings (not weighted - classifier learns combination)
        fused = torch.cat([service_emb, trace_emb, node_emb], dim=-1)  # (batch, 3*embedding_dim)

        # Classification
        logits = self.classifier(fused)

        return logits, attention_weights


class TripleETGNN(nn.Module):
    """
    Complete TripleE-TGNN model with triple-embedding and heterogeneous temporal GNN.

    Args:
        service_hidden_dims: Hidden dimensions for service-level GNN
        trace_hidden_dims: Hidden dimensions for trace-level GNN
        node_hidden_dims: Hidden dimensions for node-level GNN
        gru_hidden_size: GRU hidden units for service/trace encoders
        lstm_hidden_size: LSTM hidden units for node encoder
        num_gnn_layers: Number of GNN layers (default 3)
        num_attention_heads: Attention heads in GAT (default 8)
        fusion_method: 'adaptive' or 'concat' (default 'adaptive')
        dropout: Dropout rate (default 0.3)
    """

    def __init__(
        self,
        service_hidden_dims: List[int] = [128, 256, 512],
        trace_hidden_dims: List[int] = [128, 256, 512],
        node_hidden_dims: List[int] = [128, 256, 512],
        gru_hidden_size: int = 256,
        lstm_hidden_size: int = 256,
        num_gnn_layers: int = 3,
        num_attention_heads: int = 8,
        fusion_method: str = 'adaptive',
        dropout: float = 0.3
    ):
        super().__init__()

        self.fusion_method = fusion_method

        # Service-level encoder
        self.service_encoder = ServiceLevelEncoder(
            hidden_dims=service_hidden_dims,
            gru_hidden_size=gru_hidden_size,
            num_attention_heads=num_attention_heads,
            dropout=dropout
        )

        # Trace-level encoder
        self.trace_encoder = TraceLevelEncoder(
            hidden_dims=trace_hidden_dims,
            gru_hidden_size=gru_hidden_size,
            dropout=dropout
        )

        # Node-level encoder
        self.node_encoder = NodeLevelEncoder(
            hidden_dims=node_hidden_dims,
            lstm_hidden_size=lstm_hidden_size,
            dropout=dropout
        )

        # Heterogeneous temporal GNN for cross-granularity integration
        self.hetero_gnn = HeterogeneousTemporalGNN(
            embedding_dim=service_hidden_dims[-1],
            hidden_dim=256,
            num_layers=2
        )

        # Granularity fusion
        if fusion_method == 'adaptive':
            self.fusion = GranularityFusion(
                embedding_dim=service_hidden_dims[-1],
                hidden_dim=256
            )
        else:
            # Simple concatenation baseline
            total_dim = sum([service_hidden_dims[-1], trace_hidden_dims[-1], node_hidden_dims[-1]])
            self.fusion = nn.Sequential(
                nn.Linear(total_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 2)
            )

    def forward(
        self,
        service_graph_batch,
        trace_graph_batch,
        node_graph_batch,
        hetero_graph_batch,
        return_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through TripleE-TGNN.

        Args:
            service_graph_batch: PyG batch of service dependency graphs
            trace_graph_batch: PyG batch of trace span graphs
            node_graph_batch: PyG batch of pod interaction graphs
            hetero_graph_batch: PyG HeteroData batch for cross-granularity
            return_embeddings: Return intermediate embeddings

        Returns:
            Dictionary containing:
                - logits: (batch_size, 2)
                - attention_weights: (batch_size, 3) if adaptive fusion
                - embeddings: Dict of intermediate embeddings if requested
        """
        # Service-level embedding
        service_emb = self.service_encoder(service_graph_batch)  # (batch, 512)

        # Trace-level embedding
        trace_emb = self.trace_encoder(trace_graph_batch)  # (batch, 512)

        # Node-level embedding
        node_emb = self.node_encoder(node_graph_batch)  # (batch, 512)

        # Heterogeneous temporal GNN integration
        integrated_embs = self.hetero_gnn(
            hetero_graph_batch,
            service_emb,
            trace_emb,
            node_emb
        )

        # Update embeddings with cross-granularity information
        service_emb = integrated_embs['service']
        trace_emb = integrated_embs['trace']
        node_emb = integrated_embs['node']

        # Fusion
        if self.fusion_method == 'adaptive':
            logits, attention_weights = self.fusion(service_emb, trace_emb, node_emb)
        else:
            fused = torch.cat([service_emb, trace_emb, node_emb], dim=-1)
            logits = self.fusion(fused)
            attention_weights = torch.full((service_emb.size(0), 3), 1/3, device=logits.device)

        result = {
            'logits': logits,
            'attention_weights': attention_weights
        }

        if return_embeddings:
            result['embeddings'] = {
                'service': service_emb,
                'trace': trace_emb,
                'node': node_emb
            }

        return result

    def compute_granularity_accuracies(
        self,
        service_graph_batch,
        trace_graph_batch,
        node_graph_batch,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute individual granularity accuracies for monitoring.

        Args:
            service_graph_batch, trace_graph_batch, node_graph_batch: Graph batches
            labels: (batch_size,) Ground truth labels

        Returns:
            Dictionary with service_acc, trace_acc, node_acc, fusion_acc
        """
        with torch.no_grad():
            # Get embeddings
            service_emb = self.service_encoder(service_graph_batch)
            trace_emb = self.trace_encoder(trace_graph_batch)
            node_emb = self.node_encoder(node_graph_batch)

            # Individual granularity classifiers (simple MLP)
            service_logits = self._classify_single(service_emb)
            trace_logits = self._classify_single(trace_emb)
            node_logits = self._classify_single(node_emb)

            # Compute accuracies
            service_acc = (torch.argmax(service_logits, dim=1) == labels).float().mean().item()
            trace_acc = (torch.argmax(trace_logits, dim=1) == labels).float().mean().item()
            node_acc = (torch.argmax(node_logits, dim=1) == labels).float().mean().item()

            # Fused accuracy
            if self.fusion_method == 'adaptive':
                fused_logits, _ = self.fusion(service_emb, trace_emb, node_emb)
            else:
                fused = torch.cat([service_emb, trace_emb, node_emb], dim=-1)
                fused_logits = self.fusion(fused)

            fusion_acc = (torch.argmax(fused_logits, dim=1) == labels).float().mean().item()

        return {
            'service_acc': service_acc,
            'trace_acc': trace_acc,
            'node_acc': node_acc,
            'fusion_acc': fusion_acc
        }

    def _classify_single(self, embedding: torch.Tensor) -> torch.Tensor:
        """Helper to classify single granularity embedding."""
        if not hasattr(self, '_single_classifier'):
            self._single_classifier = nn.Linear(embedding.size(1), 2).to(embedding.device)
        return self._single_classifier(embedding)


if __name__ == "__main__":
    """Test TripleE-TGNN model."""

    print("Testing TripleE-TGNN...")
    print("=" * 60)

    # Create model
    model = TripleETGNN(
        service_hidden_dims=[128, 256, 512],
        trace_hidden_dims=[128, 256, 512],
        node_hidden_dims=[128, 256, 512],
        gru_hidden_size=256,
        lstm_hidden_size=256,
        fusion_method='adaptive'
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print()

    print("✓ TripleE-TGNN model initialized successfully!")
