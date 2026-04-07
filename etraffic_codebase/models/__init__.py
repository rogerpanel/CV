"""
Deep learning models for encrypted traffic intrusion detection

This package implements all architectures described in the paper:
"Hybrid Spatial-Temporal Deep Learning for Privacy-Preserving
 Encrypted Traffic Intrusion Detection" (EAAI 2025)

Architectures:
- Hybrid CNN-LSTM for spatial-temporal modeling
- Transformer-based models (TransECA-Net, FlowTransformer)
- Graph Neural Networks for topology analysis
- Ensemble aggregation mechanisms
"""

from .cnn_lstm import HybridCNNLSTM
from .transformer import TransECANet, FlowTransformer
from .gnn import GraphSAGENet
from .ensemble import EnsembleClassifier
from .base import BaseModel

__all__ = [
    'HybridCNNLSTM',
    'TransECANet',
    'FlowTransformer',
    'GraphSAGENet',
    'EnsembleClassifier',
    'BaseModel'
]
