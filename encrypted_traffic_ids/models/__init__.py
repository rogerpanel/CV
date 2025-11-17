"""
Deep learning models for encrypted traffic intrusion detection

This package implements all architectures described in the paper:
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
