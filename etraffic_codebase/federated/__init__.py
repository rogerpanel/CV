"""
Federated learning for privacy-preserving encrypted traffic IDS

Modules:
- fedavg: Standard Federated Averaging algorithm
- tabf: Traffic-Aware Byzantine Filtering (Algorithm 1 from paper)
- differential_privacy: (epsilon, delta)-DP mechanisms
- aggregation: Advanced aggregation strategies

Reference: Paper Section 3.5 - Byzantine-Resilient Federated Learning
"""

from .fedavg import FederatedClient, FederatedServer, federated_training
from .tabf import TABFAggregator, tabf_federated_training
from .differential_privacy import DifferentialPrivacy
from .aggregation import (
    aggregate_models, fedavg_aggregate, median_aggregate,
    gradient_similarity_aggregation
)

__all__ = [
    'FederatedClient', 'FederatedServer', 'federated_training',
    'TABFAggregator', 'tabf_federated_training',
    'DifferentialPrivacy',
    'aggregate_models', 'fedavg_aggregate', 'median_aggregate',
    'gradient_similarity_aggregation',
]
