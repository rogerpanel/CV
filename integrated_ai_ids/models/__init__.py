"""
Integrated AI-IDS Model Implementations
========================================

All dissertation models implemented in PyTorch:
- Neural ODE with Temporal Adaptive Batch Normalization
- Privacy-Preserving Federated Optimal Transport
- Encrypted Traffic Analyzer (CNN-LSTM-Transformer)
- Federated Graph Temporal Dynamics
- Heterogeneous Graph Pooling
- Bayesian Uncertainty Quantification

Author: Roger Nick Anaedevha
"""

from .neural_ode import TemporalAdaptiveNeuralODE, PointProcessNeuralODE
from .optimal_transport import PPFOTDetector, SinkhornDistance
from .encrypted_traffic import EncryptedTrafficAnalyzer
from .federated_graph import FedGTDModel, GraphTemporalODE
from .heterogeneous_graph import HGPModel, HierarchicalPooling
from .bayesian_inference import BayesianUncertaintyNet, StructuredVariationalInference

__all__ = [
    'TemporalAdaptiveNeuralODE',
    'PointProcessNeuralODE',
    'PPFOTDetector',
    'SinkhornDistance',
    'EncryptedTrafficAnalyzer',
    'FedGTDModel',
    'GraphTemporalODE',
    'HGPModel',
    'HierarchicalPooling',
    'BayesianUncertaintyNet',
    'StructuredVariationalInference'
]
