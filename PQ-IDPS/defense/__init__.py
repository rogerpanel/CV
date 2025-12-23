"""
Adversarial Defense Mechanisms

Implements quantum noise injection, randomized smoothing, and Lipschitz constraints.
"""

from .lipschitz_constraints import (
    SpectralNormalization,
    LipschitzLinear,
    LipschitzConv1d,
    compute_lipschitz_constant
)
from .adversarial_defense import (
    QuantumNoiseInjection,
    RandomizedSmoothing,
    AdversarialTrainer
)

__all__ = [
    'SpectralNormalization',
    'LipschitzLinear',
    'LipschitzConv1d',
    'compute_lipschitz_constant',
    'QuantumNoiseInjection',
    'RandomizedSmoothing',
    'AdversarialTrainer'
]
