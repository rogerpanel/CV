"""
Adversarial robustness for encrypted traffic classification

Implements protocol-constrained adversarial evaluation and certified
robustness via randomized smoothing with protocol-aware enhancements.

Reference: Paper Section 3.4 - Protocol-Aware Robustness
"""

from .protocol_aware_robustness import (
    ProtocolConstraintChecker,
    RandomizedSmoothing,
    evaluate_certified_robustness
)

__all__ = [
    'ProtocolConstraintChecker',
    'RandomizedSmoothing',
    'evaluate_certified_robustness',
]
