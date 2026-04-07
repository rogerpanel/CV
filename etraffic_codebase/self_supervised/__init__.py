"""
Self-supervised pretraining for encrypted traffic representations

Implements InfoNCE contrastive learning that bootstraps representations
from unlabeled encrypted traffic, improving few-shot zero-day detection
by 7.3 percentage points over random initialization.

Reference: Paper Section 3.6 - Self-Supervised Pretraining
"""

from .contrastive import (
    InfoNCELoss,
    ContrastiveEncoder,
    ContrastivePretrainer,
    TrafficAugmentation,
)

__all__ = [
    'InfoNCELoss',
    'ContrastiveEncoder',
    'ContrastivePretrainer',
    'TrafficAugmentation',
]
