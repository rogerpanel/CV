"""
Few-shot learning for zero-day encrypted attack detection

Modules:
- prototypical: Prototypical Networks for metric-based few-shot learning
- maml: Model-Agnostic Meta-Learning for optimization-based few-shot learning

Reference: Paper Section 3.7 - Few-Shot Zero-Day Detection
"""

from .prototypical import PrototypicalNetwork, PrototypicalTrainer
from .maml import MAML, MAMLTrainer

__all__ = [
    'PrototypicalNetwork', 'PrototypicalTrainer',
    'MAML', 'MAMLTrainer',
]
