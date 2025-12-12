"""
Few-Shot Learning for Zero-Day Attack Detection

This module implements meta-learning approaches for detecting novel attack types
with limited labeled samples from encrypted traffic.

Components:
- Prototypical Networks: Distance-based classification in embedding space
- MAML (Model-Agnostic Meta-Learning): Fast adaptation through meta-gradients

References:
    Paper Section 3.5 - Few-Shot Meta-Learning
    Snell et al. (2017) - Prototypical Networks for Few-shot Learning
    Finn et al. (2017) - Model-Agnostic Meta-Learning for Fast Adaptation
"""

from .prototypical import PrototypicalNetwork, PrototypicalTrainer
from .maml import MAML, MAMLTrainer

__all__ = [
    'PrototypicalNetwork',
    'PrototypicalTrainer',
    'MAML',
    'MAMLTrainer'
]
