"""
Data loading and preprocessing for encrypted traffic datasets

This package handles:
- Loading various encrypted traffic datasets
- Feature extraction from packet-level metadata
- Data preprocessing and normalization
- Train/validation/test splitting with stratification
"""

from .preprocessing import (
    FlowFeatureExtractor,
    preprocess_dataset,
    split_dataset
)
from .dataset import EncryptedTrafficDataset
from .loaders import create_dataloaders

__all__ = [
    'FlowFeatureExtractor',
    'preprocess_dataset',
    'split_dataset',
    'EncryptedTrafficDataset',
    'create_dataloaders'
]
