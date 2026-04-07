"""
Data loading and preprocessing for encrypted traffic analysis

Modules:
- preprocessing: Feature extraction from encrypted traffic flows
- dataset: PyTorch Dataset implementations
- dataset_loaders: Per-dataset loading logic (CICIDS, UNSW-NB15, etc.)
- loaders: DataLoader creation utilities
"""

from .preprocessing import FlowFeatureExtractor, preprocess_dataset, split_dataset
from .dataset import EncryptedTrafficDataset, MultiDatasetWrapper
from .dataset_loaders import DatasetFactory, load_all_datasets
from .loaders import create_dataloaders, create_inference_dataloader

__all__ = [
    'FlowFeatureExtractor',
    'preprocess_dataset',
    'split_dataset',
    'EncryptedTrafficDataset',
    'MultiDatasetWrapper',
    'DatasetFactory',
    'load_all_datasets',
    'create_dataloaders',
    'create_inference_dataloader',
]
