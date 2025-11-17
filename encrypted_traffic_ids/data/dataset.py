"""
PyTorch Dataset for encrypted traffic data

Implements custom Dataset class for efficient loading and batching
of encrypted traffic features during training.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple, Optional


class EncryptedTrafficDataset(Dataset):
    """
    PyTorch Dataset for encrypted traffic intrusion detection.

    Handles both temporal features (for CNN/LSTM) and statistical features
    (for traditional ML components), supporting the hybrid architecture
    described in the paper.
    """

    def __init__(
        self,
        temporal_features: np.ndarray,
        statistical_features: Optional[np.ndarray] = None,
        labels: np.ndarray = None,
        transform: Optional[callable] = None
    ):
        """
        Initialize dataset.

        Args:
            temporal_features: Temporal feature array (N, seq_len, num_features)
            statistical_features: Statistical feature array (N, num_stat_features)
            labels: Label array (N,)
            transform: Optional transform to apply to features
        """
        self.temporal_features = torch.FloatTensor(temporal_features)
        self.statistical_features = None
        if statistical_features is not None:
            self.statistical_features = torch.FloatTensor(statistical_features)

        self.labels = None
        if labels is not None:
            if labels.dtype in [np.float32, np.float64]:
                self.labels = torch.FloatTensor(labels)
            else:
                self.labels = torch.LongTensor(labels)

        self.transform = transform

        # Validate dimensions
        assert len(self.temporal_features) == len(self.labels) if self.labels is not None else True, \
            "Features and labels must have same length"

        if self.statistical_features is not None:
            assert len(self.temporal_features) == len(self.statistical_features), \
                "Temporal and statistical features must have same length"

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.temporal_features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        Get item by index.

        Args:
            idx: Sample index

        Returns:
            Tuple of (temporal_features, statistical_features, label)
            If statistical_features or labels are None, they are omitted from output
        """
        temporal = self.temporal_features[idx]

        if self.transform is not None:
            temporal = self.transform(temporal)

        # Build output tuple
        output = [temporal]

        if self.statistical_features is not None:
            output.append(self.statistical_features[idx])

        if self.labels is not None:
            output.append(self.labels[idx])

        return tuple(output) if len(output) > 1 else output[0]

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for handling imbalanced datasets.

        Returns:
            Tensor of class weights
        """
        if self.labels is None:
            raise ValueError("Cannot compute class weights without labels")

        unique_classes, class_counts = torch.unique(self.labels, return_counts=True)
        total_samples = len(self.labels)
        num_classes = len(unique_classes)

        # Inverse frequency weighting: weight = N / (num_classes * count)
        class_weights = total_samples / (num_classes * class_counts.float())

        return class_weights

    def get_class_distribution(self) -> dict:
        """
        Get class distribution statistics.

        Returns:
            Dictionary with class distribution information
        """
        if self.labels is None:
            raise ValueError("Cannot compute class distribution without labels")

        unique_classes, class_counts = torch.unique(self.labels, return_counts=True)
        total_samples = len(self.labels)

        distribution = {}
        for cls, count in zip(unique_classes.numpy(), class_counts.numpy()):
            distribution[int(cls)] = {
                'count': int(count),
                'percentage': float(count / total_samples * 100)
            }

        return distribution


class MultiDatasetWrapper(Dataset):
    """
    Wrapper for handling multiple datasets simultaneously.

    Useful for multi-task learning or domain adaptation scenarios
    where models train on multiple encrypted traffic datasets.
    """

    def __init__(self, datasets: list):
        """
        Initialize multi-dataset wrapper.

        Args:
            datasets: List of EncryptedTrafficDataset instances
        """
        self.datasets = datasets
        self.cumulative_sizes = [0]

        for dataset in datasets:
            self.cumulative_sizes.append(
                self.cumulative_sizes[-1] + len(dataset)
            )

    def __len__(self) -> int:
        """Return total size across all datasets."""
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx: int):
        """Get item from appropriate dataset."""
        # Find which dataset this index belongs to
        dataset_idx = 0
        for i, cumsum in enumerate(self.cumulative_sizes[1:]):
            if idx < cumsum:
                dataset_idx = i
                break

        # Get local index within that dataset
        local_idx = idx - self.cumulative_sizes[dataset_idx]

        return self.datasets[dataset_idx][local_idx]
