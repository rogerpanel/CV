"""
Experience Replay Buffer with Reservoir Sampling.

Implements Vitter's Algorithm R for maintaining a bounded replay buffer
that ensures each previously seen sample has equal probability of
residing in the buffer regardless of arrival time.

From Section IV-A: Buffer capacity M = 5,000.
"""

import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ReservoirReplayBuffer:
    """
    Reservoir-sampled experience replay buffer (Algorithm R, Vitter 1985).

    Maintains a fixed-size buffer from a stream of samples, ensuring
    uniform sampling probability across all observed samples.

    Args:
        capacity: Maximum number of samples to store (M = 5000).
        feature_dim: Dimensionality of feature vectors.
    """

    def __init__(self, capacity: int = 5000, feature_dim: int = 79):
        self.capacity = capacity
        self.feature_dim = feature_dim
        self.features = np.zeros((capacity, feature_dim), dtype=np.float32)
        self.labels = np.zeros(capacity, dtype=np.int64)
        self.count = 0         # Total samples seen
        self.current_size = 0  # Current buffer occupancy

    def add(self, feature: np.ndarray, label: int) -> None:
        """Add a single sample using reservoir sampling."""
        if self.current_size < self.capacity:
            self.features[self.current_size] = feature
            self.labels[self.current_size] = label
            self.current_size += 1
        else:
            # Vitter's Algorithm R: replace with probability M/n
            j = np.random.randint(0, self.count + 1)
            if j < self.capacity:
                self.features[j] = feature
                self.labels[j] = label
        self.count += 1

    def add_batch(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Add a batch of samples using reservoir sampling."""
        for i in range(len(labels)):
            self.add(features[i], labels[i])

    def sample(
        self, n: int, replace: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample n items from the buffer uniformly at random.

        Args:
            n: Number of samples to draw.
            replace: Whether to sample with replacement.

        Returns:
            features (n, D), labels (n,)
        """
        if self.current_size == 0:
            return (
                np.zeros((0, self.feature_dim), dtype=np.float32),
                np.zeros(0, dtype=np.int64),
            )

        n = min(n, self.current_size) if not replace else n
        indices = np.random.choice(self.current_size, size=n, replace=replace)
        return self.features[indices].copy(), self.labels[indices].copy()

    def get_all(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return all stored samples."""
        return (
            self.features[: self.current_size].copy(),
            self.labels[: self.current_size].copy(),
        )

    def get_class_distribution(self) -> dict:
        """Return the class distribution in the buffer."""
        if self.current_size == 0:
            return {}
        unique, counts = np.unique(
            self.labels[: self.current_size], return_counts=True
        )
        return dict(zip(unique.tolist(), counts.tolist()))

    def __len__(self) -> int:
        return self.current_size

    def __repr__(self) -> str:
        return (
            f"ReservoirReplayBuffer(capacity={self.capacity}, "
            f"size={self.current_size}, total_seen={self.count})"
        )
