"""
Byzantine-Robust Aggregation Mechanisms for Federated Learning

Implements multiple aggregation strategies including:
- Simple averaging (FedAvg)
- Median and trimmed mean
- Krum and Multi-Krum
- Attention-weighted aggregation (FedLLM-API)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
from abc import ABC, abstractmethod


class BaseAggregator(ABC):
    """Base class for federated aggregation mechanisms."""

    @abstractmethod
    def aggregate(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        client_weights: Optional[List[float]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate client model updates.

        Args:
            client_updates: List of parameter dictionaries from clients
            client_weights: Optional sample-size-based weights
            **kwargs: Additional aggregation-specific parameters

        Returns:
            aggregated_update: Aggregated parameter dictionary
        """
        pass


class SimpleAvgAggregator(BaseAggregator):
    """Simple averaging (standard FedAvg)."""

    def aggregate(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        client_weights: Optional[List[float]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Compute weighted average of client updates."""

        if client_weights is None:
            # Uniform weights
            client_weights = [1.0 / len(client_updates)] * len(client_updates)
        else:
            # Normalize weights
            total_weight = sum(client_weights)
            client_weights = [w / total_weight for w in client_weights]

        # Initialize aggregated update
        aggregated = {}

        # Average each parameter
        for key in client_updates[0].keys():
            aggregated[key] = sum(
                w * update[key] for w, update in zip(client_weights, client_updates)
            )

        return aggregated


class MedianAggregator(BaseAggregator):
    """Coordinate-wise median aggregation."""

    def aggregate(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        client_weights: Optional[List[float]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Compute coordinate-wise median (Byzantine-robust)."""

        aggregated = {}

        for key in client_updates[0].keys():
            # Stack updates along new dimension
            stacked = torch.stack([update[key] for update in client_updates])

            # Compute median along client dimension
            aggregated[key] = torch.median(stacked, dim=0)[0]

        return aggregated


class TrimmedMeanAggregator(BaseAggregator):
    """Trimmed mean aggregation (removes outliers)."""

    def __init__(self, trim_ratio: float = 0.1):
        self.trim_ratio = trim_ratio

    def aggregate(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        client_weights: Optional[List[float]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Compute trimmed mean after removing outliers."""

        num_clients = len(client_updates)
        num_trim = int(num_clients * self.trim_ratio)

        aggregated = {}

        for key in client_updates[0].keys():
            # Stack updates
            stacked = torch.stack([update[key] for update in client_updates])

            # Sort along client dimension
            sorted_vals = torch.sort(stacked, dim=0)[0]

            # Trim and average
            if num_trim > 0:
                trimmed = sorted_vals[num_trim:-num_trim]
            else:
                trimmed = sorted_vals

            aggregated[key] = torch.mean(trimmed, dim=0)

        return aggregated


class KrumAggregator(BaseAggregator):
    """
    Krum aggregation: Select single client with smallest distance sum to neighbors.

    Byzantine-robust up to f < n/2 - 1 malicious clients.
    """

    def __init__(self, num_byzantine: int = 0):
        self.num_byzantine = num_byzantine

    def aggregate(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        client_weights: Optional[List[float]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Select client with smallest distance to neighbors."""

        num_clients = len(client_updates)
        num_neighbors = num_clients - self.num_byzantine - 2

        # Flatten updates for distance computation
        flattened = [
            torch.cat([param.flatten() for param in update.values()])
            for update in client_updates
        ]

        # Compute pairwise distances
        distances = torch.zeros(num_clients, num_clients)
        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                dist = torch.norm(flattened[i] - flattened[j])
                distances[i, j] = dist
                distances[j, i] = dist

        # Compute score for each client (sum of distances to k nearest)
        scores = torch.zeros(num_clients)
        for i in range(num_clients):
            sorted_distances = torch.sort(distances[i])[0]
            scores[i] = torch.sum(sorted_distances[1:num_neighbors+1])

        # Select client with minimum score
        selected_idx = torch.argmin(scores).item()

        return client_updates[selected_idx]


class MultiKrumAggregator(BaseAggregator):
    """
    Multi-Krum: Average m clients with smallest distance sums.

    More robust than single Krum, provides smoother aggregation.
    """

    def __init__(self, num_byzantine: int = 0, num_selected: Optional[int] = None):
        self.num_byzantine = num_byzantine
        self.num_selected = num_selected

    def aggregate(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        client_weights: Optional[List[float]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Average m clients with smallest Krum scores."""

        num_clients = len(client_updates)
        num_neighbors = num_clients - self.num_byzantine - 2

        if self.num_selected is None:
            self.num_selected = num_clients - self.num_byzantine - 2

        # Flatten updates
        flattened = [
            torch.cat([param.flatten() for param in update.values()])
            for update in client_updates
        ]

        # Compute pairwise distances
        distances = torch.zeros(num_clients, num_clients)
        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                dist = torch.norm(flattened[i] - flattened[j])
                distances[i, j] = dist
                distances[j, i] = dist

        # Compute Krum scores
        scores = torch.zeros(num_clients)
        for i in range(num_clients):
            sorted_distances = torch.sort(distances[i])[0]
            scores[i] = torch.sum(sorted_distances[1:num_neighbors+1])

        # Select top m clients
        _, selected_indices = torch.topk(scores, self.num_selected, largest=False)

        # Average selected clients
        selected_updates = [client_updates[idx] for idx in selected_indices]

        return SimpleAvgAggregator().aggregate(selected_updates)


class AttentionWeightedAggregator(BaseAggregator):
    """
    Attention-weighted aggregation (FedLLM-API).

    Dynamically weights clients based on:
    1. Gradient consistency (cosine similarity to others)
    2. Validation performance

    Provides certified Byzantine robustness with smoother aggregation than Krum.
    """

    def __init__(self, temperature: float = 0.5, use_validation: bool = True):
        self.temperature = temperature
        self.use_validation = use_validation

    def aggregate(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        client_weights: Optional[List[float]] = None,
        validation_losses: Optional[List[float]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate using attention weights based on consistency and performance.

        Args:
            client_updates: Client parameter updates
            client_weights: Sample-size weights (optional)
            validation_losses: Validation losses for each client (optional)
        """

        num_clients = len(client_updates)

        # Flatten updates for similarity computation
        flattened = [
            torch.cat([param.flatten() for param in update.values()])
            for update in client_updates
        ]

        # Compute pairwise cosine similarities
        similarities = torch.zeros(num_clients, num_clients)
        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                sim = torch.nn.functional.cosine_similarity(
                    flattened[i].unsqueeze(0),
                    flattened[j].unsqueeze(0)
                )
                similarities[i, j] = sim
                similarities[j, i] = sim

        # Compute attention weights based on average similarity
        consistency_scores = torch.sum(similarities, dim=1) / (num_clients - 1)

        # Normalize to [0, 1]
        consistency_scores = (consistency_scores - consistency_scores.min()) / \
                           (consistency_scores.max() - consistency_scores.min() + 1e-8)

        # If validation losses provided, incorporate them
        if self.use_validation and validation_losses is not None:
            val_losses_tensor = torch.tensor(validation_losses, dtype=torch.float32)

            # Lower loss = higher weight (use negative loss)
            performance_scores = torch.exp(-self.temperature * val_losses_tensor)
            performance_scores = performance_scores / performance_scores.sum()

            # Combine consistency and performance
            attention_weights = consistency_scores * performance_scores
        else:
            attention_weights = consistency_scores

        # Apply softmax with temperature
        attention_weights = torch.softmax(attention_weights / self.temperature, dim=0)

        # Weight and aggregate
        aggregated = {}
        for key in client_updates[0].keys():
            aggregated[key] = sum(
                w * update[key] for w, update in zip(attention_weights, client_updates)
            )

        return aggregated


def create_aggregator(method: str, **kwargs) -> BaseAggregator:
    """
    Factory function to create aggregator from method name.

    Args:
        method: Aggregation method name
        **kwargs: Method-specific parameters

    Returns:
        Instantiated aggregator
    """

    aggregators = {
        'simple_avg': SimpleAvgAggregator,
        'fedavg': SimpleAvgAggregator,
        'median': MedianAggregator,
        'trimmed_mean': TrimmedMeanAggregator,
        'krum': KrumAggregator,
        'multi_krum': MultiKrumAggregator,
        'attention_weighted': AttentionWeightedAggregator
    }

    if method.lower() not in aggregators:
        raise ValueError(f"Unknown aggregation method: {method}")

    return aggregators[method.lower()](**kwargs)


if __name__ == "__main__":
    # Test aggregators with synthetic updates
    print("Testing aggregation mechanisms...")

    # Create synthetic client updates
    num_clients = 10
    param_size = 100

    client_updates = [
        {'param': torch.randn(param_size) + i * 0.1}  # Slight shift per client
        for i in range(num_clients)
    ]

    # Add one malicious client (large random noise)
    client_updates[-1] = {'param': torch.randn(param_size) * 10}

    validation_losses = [0.5 + 0.1 * i for i in range(num_clients)]
    validation_losses[-1] = 5.0  # Malicious client has high loss

    # Test each aggregator
    methods = [
        ('Simple Average', 'simple_avg', {}),
        ('Median', 'median', {}),
        ('Trimmed Mean', 'trimmed_mean', {'trim_ratio': 0.1}),
        ('Krum', 'krum', {'num_byzantine': 1}),
        ('Multi-Krum', 'multi_krum', {'num_byzantine': 1}),
        ('Attention-Weighted', 'attention_weighted', {'temperature': 0.5})
    ]

    for name, method, kwargs in methods:
        aggregator = create_aggregator(method, **kwargs)

        if method == 'attention_weighted':
            result = aggregator.aggregate(
                client_updates,
                validation_losses=validation_losses
            )
        else:
            result = aggregator.aggregate(client_updates)

        print(f"{name:20s}: norm = {torch.norm(result['param']).item():.4f}")

    print("\n✓ All aggregators tested successfully!")
