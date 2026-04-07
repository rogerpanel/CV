"""
Advanced aggregation strategies for federated learning

Implements:
- Standard FedAvg aggregation
- Coordinate-wise median (robust to outliers)
- Trimmed mean aggregation
- Gradient Similarity Aggregation (reduces communication by 35%)

Reference:
    Paper Section 3.5.1 - Aggregation Strategies
    Wang et al. (2024) - NIDS-FGPA with Gradient Similarity Aggregation
"""

import torch
import numpy as np
from typing import List, Dict


def aggregate_models(
    client_parameters: List[Dict[str, torch.Tensor]],
    client_weights: List[float],
    strategy: str = 'fedavg'
) -> Dict[str, torch.Tensor]:
    """
    Aggregate client model parameters.

    Args:
        client_parameters: List of client model parameters
        client_weights: Weights (e.g., dataset sizes)
        strategy: 'fedavg', 'median', or 'trimmed_mean'

    Returns:
        Aggregated model parameters
    """
    if strategy == 'fedavg':
        return fedavg_aggregate(client_parameters, client_weights)
    elif strategy == 'median':
        return median_aggregate(client_parameters)
    elif strategy == 'trimmed_mean':
        return trimmed_mean_aggregate(client_parameters, trim_ratio=0.1)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def fedavg_aggregate(
    client_parameters: List[Dict[str, torch.Tensor]],
    client_weights: List[float]
) -> Dict[str, torch.Tensor]:
    """FedAvg: weighted average of client parameters."""
    total_weight = sum(client_weights)
    normalized = [w / total_weight for w in client_weights]

    aggregated = {}
    for name in client_parameters[0]:
        aggregated[name] = sum(
            w * params[name] for w, params in zip(normalized, client_parameters)
        )
    return aggregated


def median_aggregate(
    client_parameters: List[Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """Coordinate-wise median aggregation (robust to outliers)."""
    aggregated = {}
    for name in client_parameters[0]:
        stacked = torch.stack([p[name] for p in client_parameters])
        aggregated[name] = torch.median(stacked, dim=0)[0]
    return aggregated


def trimmed_mean_aggregate(
    client_parameters: List[Dict[str, torch.Tensor]],
    trim_ratio: float = 0.1
) -> Dict[str, torch.Tensor]:
    """Trimmed mean: removes top/bottom fraction before averaging."""
    num_clients = len(client_parameters)
    num_trim = int(num_clients * trim_ratio)

    aggregated = {}
    for name in client_parameters[0]:
        stacked = torch.stack([p[name] for p in client_parameters])
        sorted_params, _ = torch.sort(stacked, dim=0)

        if num_trim > 0:
            trimmed = sorted_params[num_trim:-num_trim]
        else:
            trimmed = sorted_params

        aggregated[name] = torch.mean(trimmed, dim=0)

    return aggregated


def gradient_similarity_aggregation(
    client_parameters: List[Dict[str, torch.Tensor]],
    global_parameters: Dict[str, torch.Tensor],
    client_weights: List[float],
    similarity_threshold: float = 0.5
) -> Dict[str, torch.Tensor]:
    """
    Gradient Similarity Aggregation.

    Filters clients based on cosine similarity of their gradient
    updates to the mean gradient. Reduces communication rounds by 35%.

    Reference: Wang et al. (2024) - NIDS-FGPA
    """
    # Compute gradients (difference from global)
    client_gradients = []
    for params in client_parameters:
        gradient = {
            name: params[name] - global_parameters[name]
            for name in global_parameters
        }
        client_gradients.append(gradient)

    # Mean gradient
    mean_gradient = {}
    for name in global_parameters:
        mean_gradient[name] = sum(
            g[name] for g in client_gradients
        ) / len(client_gradients)

    # Cosine similarity per client
    similarities = []
    for grad in client_gradients:
        sim = _cosine_similarity(grad, mean_gradient)
        similarities.append(sim)

    # Filter
    selected = [
        i for i, sim in enumerate(similarities)
        if sim >= similarity_threshold
    ]
    if len(selected) == 0:
        selected = list(range(len(client_parameters)))

    selected_params = [client_parameters[i] for i in selected]
    selected_weights = [client_weights[i] for i in selected]

    print(f"Gradient Similarity: Selected "
          f"{len(selected)}/{len(client_parameters)} clients")

    return fedavg_aggregate(selected_params, selected_weights)


def _cosine_similarity(
    params1: Dict[str, torch.Tensor],
    params2: Dict[str, torch.Tensor]
) -> float:
    """Cosine similarity between two parameter dictionaries."""
    flat1 = torch.cat([p.flatten() for p in params1.values()])
    flat2 = torch.cat([p.flatten() for p in params2.values()])

    return torch.nn.functional.cosine_similarity(
        flat1.unsqueeze(0), flat2.unsqueeze(0), dim=1
    ).item()
