"""
Differential Privacy for Federated Learning

Implements (epsilon, delta)-differential privacy through Gaussian noise
addition to model updates, providing formal privacy guarantees.

The noise multiplier sigma = sqrt(2 * ln(1.25/delta)) / epsilon
ensures that individual client contributions remain private.

Reference:
    Paper Section 3.5.2 - Differential Privacy Guarantees
    Abadi et al. (2016) - Deep Learning with Differential Privacy
"""

import torch
import numpy as np
from typing import Dict, Tuple


class DifferentialPrivacy:
    """
    Differential Privacy mechanism for federated learning.

    Adds calibrated Gaussian noise to model updates to provide
    (epsilon, delta)-differential privacy guarantees.
    """

    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5,
                 max_grad_norm: float = 1.0,
                 noise_multiplier: float = None):
        """
        Args:
            epsilon: Privacy budget (smaller = more private)
            delta: Failure probability
            max_grad_norm: Maximum gradient norm for clipping
            noise_multiplier: Noise scale (computed from eps/delta if None)
        """
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm

        if noise_multiplier is None:
            self.noise_multiplier = self._compute_noise_multiplier()
        else:
            self.noise_multiplier = noise_multiplier

    def _compute_noise_multiplier(self) -> float:
        """Compute sigma = sqrt(2 * ln(1.25/delta)) / epsilon."""
        return np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon

    def clip_gradients(self, parameters: Dict[str, torch.Tensor],
                       max_norm: float = None) -> Dict[str, torch.Tensor]:
        """Clip gradients to bound sensitivity."""
        if max_norm is None:
            max_norm = self.max_grad_norm

        total_norm = torch.sqrt(sum(
            torch.sum(param ** 2) for param in parameters.values()
        ))

        clip_coef = max_norm / (total_norm + 1e-6)

        if clip_coef < 1:
            return {name: param * clip_coef for name, param in parameters.items()}
        return parameters

    def add_noise(self, parameters: Dict[str, torch.Tensor],
                  sensitivity: float = None,
                  device: torch.device = None) -> Dict[str, torch.Tensor]:
        """Add Gaussian noise to parameters for differential privacy."""
        if sensitivity is None:
            sensitivity = self.max_grad_norm

        if device is None:
            device = next(iter(parameters.values())).device

        noise_scale = self.noise_multiplier * sensitivity

        noisy = {}
        for name, param in parameters.items():
            noise = torch.randn_like(param) * noise_scale
            noisy[name] = param + noise

        return noisy

    def privatize_aggregation(self, aggregated_parameters: Dict[str, torch.Tensor],
                              num_clients: int,
                              device: torch.device = None) -> Dict[str, torch.Tensor]:
        """Apply DP to aggregated parameters (clip + noise)."""
        clipped = self.clip_gradients(aggregated_parameters)
        sensitivity = self.max_grad_norm / num_clients
        return self.add_noise(clipped, sensitivity=sensitivity, device=device)

    def get_privacy_spent(self, steps: int, batch_size: int,
                          dataset_size: int) -> Tuple[float, float]:
        """
        Compute privacy budget spent (simplified moments accountant).

        Returns:
            (epsilon_total, delta)
        """
        q = batch_size / dataset_size
        epsilon_step = (
            np.sqrt(2 * np.log(1.25 / self.delta)) * q / self.noise_multiplier
        )
        epsilon_total = epsilon_step * np.sqrt(steps)
        return epsilon_total, self.delta
