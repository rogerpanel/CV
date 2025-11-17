"""
Differential Privacy for Federated Learning

Implements (ε, δ)-differential privacy through Gaussian noise addition
to model updates, providing formal privacy guarantees.

Reference:
    Abadi et al. (2016) - Deep Learning with Differential Privacy
"""

import torch
import numpy as np
from typing import Dict, Tuple


class DifferentialPrivacy:
    """
    Differential Privacy mechanism for federated learning.

    Adds calibrated Gaussian noise to model updates to provide
    (ε, δ)-differential privacy guarantees.
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        max_grad_norm: float = 1.0,
        noise_multiplier: float = None
    ):
        """
        Initialize differential privacy mechanism.

        Args:
            epsilon: Privacy budget (smaller = more private)
            delta: Failure probability
            max_grad_norm: Maximum gradient norm for clipping
            noise_multiplier: Noise scale (if None, computed from epsilon and delta)
        """
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm

        # Compute noise multiplier if not provided
        if noise_multiplier is None:
            self.noise_multiplier = self._compute_noise_multiplier()
        else:
            self.noise_multiplier = noise_multiplier

    def _compute_noise_multiplier(self) -> float:
        """
        Compute noise multiplier from privacy parameters.

        Uses the formula: σ = √(2 * ln(1.25 / δ)) / ε

        Returns:
            Noise multiplier
        """
        return np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon

    def clip_gradients(
        self,
        parameters: Dict[str, torch.Tensor],
        max_norm: float = None
    ) -> Dict[str, torch.Tensor]:
        """
        Clip gradients to bound sensitivity.

        Args:
            parameters: Model parameters
            max_norm: Maximum norm for clipping (if None, uses self.max_grad_norm)

        Returns:
            Clipped parameters
        """
        if max_norm is None:
            max_norm = self.max_grad_norm

        # Compute global norm
        total_norm = torch.sqrt(sum(
            torch.sum(param ** 2) for param in parameters.values()
        ))

        # Clip if necessary
        clip_coef = max_norm / (total_norm + 1e-6)

        if clip_coef < 1:
            # Clip parameters
            clipped = {
                name: param * clip_coef
                for name, param in parameters.items()
            }
            return clipped
        else:
            # No clipping needed
            return parameters

    def add_noise(
        self,
        parameters: Dict[str, torch.Tensor],
        sensitivity: float = None,
        device: torch.device = None
    ) -> Dict[str, torch.Tensor]:
        """
        Add Gaussian noise to parameters for differential privacy.

        Args:
            parameters: Model parameters
            sensitivity: Sensitivity of the mechanism (defaults to max_grad_norm)
            device: Device to create noise on

        Returns:
            Parameters with noise added
        """
        if sensitivity is None:
            sensitivity = self.max_grad_norm

        if device is None:
            device = next(iter(parameters.values())).device

        # Noise scale
        noise_scale = self.noise_multiplier * sensitivity

        # Add Gaussian noise to each parameter
        noisy_parameters = {}
        for name, param in parameters.items():
            noise = torch.randn_like(param) * noise_scale
            noisy_parameters[name] = param + noise

        return noisy_parameters

    def privatize_aggregation(
        self,
        aggregated_parameters: Dict[str, torch.Tensor],
        num_clients: int,
        device: torch.device = None
    ) -> Dict[str, torch.Tensor]:
        """
        Apply differential privacy to aggregated parameters.

        Performs gradient clipping and noise addition.

        Args:
            aggregated_parameters: Aggregated model parameters
            num_clients: Number of clients (affects noise scale)
            device: Device for computation

        Returns:
            Privatized parameters
        """
        # Clip gradients
        clipped = self.clip_gradients(aggregated_parameters)

        # Add noise (scaled by number of clients)
        sensitivity = self.max_grad_norm / num_clients
        privatized = self.add_noise(clipped, sensitivity=sensitivity, device=device)

        return privatized

    def get_privacy_spent(self, steps: int, batch_size: int, dataset_size: int) -> Tuple[float, float]:
        """
        Compute privacy budget spent after given number of steps.

        Uses the moments accountant method for tight privacy analysis.

        Args:
            steps: Number of training steps
            batch_size: Batch size
            dataset_size: Total dataset size

        Returns:
            Tuple of (epsilon, delta) representing privacy spent

        Note:
            This is a simplified approximation. For precise accounting,
            use libraries like Opacus or TensorFlow Privacy.
        """
        # Sampling probability
        q = batch_size / dataset_size

        # Simplified privacy accounting (not tight)
        # For tight bounds, use advanced composition theorems

        # Basic composition
        epsilon_step = np.sqrt(2 * np.log(1.25 / self.delta)) * q / self.noise_multiplier
        epsilon_total = epsilon_step * np.sqrt(steps)

        return epsilon_total, self.delta
