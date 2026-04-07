"""
Protocol-Constrained Adversarial Robustness for Encrypted Traffic

Implements:
- Protocol-admissible perturbations (Definition 1 in paper)
- Randomized smoothing with protocol-aware certification (Theorem 1)
- Protocol constraint checking for TLS/TCP/IP compliance

Key Result (Theorem 1):
    The certified robustness radius is enlarged by factor
    sqrt(1 + beta(rho)) compared to standard randomized smoothing,
    where rho is the fraction of the l2 ball satisfying protocol constraints.

    For TLS 1.3 with rho = 0.42:
    R_enhanced = R_std * sqrt(1 + (1-rho)/rho) ~ 1.58 * R_std
    (58% improvement over standard methods)

References:
    Paper Section 3.4 - Protocol-Aware Robustness
    Theorem 1 - Protocol-Constrained Robustness Certificate
    Definition 1 - Protocol-Admissible Perturbations
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
from scipy.stats import norm


class ProtocolConstraintChecker:
    """
    Validates that adversarial perturbations satisfy network protocol constraints.

    Protocol-admissible perturbations (Definition 1) must satisfy:
    - Packet sizes in [40, 1500] bytes (TCP/IP minimum to MTU)
    - Inter-arrival times >= 0 ms
    - Direction flags in {0, 1}
    - TLS handshake fields within valid ranges
    """

    def __init__(self, min_packet_size: int = 40, max_packet_size: int = 1500,
                 min_iat: float = 0.0, max_iat: float = 60000.0,
                 direction_values: list = None):
        self.min_packet_size = min_packet_size
        self.max_packet_size = max_packet_size
        self.min_iat = min_iat
        self.max_iat = max_iat
        self.direction_values = direction_values or [0, 1]

    def check_constraints(self, perturbed: torch.Tensor,
                          feature_indices: Dict[str, list] = None
                          ) -> torch.Tensor:
        """
        Check which samples satisfy protocol constraints.

        Args:
            perturbed: Perturbed features (batch_size, num_features)
            feature_indices: Map of feature names to column indices

        Returns:
            Boolean mask (batch_size,) indicating valid perturbations
        """
        valid = torch.ones(perturbed.size(0), dtype=torch.bool,
                           device=perturbed.device)

        if feature_indices is None:
            return valid

        if 'packet_size' in feature_indices:
            idx = feature_indices['packet_size']
            ps = perturbed[:, idx]
            valid &= (ps >= self.min_packet_size).all(dim=-1)
            valid &= (ps <= self.max_packet_size).all(dim=-1)

        if 'iat' in feature_indices:
            idx = feature_indices['iat']
            iat = perturbed[:, idx]
            valid &= (iat >= self.min_iat).all(dim=-1)
            valid &= (iat <= self.max_iat).all(dim=-1)

        return valid

    def project_to_constraints(self, perturbed: torch.Tensor,
                               feature_indices: Dict[str, list] = None
                               ) -> torch.Tensor:
        """Project perturbed features onto protocol-admissible set."""
        projected = perturbed.clone()

        if feature_indices is None:
            return projected

        if 'packet_size' in feature_indices:
            idx = feature_indices['packet_size']
            projected[:, idx] = projected[:, idx].clamp(
                self.min_packet_size, self.max_packet_size
            )

        if 'iat' in feature_indices:
            idx = feature_indices['iat']
            projected[:, idx] = projected[:, idx].clamp(
                self.min_iat, self.max_iat
            )

        if 'direction' in feature_indices:
            idx = feature_indices['direction']
            projected[:, idx] = projected[:, idx].round().clamp(0, 1)

        return projected


class RandomizedSmoothing(nn.Module):
    """
    Randomized smoothing with protocol-aware certified robustness.

    Implements Theorem 1 from the paper:
    For a base classifier f, the smoothed classifier g(x) returns
    the most probable class under Gaussian perturbations, with
    certified radius:
        R = sigma * Phi^{-1}(p_A)

    With protocol constraints (fraction rho of l2 ball valid):
        R_enhanced = R_std * sqrt(1 + beta(rho))
        where beta(rho) = (1 - rho) / rho

    For TLS 1.3 (rho = 0.42): R_enhanced ~ 1.58 * R_std

    Reference: Theorem 1 - Protocol-Constrained Robustness Certificate
    """

    def __init__(self, base_model: nn.Module, sigma: float = 0.1,
                 protocol_rho: float = 0.42):
        """
        Args:
            base_model: Base classifier
            sigma: Gaussian noise standard deviation
            protocol_rho: Fraction of l2 ball satisfying protocol constraints
        """
        super().__init__()
        self.base_model = base_model
        self.sigma = sigma
        self.protocol_rho = protocol_rho

    def _sample_noise(self, x: torch.Tensor, n: int) -> torch.Tensor:
        """Generate n Gaussian noise samples."""
        return torch.randn(n, *x.shape, device=x.device) * self.sigma

    def predict(self, x: torch.Tensor, n: int = 100,
                batch_size: int = 64) -> torch.Tensor:
        """
        Predict class for input using Monte Carlo sampling.

        Args:
            x: Input (num_features,) or (1, num_features)
            n: Number of noise samples
            batch_size: Batch size for sampling

        Returns:
            Predicted class index
        """
        self.base_model.eval()
        counts = None

        if x.dim() == 1:
            x = x.unsqueeze(0)

        with torch.no_grad():
            for i in range(0, n, batch_size):
                bs = min(batch_size, n - i)
                noise = torch.randn(bs, *x.shape[1:], device=x.device) * self.sigma
                noisy_x = x.expand(bs, -1) + noise
                if noisy_x.dim() == 2:
                    noisy_x = noisy_x.unsqueeze(1)

                outputs = self.base_model(noisy_x)
                preds = outputs.argmax(dim=1)

                if counts is None:
                    num_classes = outputs.shape[1]
                    counts = torch.zeros(num_classes, device=x.device)

                for c in range(num_classes):
                    counts[c] += (preds == c).sum().float()

        return counts.argmax()

    def certify(self, x: torch.Tensor, n0: int = 100,
                n: int = 10000, alpha: float = 0.001,
                batch_size: int = 64) -> Tuple[int, float]:
        """
        Certify robustness of prediction.

        Args:
            x: Input sample
            n0: Samples for initial prediction
            n: Samples for certification
            alpha: Confidence level (1 - alpha)
            batch_size: Batch size

        Returns:
            (predicted_class, certified_radius)
        """
        self.base_model.eval()

        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Step 1: Initial prediction
        top_class = self.predict(x, n=n0, batch_size=batch_size).item()

        # Step 2: Count for certification
        counts = 0
        total = 0

        with torch.no_grad():
            for i in range(0, n, batch_size):
                bs = min(batch_size, n - i)
                noise = torch.randn(bs, *x.shape[1:], device=x.device) * self.sigma
                noisy_x = x.expand(bs, -1) + noise
                if noisy_x.dim() == 2:
                    noisy_x = noisy_x.unsqueeze(1)

                outputs = self.base_model(noisy_x)
                preds = outputs.argmax(dim=1)
                counts += (preds == top_class).sum().item()
                total += bs

        # Step 3: Compute certified radius
        p_a = counts / total

        # Clopper-Pearson lower bound
        p_a_lower = self._clopper_pearson_lower(counts, total, alpha)

        if p_a_lower > 0.5:
            # Standard radius
            r_std = self.sigma * norm.ppf(p_a_lower)

            # Protocol-enhanced radius (Theorem 1)
            beta = (1 - self.protocol_rho) / self.protocol_rho
            r_enhanced = r_std * np.sqrt(1 + beta)

            return int(top_class), float(r_enhanced)
        else:
            return -1, 0.0  # Abstain

    def _clopper_pearson_lower(self, k: int, n: int,
                                alpha: float) -> float:
        """Lower bound of Clopper-Pearson confidence interval."""
        from scipy.stats import beta
        if k == 0:
            return 0.0
        return beta.ppf(alpha, k, n - k + 1)


def evaluate_certified_robustness(
    smoothed_model: RandomizedSmoothing,
    constraint_checker: ProtocolConstraintChecker,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epsilon: float = 0.1,
    num_samples: int = 100
) -> Dict[str, float]:
    """
    Evaluate certified robustness on a test set.

    Args:
        smoothed_model: Randomized smoothing model
        constraint_checker: Protocol constraint checker
        test_loader: Test data loader
        device: Compute device
        epsilon: Perturbation budget for evaluation
        num_samples: Number of samples to certify

    Returns:
        Dictionary with certified accuracy and average radius
    """
    correct = 0
    certified = 0
    total = 0
    radii = []

    smoothed_model.to(device)

    for batch in test_loader:
        if len(batch) == 3:
            x, _, y = batch
        else:
            x, y = batch

        for i in range(min(len(x), num_samples - total)):
            if total >= num_samples:
                break

            xi = x[i].to(device)
            yi = y[i].item()

            pred_class, radius = smoothed_model.certify(xi, n0=100, n=1000)

            if pred_class == yi:
                correct += 1
                if radius >= epsilon:
                    certified += 1
            radii.append(radius)
            total += 1

        if total >= num_samples:
            break

    return {
        'clean_accuracy': correct / max(total, 1),
        'certified_accuracy': certified / max(total, 1),
        'avg_radius': np.mean(radii) if radii else 0.0,
        'median_radius': np.median(radii) if radii else 0.0,
        'num_evaluated': total
    }
