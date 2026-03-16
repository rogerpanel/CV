"""
KL-Divergence Drift Detector.

Monitors the class-marginal distribution shift between the model's
predictions on the validation set and incoming batches.

From Section IV-A (Equation 8):
  D_KL(P_val || P_new) = Σ_k P_val(k) * log(P_val(k) / P_new(k))

Three operational regimes:
  - Stable:  D_KL < τ_1 = 0.05  (continue monitoring)
  - Monitor: τ_1 ≤ D_KL < τ_2   (alert, optional update)
  - Drift:   D_KL ≥ τ_2 = 0.15  (update recommended)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DriftStatus:
    STABLE = "stable"
    MONITOR = "monitor"
    DRIFT = "drift"


class DriftDetector:
    """
    KL-divergence based drift detector for network traffic distributions.

    Compares the class-marginal distribution of the model's predictions on
    a reference validation set against incoming data batches.

    Args:
        num_classes: Number of traffic classes.
        tau_1: Lower threshold (stable/monitor boundary). Default: 0.05.
        tau_2: Upper threshold (monitor/drift boundary). Default: 0.15.
        smoothing: Laplace smoothing constant for distribution estimation.
    """

    def __init__(
        self,
        num_classes: int,
        tau_1: float = 0.05,
        tau_2: float = 0.15,
        smoothing: float = 1e-8,
    ):
        self.num_classes = num_classes
        self.tau_1 = tau_1
        self.tau_2 = tau_2
        self.smoothing = smoothing

        self.reference_distribution: Optional[np.ndarray] = None
        self.drift_history: list = []

    def set_reference(
        self,
        model: nn.Module,
        val_features: np.ndarray,
        val_labels: np.ndarray,
        device: torch.device,
    ) -> np.ndarray:
        """
        Set the reference class-marginal distribution from validation data.

        Uses the model's predictions (not ground truth) to establish the
        expected distribution under the current model.
        """
        model.eval()
        predictions = self._get_predictions(model, val_features, device)
        self.reference_distribution = self._compute_marginal(predictions)
        logger.info(
            f"Reference distribution set from {len(val_features)} samples"
        )
        return self.reference_distribution

    def check_drift(
        self,
        model: nn.Module,
        new_features: np.ndarray,
        device: torch.device,
    ) -> Tuple[str, float]:
        """
        Check for distribution drift in incoming data.

        Returns:
            (status, kl_divergence): Status string and KL value.
        """
        if self.reference_distribution is None:
            logger.warning("No reference distribution set; returning stable")
            return DriftStatus.STABLE, 0.0

        model.eval()
        predictions = self._get_predictions(model, new_features, device)
        new_distribution = self._compute_marginal(predictions)

        kl_div = self._kl_divergence(
            self.reference_distribution, new_distribution
        )

        if kl_div < self.tau_1:
            status = DriftStatus.STABLE
        elif kl_div < self.tau_2:
            status = DriftStatus.MONITOR
        else:
            status = DriftStatus.DRIFT

        self.drift_history.append({
            "kl_divergence": kl_div,
            "status": status,
            "num_samples": len(new_features),
        })

        logger.info(f"Drift check: KL={kl_div:.4f}, status={status}")
        return status, kl_div

    def _get_predictions(
        self,
        model: nn.Module,
        features: np.ndarray,
        device: torch.device,
    ) -> np.ndarray:
        """Get model predictions on features."""
        model.eval()
        tensor = torch.FloatTensor(features).to(device)
        batch_size = 512

        all_preds = []
        with torch.no_grad():
            for i in range(0, len(tensor), batch_size):
                batch = tensor[i : i + batch_size]
                output = model(batch)
                logits = output[0] if isinstance(output, tuple) else output
                preds = logits.argmax(dim=-1).cpu().numpy()
                all_preds.append(preds)

        return np.concatenate(all_preds)

    def _compute_marginal(self, predictions: np.ndarray) -> np.ndarray:
        """Compute smoothed class-marginal distribution."""
        counts = np.bincount(predictions, minlength=self.num_classes)
        distribution = (counts + self.smoothing) / (
            counts.sum() + self.smoothing * self.num_classes
        )
        return distribution

    def _kl_divergence(
        self, p: np.ndarray, q: np.ndarray
    ) -> float:
        """
        Compute KL divergence: D_KL(P || Q) = Σ P(k) * log(P(k) / Q(k))
        """
        # Ensure non-zero
        p = np.clip(p, self.smoothing, None)
        q = np.clip(q, self.smoothing, None)

        # Normalise
        p = p / p.sum()
        q = q / q.sum()

        return float(np.sum(p * np.log(p / q)))

    def get_drift_summary(self) -> Dict:
        """Get summary of drift detection history."""
        if not self.drift_history:
            return {"total_checks": 0}

        kl_values = [h["kl_divergence"] for h in self.drift_history]
        statuses = [h["status"] for h in self.drift_history]

        return {
            "total_checks": len(self.drift_history),
            "mean_kl": np.mean(kl_values),
            "max_kl": np.max(kl_values),
            "drift_count": statuses.count(DriftStatus.DRIFT),
            "monitor_count": statuses.count(DriftStatus.MONITOR),
            "stable_count": statuses.count(DriftStatus.STABLE),
        }
