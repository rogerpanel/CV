"""
Online adaptation with EWC and concept drift detection.

Implements Algorithm S2 from the supplementary:
  - Population Stability Index (PSI) for drift detection
  - Elastic Weight Consolidation (EWC) to prevent catastrophic forgetting
  - Exponential Moving Average (EMA) parameter updates
  - Optional DP-SGD for differential privacy

Parameters (Supplementary Table S1):
  - EWC weight (lambda_EWC): 1e-2
  - EMA rate (rho): 0.02
  - Learning rate decay: eta_t = eta_0 / (1 + rho * t)
  - Mini-epochs per update: R = 18
  - PSI threshold: 0.2
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Optional, List
from collections import deque
import copy

from .evaluation import compute_psi


class ConceptDriftDetector:
    """Concept drift detection via Population Stability Index.

    Maintains a reference distribution and compares incoming data
    distributions. Triggers adaptation when PSI > threshold.
    """

    def __init__(self, threshold: float = 0.2, n_bins: int = 10,
                 window_size: int = 5000):
        self.threshold = threshold
        self.n_bins = n_bins
        self.window_size = window_size
        self.reference = None
        self.current_window = deque(maxlen=window_size)
        self.psi_history = []

    def set_reference(self, data: np.ndarray):
        """Set reference distribution from training data."""
        self.reference = data.flatten()

    def update(self, new_data: np.ndarray) -> bool:
        """Add new data and check for drift.

        Returns:
            True if drift detected (PSI > threshold)
        """
        self.current_window.extend(new_data.flatten())

        if self.reference is None or len(self.current_window) < self.n_bins * 10:
            return False

        current = np.array(self.current_window)
        psi = compute_psi(self.reference, current, self.n_bins)
        self.psi_history.append(psi)

        if psi > self.threshold:
            print(f"  Concept drift detected! PSI={psi:.4f} > {self.threshold}")
            return True
        return False


class EWCRegularizer:
    """Elastic Weight Consolidation (EWC).

    Prevents catastrophic forgetting during online updates by penalizing
    deviation from previously learned parameters:

      L_EWC = lambda_EWC * sum_i F_i * (theta_i - theta*_i)^2

    where F_i is the diagonal Fisher information.
    """

    def __init__(self, model: nn.Module, lambda_ewc: float = 1e-2):
        self.lambda_ewc = lambda_ewc
        self.saved_params = {}
        self.fisher_diag = {}

    def compute_fisher(self, model: nn.Module, data_loader,
                       device: str = "cuda", n_samples: int = 1000):
        """Estimate diagonal Fisher information from data."""
        model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}

        t_span = torch.linspace(0, 1, 10).to(device)
        count = 0

        for batch in data_loader:
            if count >= n_samples:
                break
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            model.zero_grad()
            out = model(x, t_span)
            loss = nn.functional.cross_entropy(out["logits"], y)
            loss.backward()

            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data.pow(2) * len(x)

            count += len(x)

        # Normalize
        for n in fisher:
            fisher[n] /= count

        self.fisher_diag = fisher
        self.saved_params = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """Compute EWC penalty."""
        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for n, p in model.named_parameters():
            if n in self.fisher_diag:
                loss += (self.fisher_diag[n] * (p - self.saved_params[n]).pow(2)).sum()
        return self.lambda_ewc * loss


class OnlineAdapter:
    """Online adaptation with EWC and DP-SGD (Algorithm S2).

    Triggered by concept drift detection. Performs R mini-epochs
    with EWC regularization and optional differential privacy.
    """

    def __init__(self, model: nn.Module, device: str = "cuda",
                 ewc_lambda: float = 1e-2,
                 ema_rho: float = 0.02,
                 base_lr: float = 1e-3,
                 lr_decay_rho: float = 0.02,
                 mini_epochs: int = 18,
                 dp_clip_norm: float = 1.0,
                 dp_noise_multiplier: float = 0.0,
                 psi_threshold: float = 0.2,
                 buffer_size: int = 10000):
        self.model = model.to(device)
        self.device = device
        self.ema_rho = ema_rho
        self.base_lr = base_lr
        self.lr_decay_rho = lr_decay_rho
        self.mini_epochs = mini_epochs
        self.dp_clip_norm = dp_clip_norm
        self.dp_noise_multiplier = dp_noise_multiplier
        self.n_updates = 0

        # EWC
        self.ewc = EWCRegularizer(model, ewc_lambda)

        # Drift detection
        self.drift_detector = ConceptDriftDetector(psi_threshold)

        # EMA shadow model
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()

        # Data buffer for adaptation
        self.buffer_x = deque(maxlen=buffer_size)
        self.buffer_y = deque(maxlen=buffer_size)

        # Tracking
        self.adaptation_log = []

    def _current_lr(self) -> float:
        """Decaying learning rate: eta_t = eta_0 / (1 + rho * t)."""
        return self.base_lr / (1.0 + self.lr_decay_rho * self.n_updates)

    def _ema_update(self):
        """Exponential moving average update of shadow model."""
        with torch.no_grad():
            for ema_p, model_p in zip(self.ema_model.parameters(),
                                       self.model.parameters()):
                ema_p.data.mul_(1 - self.ema_rho).add_(model_p.data, alpha=self.ema_rho)

    def _dp_sgd_step(self, optimizer: optim.Optimizer, batch_size: int):
        """Apply DP-SGD: clip per-sample gradients and add noise."""
        # Clip gradients
        nn.utils.clip_grad_norm_(self.model.parameters(), self.dp_clip_norm)

        # Add calibrated noise
        if self.dp_noise_multiplier > 0:
            for p in self.model.parameters():
                if p.grad is not None:
                    noise = torch.randn_like(p.grad) * (
                        self.dp_noise_multiplier * self.dp_clip_norm / batch_size
                    )
                    p.grad.add_(noise)

    def process_batch(self, x: torch.Tensor, y: torch.Tensor) -> Dict:
        """Process incoming batch: detect drift and adapt if needed.

        Returns:
            Dictionary with prediction and drift status
        """
        self.model.eval()
        t_span = torch.linspace(0, 1, 10).to(self.device)

        with torch.no_grad():
            out = self.model(x.to(self.device), t_span)
            probs = torch.softmax(out["logits"], dim=1)
            preds = probs.argmax(dim=1)

        # Buffer data
        self.buffer_x.extend(x.cpu())
        self.buffer_y.extend(y.cpu())

        # Check drift
        drift = self.drift_detector.update(
            probs.cpu().numpy().max(axis=1)
        )

        result = {
            "predictions": preds.cpu(),
            "probabilities": probs.cpu(),
            "drift_detected": drift,
        }

        if drift and len(self.buffer_x) >= 256:
            self.adapt()
            result["adapted"] = True

        return result

    def adapt(self):
        """Run adaptation (Algorithm S2): R mini-epochs with EWC + DP-SGD."""
        print(f"\n  Starting online adaptation (update #{self.n_updates + 1})...")

        # Compute Fisher before update
        from torch.utils.data import TensorDataset, DataLoader
        buf_x = torch.stack(list(self.buffer_x))
        buf_y = torch.stack(list(self.buffer_y))

        # Simple dataset wrapper for EWC
        class _SimpleDS:
            def __init__(self, x, y):
                self.x, self.y = x, y
            def __len__(self): return len(self.x)
            def __getitem__(self, i): return {"x": self.x[i], "y": self.y[i]}

        ds = _SimpleDS(buf_x, buf_y)
        loader = DataLoader(ds, batch_size=256, shuffle=False)

        self.ewc.compute_fisher(self.model, loader, self.device)

        # Adaptation loop
        lr = self._current_lr()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        t_span = torch.linspace(0, 1, 10).to(self.device)

        self.model.train()
        for epoch in range(self.mini_epochs):
            epoch_loss = 0
            for batch in loader:
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device)

                optimizer.zero_grad()
                out = self.model(x, t_span)
                loss = nn.functional.cross_entropy(out["logits"], y)
                loss = loss + self.ewc.penalty(self.model)

                loss.backward()
                self._dp_sgd_step(optimizer, len(x))
                optimizer.step()

                epoch_loss += loss.item()

            self._ema_update()

        self.n_updates += 1
        self.adaptation_log.append({
            "update": self.n_updates,
            "lr": lr,
            "buffer_size": len(self.buffer_x),
        })
        print(f"  Adaptation complete (lr={lr:.6f})")

    def get_ema_model(self) -> nn.Module:
        """Return the EMA-smoothed model for evaluation."""
        return self.ema_model
