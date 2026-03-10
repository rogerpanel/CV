"""
Online Adaptation with EWC + PSI Drift Detection
=================================================
Implements Section VIII-F and Algorithm S2 (supplementary):
  - Population Stability Index (PSI) for drift detection
  - Elastic Weight Consolidation (EWC) to prevent catastrophic forgetting
  - EMA parameter updates: θ ← ρ·θ + (1-ρ)·θ_new - η·Ω⊙(θ-θ*)
  - Optional DP-SGD noise for (ε,δ)-differential privacy
"""

import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Dict, List
from collections import deque


class OnlineAdapter:
    """Online learning with EWC for concept drift adaptation.

    From Table S1 (hyperparameters):
        EMA rate ρ = 0.98
        EWC weight η = 5×10⁻³
        PSI threshold = 0.2
        Mini-epochs R = 18
        Learning rate schedule: η_t = η_0 / (1 + ρ·t)
    """

    def __init__(self, model: nn.Module, device: torch.device,
                 buffer_size: int = 10000,
                 ema_rate: float = 0.98,
                 ewc_weight: float = 5e-3,
                 psi_threshold: float = 0.2,
                 n_psi_bins: int = 10,
                 mini_epochs: int = 18,
                 lr: float = 1e-3,
                 lr_decay: float = 0.02,
                 dp_clip: Optional[float] = 1.0,
                 dp_noise_sigma: Optional[float] = None):
        self.model = model.to(device)
        self.device = device
        self.ema_rate = ema_rate
        self.ewc_weight = ewc_weight
        self.psi_threshold = psi_threshold
        self.n_psi_bins = n_psi_bins
        self.mini_epochs = mini_epochs
        self.base_lr = lr
        self.lr_decay = lr_decay
        self.dp_clip = dp_clip
        self.dp_noise_sigma = dp_noise_sigma

        # Anchor parameters θ* and Fisher diagonal Ω
        self.anchor_params = {
            n: p.data.clone() for n, p in model.named_parameters()
        }
        self.fisher_diag = {
            n: torch.zeros_like(p) for n, p in model.named_parameters()
        }

        # Experience replay buffer
        self.buffer_x = deque(maxlen=buffer_size)
        self.buffer_y = deque(maxlen=buffer_size)

        # Baseline feature distribution for PSI
        self.baseline_hist = None
        self.n_updates = 0
        self.adaptation_log: List[Dict] = []

    def compute_psi(self, current_features: np.ndarray) -> float:
        """Population Stability Index.

        PSI = Σ_b (p_b - q_b) ln(p_b / q_b)

        where p = baseline distribution, q = current distribution.
        """
        if self.baseline_hist is None:
            # Initialise baseline from first window
            self._set_baseline(current_features)
            return 0.0

        # Compute histograms over first 5 features (representative)
        n_feat = min(current_features.shape[1], 5)
        psi = 0.0

        for f in range(n_feat):
            baseline = self.baseline_hist[f]
            current = np.histogram(
                current_features[:, f],
                bins=self.n_psi_bins, density=True
            )[0]

            # Add small epsilon to avoid log(0)
            p = baseline + 1e-8
            q = current + 1e-8

            # Normalise
            p = p / p.sum()
            q = q / q.sum()

            psi += np.sum((p - q) * np.log(p / q))

        return psi / n_feat

    def _set_baseline(self, features: np.ndarray):
        """Store baseline feature distribution."""
        n_feat = min(features.shape[1], 5)
        self.baseline_hist = {}
        for f in range(n_feat):
            self.baseline_hist[f] = np.histogram(
                features[:, f], bins=self.n_psi_bins, density=True
            )[0]

    def update_fisher(self, dataloader, n_batches: int = 10):
        """Estimate Fisher information diagonal from data."""
        self.model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}

        count = 0
        for batch in dataloader:
            if count >= n_batches:
                break
            x = batch["features"].to(self.device)
            y = batch["label"].to(self.device)

            self.model.zero_grad()
            t_span = torch.linspace(0, 1, 10).to(self.device)
            logits, _, _ = self.model(x, t_span)
            loss = nn.functional.cross_entropy(logits, y)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.data.pow(2)
            count += 1

        # Average
        for n in fisher:
            fisher[n] /= max(count, 1)
        self.fisher_diag = fisher

    def should_adapt(self, current_features: np.ndarray) -> bool:
        """Check if PSI exceeds threshold, triggering adaptation."""
        psi = self.compute_psi(current_features)
        return psi > self.psi_threshold

    def adapt(self, stream_x: torch.Tensor, stream_y: torch.Tensor):
        """Online adaptation with EWC + EMA (Algorithm S2).

        θ ← ρ·θ + (1-ρ)·θ_new - η·Ω⊙(θ-θ*)
        """
        self.n_updates += 1
        lr_t = self.base_lr / (1 + self.lr_decay * self.n_updates)

        optimizer = optim.Adam(self.model.parameters(), lr=lr_t)
        t_span = torch.linspace(0, 1, 10).to(self.device)

        # Add to buffer
        for i in range(len(stream_x)):
            self.buffer_x.append(stream_x[i].cpu())
            self.buffer_y.append(stream_y[i].cpu())

        # Sample from buffer
        buffer_size = len(self.buffer_x)
        if buffer_size < 32:
            return

        self.model.train()
        for _ in range(self.mini_epochs):
            # Sample mini-batch from buffer
            idx = np.random.choice(buffer_size, min(256, buffer_size),
                                   replace=False)
            batch_x = torch.stack([self.buffer_x[i] for i in idx]).to(self.device)
            batch_y = torch.stack([self.buffer_y[i] for i in idx]).to(self.device)

            optimizer.zero_grad()
            logits, _, _ = self.model(batch_x, t_span)
            loss = nn.functional.cross_entropy(logits, batch_y)

            # EWC penalty: λ_EWC Σ_i F_i (θ_i - θ*_i)²
            ewc_loss = torch.tensor(0.0, device=self.device)
            for n, p in self.model.named_parameters():
                if n in self.fisher_diag:
                    ewc_loss += (self.fisher_diag[n] *
                                 (p - self.anchor_params[n]).pow(2)).sum()
            loss = loss + self.ewc_weight * ewc_loss

            loss.backward()

            # DP-SGD: clip gradients and optionally add noise
            if self.dp_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.dp_clip
                )
            if self.dp_noise_sigma:
                for p in self.model.parameters():
                    if p.grad is not None:
                        noise = torch.randn_like(p.grad) * self.dp_noise_sigma
                        p.grad.add_(noise)

            optimizer.step()

        # EMA update
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if n in self.anchor_params:
                    p.data.mul_(self.ema_rate).add_(
                        (1 - self.ema_rate) * self.anchor_params[n]
                    )

        self.adaptation_log.append({
            "step": self.n_updates,
            "buffer_size": buffer_size,
            "lr": lr_t,
        })

    def update_anchor(self):
        """Update anchor parameters after successful adaptation."""
        self.anchor_params = {
            n: p.data.clone() for n, p in self.model.named_parameters()
        }
