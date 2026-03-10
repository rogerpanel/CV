"""
Training Pipeline
=================
Full training with:
  - Adam optimiser, cosine annealing LR (10⁻³ → 10⁻⁵)
  - Combined loss: L_cls + L_TPP + L_ELBO + L_reg  (Eq. 3)
  - Gradient clipping (max_norm=1.0)
  - Early stopping on validation loss
  - Checkpoint saving
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict
from typing import Optional, Dict

from ..models.framework import TABNODEPointProcessFramework


class Trainer:
    """Training loop for TA-BN-ODE + DSTPP framework."""

    def __init__(self, model: TABNODEPointProcessFramework,
                 device: torch.device,
                 lr: float = 1e-3, lr_min: float = 1e-5,
                 weight_decay: float = 1e-4,
                 max_grad_norm: float = 1.0,
                 patience: int = 10,
                 checkpoint_dir: str = "checkpoints"):
        self.model = model.to(device)
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.optimizer = optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.lr = lr
        self.lr_min = lr_min
        self.scheduler = None  # Set in train()

        self.history = defaultdict(list)
        self.best_val_loss = float("inf")
        self.epochs_no_improve = 0

    def train(self, train_loader, val_loader, epochs: int = 100,
              tpp_weight: float = 0.1, kl_weight: float = 1e-4,
              reg_weight: float = 1e-3,
              n_ode_steps: int = 10) -> Dict:
        """Full training loop.

        Args:
            train_loader: training DataLoader
            val_loader: validation DataLoader
            epochs: maximum epochs
            tpp_weight: weight for point process loss
            kl_weight: weight for KL divergence
            reg_weight: weight for TA-BN regularisation
            n_ode_steps: number of ODE integration steps
        Returns:
            history dict
        """
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=self.lr_min
        )

        t_span = torch.linspace(0, 1, n_ode_steps).to(self.device)

        for epoch in range(1, epochs + 1):
            # --- Training ---
            self.model.train()
            epoch_losses = defaultdict(float)
            n_batches = 0

            for batch in train_loader:
                x = batch["features"].to(self.device)
                y = batch["label"].to(self.device)

                self.optimizer.zero_grad()

                losses = self.model.compute_loss(
                    x, y, t_span,
                    tpp_weight=tpp_weight,
                    kl_weight=kl_weight,
                    reg_weight=reg_weight,
                )

                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                for k, v in losses.items():
                    epoch_losses[k] += v.item()
                n_batches += 1

            self.scheduler.step()

            # Average training losses
            for k in epoch_losses:
                epoch_losses[k] /= max(n_batches, 1)
                self.history[f"train_{k}"].append(epoch_losses[k])

            # --- Validation ---
            val_metrics = self._validate(val_loader, t_span)
            for k, v in val_metrics.items():
                self.history[f"val_{k}"].append(v)

            # --- Early stopping ---
            val_loss = val_metrics["loss"]
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                self._save_checkpoint("best_model.pt")
            else:
                self.epochs_no_improve += 1

            # --- Logging ---
            if epoch % 5 == 0 or epoch == 1:
                lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch:3d}/{epochs} | "
                    f"Train Loss: {epoch_losses['total']:.4f} "
                    f"(cls={epoch_losses['cls']:.4f}) | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {val_metrics['accuracy']:.4f} | "
                    f"LR: {lr:.2e}"
                )

            if self.epochs_no_improve >= self.patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # Load best model
        self._load_checkpoint("best_model.pt")
        return dict(self.history)

    @torch.no_grad()
    def _validate(self, val_loader, t_span) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in val_loader:
            x = batch["features"].to(self.device)
            y = batch["label"].to(self.device)

            logits, _, _ = self.model(x, t_span)
            loss = nn.functional.cross_entropy(logits, y)

            total_loss += loss.item() * len(y)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += len(y)

        return {
            "loss": total_loss / max(total, 1),
            "accuracy": correct / max(total, 1),
        }

    def _save_checkpoint(self, filename: str):
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "history": dict(self.history),
        }, path)

    def _load_checkpoint(self, filename: str):
        path = os.path.join(self.checkpoint_dir, filename)
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device,
                              weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
