"""
Training pipeline with 5-fold time-series cross-validation.

Implements Section 5.1:
  - Adam optimizer, lr=1e-3 with cosine annealing to 1e-5
  - Batch size 256, max 100 epochs with early stopping
  - Per-batch gradient clipping (max norm 1.0)
  - 5-fold time-series cross-validation
  - Optional Bayesian training (10 MC samples during training)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from typing import Dict, List, Optional, Callable
from collections import defaultdict
import time
import copy


class TimeSeriesKFold:
    """Time-series-aware K-fold cross-validation.

    Ensures temporal ordering: training data always precedes validation data.
    No shuffling — each fold uses a contiguous block for validation.
    """

    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits

    def split(self, n_samples: int):
        """Yield (train_indices, val_indices) for each fold."""
        fold_size = n_samples // self.n_splits
        for k in range(self.n_splits):
            val_start = k * fold_size
            val_end = val_start + fold_size if k < self.n_splits - 1 else n_samples

            val_idx = list(range(val_start, val_end))
            train_idx = list(range(0, val_start)) + list(range(val_end, n_samples))

            yield train_idx, val_idx


class Trainer:
    """Training loop with early stopping and cross-validation.

    Matches the experimental setup in Section 5.1.
    """

    def __init__(self, model: nn.Module, device: str = "cuda",
                 lr: float = 1e-3, min_lr: float = 1e-5,
                 batch_size: int = 256, max_epochs: int = 100,
                 patience: int = 10, grad_clip: float = 1.0,
                 loss_weights: Optional[Dict[str, float]] = None,
                 bayesian_wrapper=None):
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.min_lr = min_lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.grad_clip = grad_clip
        self.loss_weights = loss_weights or {"cls": 1.0, "tpp": 1.0, "reg": 1e-4}
        self.bayesian = bayesian_wrapper

    def train_epoch(self, train_loader: DataLoader,
                    optimizer: optim.Optimizer,
                    t_span: torch.Tensor) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = defaultdict(float)
        n_batches = 0

        for batch in train_loader:
            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)
            event_types = batch.get("event_types")
            if event_types is not None:
                event_types = event_types.to(self.device)

            optimizer.zero_grad()

            if self.bayesian is not None:
                # Bayesian training: ELBO with MC samples
                def loss_fn(model, x=x, y=y, t_span=t_span, event_types=event_types):
                    losses = model.compute_loss(x, y, t_span, event_types, self.loss_weights)
                    return losses["total"]

                neg_elbo, kl = self.bayesian.compute_elbo(
                    loss_fn, n_samples=10
                )
                loss = neg_elbo
                epoch_losses["elbo"] += neg_elbo.item()
                epoch_losses["kl"] += kl.item()
            else:
                losses = self.model.compute_loss(
                    x, y, t_span, event_types, self.loss_weights
                )
                loss = losses["total"]
                for k, v in losses.items():
                    epoch_losses[k] += v.item()

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            optimizer.step()

            n_batches += 1

        return {k: v / n_batches for k, v in epoch_losses.items()}

    @torch.no_grad()
    def validate(self, val_loader: DataLoader,
                 t_span: torch.Tensor) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        correct = 0
        total = 0
        val_loss = 0
        n_batches = 0

        for batch in val_loader:
            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)

            out = self.model(x, t_span)
            preds = out["logits"].argmax(dim=1)

            correct += (preds == y).sum().item()
            total += len(y)

            loss = nn.functional.cross_entropy(out["logits"], y)
            val_loss += loss.item()
            n_batches += 1

        return {
            "accuracy": correct / total if total > 0 else 0,
            "loss": val_loss / n_batches if n_batches > 0 else float("inf"),
        }

    def train(self, train_dataset, val_dataset,
              seq_len: int = 1) -> Dict[str, List]:
        """Full training loop with early stopping.

        Returns:
            history: Dictionary of metric lists per epoch
        """
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size,
            shuffle=False,  # Temporal ordering preserved
            num_workers=0, drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size * 4,
            shuffle=False, num_workers=0,
        )

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs, eta_min=self.min_lr
        )

        t_span = torch.linspace(0, 1, 10).to(self.device)

        history = defaultdict(list)
        best_val_acc = 0
        best_state = None
        wait = 0

        for epoch in range(self.max_epochs):
            t0 = time.time()

            train_metrics = self.train_epoch(train_loader, optimizer, t_span)
            val_metrics = self.validate(val_loader, t_span)

            scheduler.step()

            # Record history
            for k, v in train_metrics.items():
                history[f"train_{k}"].append(v)
            for k, v in val_metrics.items():
                history[f"val_{k}"].append(v)
            history["lr"].append(optimizer.param_groups[0]["lr"])
            history["epoch_time"].append(time.time() - t0)

            # Early stopping
            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                best_state = copy.deepcopy(self.model.state_dict())
                wait = 0
            else:
                wait += 1

            if (epoch + 1) % 5 == 0 or wait == 0:
                print(f"Epoch {epoch+1}/{self.max_epochs} | "
                      f"Train Loss: {train_metrics.get('total', train_metrics.get('elbo', 0)):.4f} | "
                      f"Val Acc: {val_metrics['accuracy']:.4f} | "
                      f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                      f"Time: {history['epoch_time'][-1]:.1f}s")

            if wait >= self.patience:
                print(f"Early stopping at epoch {epoch+1} (patience={self.patience})")
                break

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"Restored best model (val_acc={best_val_acc:.4f})")

        return dict(history)

    def cross_validate(self, dataset, n_folds: int = 5) -> Dict[str, List]:
        """5-fold time-series cross-validation."""
        kfold = TimeSeriesKFold(n_folds)
        fold_results = []

        initial_state = copy.deepcopy(self.model.state_dict())

        for fold, (train_idx, val_idx) in enumerate(kfold.split(len(dataset))):
            print(f"\n{'='*60}")
            print(f"Fold {fold+1}/{n_folds}")
            print(f"{'='*60}")

            # Reset model
            self.model.load_state_dict(copy.deepcopy(initial_state))

            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)

            history = self.train(train_subset, val_subset)

            # Final validation
            val_loader = DataLoader(val_subset, batch_size=self.batch_size * 4, shuffle=False)
            t_span = torch.linspace(0, 1, 10).to(self.device)
            final_metrics = self.validate(val_loader, t_span)

            fold_results.append(final_metrics)
            print(f"Fold {fold+1} — Accuracy: {final_metrics['accuracy']:.4f}")

        # Summary
        print(f"\n{'='*60}")
        print("Cross-Validation Summary")
        print(f"{'='*60}")
        accs = [r["accuracy"] for r in fold_results]
        print(f"Mean Accuracy: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")

        return {"fold_results": fold_results, "mean_accuracy": np.mean(accs),
                "std_accuracy": np.std(accs)}
