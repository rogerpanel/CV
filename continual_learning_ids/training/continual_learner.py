"""
Continual Learning Module: EWC + Experience Replay.

Implements the five-step training procedure from Section IV-A:
  1. Construct merged dataset: D_hat = D_new ∪ Sample(B, min(|B|, |D_new|))
  2. Optimise composite loss: L_total = L_CE(D_hat; θ) + (λ/2) Σ F_k (θ_k - θ*_k)^2
  3. Recompute Fisher diagonal and update running FIM + reference params θ*
  4. Update replay buffer via reservoir sampling
  5. Checkpoint model and EWC state

Algorithm 1 from the paper.
"""

import copy
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..utils.replay_buffer import ReservoirReplayBuffer
from ..utils.fisher import (
    compute_fisher_diagonal_efficient,
    update_running_fisher,
)
from ..data.dataset_loader import NIDSDataset, create_dataloader

logger = logging.getLogger(__name__)


class ContinualLearner:
    """
    EWC + Experience Replay continual learning pipeline.

    Maintains four persistent state elements between tasks:
      (i)   Current model parameters θ
      (ii)  Running diagonal FIM estimate F_hat^(t)
      (iii) Reference parameters θ* stored after each update
      (iv)  Experience replay buffer B of bounded size M

    Args:
        model: The SurrogateIDS detection model.
        config: Configuration dict with CL hyperparameters.
        device: Computation device.
    """

    def __init__(
        self,
        model: nn.Module,
        config: dict,
        device: torch.device,
    ):
        self.model = model
        self.device = device
        self.config = config

        # EWC hyperparameters
        cl_config = config.get("continual_learning", {})
        ewc_config = cl_config.get("ewc", {})
        replay_config = cl_config.get("replay", {})
        train_config = cl_config.get("training", {})

        self.ewc_lambda = ewc_config.get("lambda", 5000)
        self.fisher_decay = ewc_config.get("fisher_decay", 0.5)
        self.fisher_samples = ewc_config.get("fisher_samples", None)

        self.buffer_size = replay_config.get("buffer_size", 5000)
        self.replay_ratio = replay_config.get("replay_ratio", 1.0)

        self.lr = train_config.get("learning_rate", 1e-4)
        self.epochs = train_config.get("epochs_per_task", 10)
        self.batch_size = train_config.get("batch_size", 256)
        self.weight_decay = train_config.get("weight_decay", 1e-4)

        # Determine feature dim from model input
        input_dim = getattr(model, "branches", [None])[0]
        if input_dim is not None:
            feat_dim = input_dim.network[0].in_features
        else:
            feat_dim = 79

        # Persistent state
        self.replay_buffer = ReservoirReplayBuffer(
            capacity=self.buffer_size, feature_dim=feat_dim
        )
        self.running_fisher: Dict[str, torch.Tensor] = {}
        self.reference_params: Dict[str, torch.Tensor] = {}
        self.task_count = 0
        self.training_history: List[Dict] = []

    def train_on_task(
        self,
        train_X: np.ndarray,
        train_y: np.ndarray,
        val_X: Optional[np.ndarray] = None,
        val_y: Optional[np.ndarray] = None,
        task_id: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Execute the five-step EWC+Replay training procedure for one task.

        Args:
            train_X: New task training features (N, D).
            train_y: New task training labels (N,).
            val_X: Optional validation features.
            val_y: Optional validation labels.
            task_id: Task identifier.

        Returns:
            Dict with training metrics (loss, accuracy, etc.).
        """
        self.task_count += 1
        task_id = task_id or self.task_count
        logger.info(f"=== Training Task {task_id} ({len(train_y)} samples) ===")

        # ── Step 1: Construct merged dataset ─────────────────────────
        replay_X, replay_y = self.replay_buffer.sample(
            min(len(self.replay_buffer), len(train_y))
        )

        if len(replay_X) > 0:
            merged_X = np.concatenate([train_X, replay_X], axis=0)
            merged_y = np.concatenate([train_y, replay_y], axis=0)
            logger.info(
                f"  Merged: {len(train_y)} new + {len(replay_y)} replay "
                f"= {len(merged_y)} total"
            )
        else:
            merged_X, merged_y = train_X, train_y
            logger.info(f"  No replay data (first task)")

        train_loader = create_dataloader(
            merged_X, merged_y, batch_size=self.batch_size, shuffle=True
        )

        # ── Step 2: Optimise composite loss ──────────────────────────
        self.model.to(self.device)
        self.model.train()

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        metrics = {"task_id": task_id, "epoch_losses": []}
        best_loss = float("inf")

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            epoch_ce_loss = 0.0
            epoch_ewc_loss = 0.0
            correct = 0
            total = 0

            for features, labels in train_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                output = self.model(features)
                logits = output[0] if isinstance(output, tuple) else output
                ce_loss = F.cross_entropy(logits, labels)

                # EWC penalty: (λ/2) * Σ F_k * (θ_k - θ*_k)^2
                ewc_loss = self._compute_ewc_penalty()
                total_loss = ce_loss + ewc_loss

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0
                )
                optimizer.step()

                epoch_loss += total_loss.item() * features.size(0)
                epoch_ce_loss += ce_loss.item() * features.size(0)
                epoch_ewc_loss += ewc_loss.item() * features.size(0)
                _, predicted = logits.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

            avg_loss = epoch_loss / total
            avg_ce = epoch_ce_loss / total
            avg_ewc = epoch_ewc_loss / total
            accuracy = 100.0 * correct / total
            metrics["epoch_losses"].append(avg_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss

            if (epoch + 1) % 2 == 0 or epoch == 0:
                logger.info(
                    f"  Epoch {epoch+1}/{self.epochs}: "
                    f"Loss={avg_loss:.4f} (CE={avg_ce:.4f}, EWC={avg_ewc:.4f}), "
                    f"Acc={accuracy:.2f}%"
                )

        metrics["final_accuracy"] = accuracy
        metrics["final_loss"] = avg_loss

        # ── Step 3: Recompute Fisher and update running FIM ──────────
        fisher_loader = create_dataloader(
            train_X, train_y, batch_size=self.batch_size, shuffle=False
        )
        new_fisher = compute_fisher_diagonal_efficient(
            self.model, fisher_loader, self.device,
            num_samples=self.fisher_samples,
        )

        self.running_fisher = update_running_fisher(
            self.running_fisher, new_fisher, decay=self.fisher_decay
        )

        # Update reference parameters
        self.reference_params = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        logger.info("  Updated Fisher diagonal and reference parameters")

        # ── Step 4: Update replay buffer ─────────────────────────────
        self.replay_buffer.add_batch(train_X, train_y)
        logger.info(
            f"  Replay buffer: {len(self.replay_buffer)}/{self.buffer_size} "
            f"(total seen: {self.replay_buffer.count})"
        )

        # ── Step 5: Checkpoint ───────────────────────────────────────
        self.training_history.append(metrics)

        # Validation
        if val_X is not None and val_y is not None:
            val_acc = self._evaluate(val_X, val_y)
            metrics["val_accuracy"] = val_acc
            logger.info(f"  Validation accuracy: {val_acc:.2f}%")

        return metrics

    def _compute_ewc_penalty(self) -> torch.Tensor:
        """
        Compute EWC regularisation penalty.

        L_EWC = (λ/2) * Σ_k F_hat_k * (θ_k - θ*_k)^2
        """
        if not self.running_fisher or not self.reference_params:
            return torch.tensor(0.0, device=self.device)

        penalty = torch.tensor(0.0, device=self.device)
        for name, param in self.model.named_parameters():
            if name in self.running_fisher and name in self.reference_params:
                fisher = self.running_fisher[name].to(self.device)
                ref = self.reference_params[name].to(self.device)
                penalty += (fisher * (param - ref).pow(2)).sum()

        return (self.ewc_lambda / 2.0) * penalty

    def _evaluate(
        self, test_X: np.ndarray, test_y: np.ndarray
    ) -> float:
        """Evaluate model accuracy on test data."""
        self.model.eval()
        loader = create_dataloader(
            test_X, test_y, batch_size=self.batch_size, shuffle=False
        )
        correct = 0
        total = 0

        with torch.no_grad():
            for features, labels in loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                output = self.model(features)
                logits = output[0] if isinstance(output, tuple) else output
                _, predicted = logits.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        return 100.0 * correct / total if total > 0 else 0.0

    def evaluate_all_tasks(
        self, tasks: List[Dict[str, np.ndarray]]
    ) -> Dict[str, float]:
        """
        Evaluate on all tasks and compute CL metrics.

        Returns:
            Dict with per-task accuracies and aggregate metrics.
        """
        results = {}
        accuracies = []

        for task in tasks:
            task_id = task.get("split_id", task.get("dataset", "unknown"))
            acc = self._evaluate(task["test_X"], task["test_y"])
            results[f"task_{task_id}_accuracy"] = acc
            accuracies.append(acc)

        results["average_accuracy"] = np.mean(accuracies)
        return results

    def save_checkpoint(self, path: str) -> None:
        """Save model, Fisher, reference params, and replay buffer."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "running_fisher": self.running_fisher,
            "reference_params": self.reference_params,
            "task_count": self.task_count,
            "replay_buffer_features": self.replay_buffer.features[
                : self.replay_buffer.current_size
            ],
            "replay_buffer_labels": self.replay_buffer.labels[
                : self.replay_buffer.current_size
            ],
            "replay_buffer_count": self.replay_buffer.count,
            "training_history": self.training_history,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load checkpoint and restore all state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.running_fisher = checkpoint["running_fisher"]
        self.reference_params = checkpoint["reference_params"]
        self.task_count = checkpoint["task_count"]

        buf_features = checkpoint["replay_buffer_features"]
        buf_labels = checkpoint["replay_buffer_labels"]
        self.replay_buffer.features[: len(buf_labels)] = buf_features
        self.replay_buffer.labels[: len(buf_labels)] = buf_labels
        self.replay_buffer.current_size = len(buf_labels)
        self.replay_buffer.count = checkpoint["replay_buffer_count"]
        self.training_history = checkpoint.get("training_history", [])

        logger.info(f"Checkpoint loaded from {path}")
