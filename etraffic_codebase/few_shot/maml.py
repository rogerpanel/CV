"""
Model-Agnostic Meta-Learning (MAML) for encrypted traffic classification

Optimization-based meta-learning that finds model initialization points
enabling rapid adaptation to new attack types with few samples.

Reference:
    Paper Section 3.7 - Few-Shot Learning
    Finn et al. (2017) - Model-Agnostic Meta-Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
import copy


class MAMLModel(nn.Module):
    """
    Feed-forward model for MAML meta-learning.

    Simple architecture that supports fast gradient-based adaptation
    during the inner loop of MAML.
    """

    def __init__(self, input_dim: int = 88, hidden_dim: int = 256,
                 num_classes: int = 5, num_layers: int = 3,
                 dropout: float = 0.2):
        super().__init__()

        layers = []
        in_dim = input_dim
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.mean(dim=1)
        return self.network(x)


class MAML:
    """
    Model-Agnostic Meta-Learning.

    Outer loop: updates model initialization across tasks
    Inner loop: adapts model to each task via gradient steps
    """

    def __init__(self, model: nn.Module,
                 inner_lr: float = 0.01,
                 outer_lr: float = 0.001,
                 inner_steps: int = 5,
                 device: torch.device = None):
        self.model = model
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model.to(self.device)
        self.meta_optimizer = torch.optim.Adam(
            self.model.parameters(), lr=outer_lr
        )

    def inner_loop(self, support_x: torch.Tensor,
                   support_y: torch.Tensor) -> nn.Module:
        """
        Inner loop adaptation on support set.

        Returns:
            Adapted model (copy)
        """
        adapted_model = copy.deepcopy(self.model)
        adapted_optimizer = torch.optim.SGD(
            adapted_model.parameters(), lr=self.inner_lr
        )

        for _ in range(self.inner_steps):
            logits = adapted_model(support_x)
            loss = F.cross_entropy(logits, support_y)
            adapted_optimizer.zero_grad()
            loss.backward()
            adapted_optimizer.step()

        return adapted_model

    def outer_step(self, tasks: List[Tuple]) -> float:
        """
        Outer loop update across multiple tasks.

        Args:
            tasks: List of (support_x, support_y, query_x, query_y) tuples

        Returns:
            Average meta-loss
        """
        meta_loss = 0.0

        for support_x, support_y, query_x, query_y in tasks:
            # Inner loop adaptation
            adapted_model = self.inner_loop(support_x, support_y)

            # Evaluate on query set
            query_logits = adapted_model(query_x)
            task_loss = F.cross_entropy(query_logits, query_y)
            meta_loss += task_loss

        meta_loss /= len(tasks)

        # Outer loop update
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()

        return meta_loss.item()


class MAMLTrainer:
    """Trainer for MAML meta-learning on encrypted traffic."""

    def __init__(self, maml: MAML,
                 n_way: int = 5, k_shot: int = 5,
                 n_query: int = 15,
                 tasks_per_batch: int = 4):
        self.maml = maml
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.tasks_per_batch = tasks_per_batch

    def sample_task(self, features: np.ndarray,
                    labels: np.ndarray) -> Tuple:
        """Sample a single N-way K-shot task."""
        unique_classes = np.unique(labels)
        selected = np.random.choice(unique_classes, self.n_way, replace=False)

        support_x, support_y = [], []
        query_x, query_y = [], []

        for i, c in enumerate(selected):
            indices = np.where(labels == c)[0]
            chosen = np.random.choice(
                indices, self.k_shot + self.n_query, replace=False
            )
            support_x.append(features[chosen[:self.k_shot]])
            support_y.extend([i] * self.k_shot)
            query_x.append(features[chosen[self.k_shot:]])
            query_y.extend([i] * self.n_query)

        device = self.maml.device
        return (
            torch.FloatTensor(np.vstack(support_x)).to(device),
            torch.LongTensor(support_y).to(device),
            torch.FloatTensor(np.vstack(query_x)).to(device),
            torch.LongTensor(query_y).to(device),
        )

    def train(self, features: np.ndarray, labels: np.ndarray,
              num_iterations: int = 1000,
              verbose: bool = True) -> List[float]:
        """Train MAML across meta-learning iterations."""
        losses = []

        for it in range(num_iterations):
            tasks = [
                self.sample_task(features, labels)
                for _ in range(self.tasks_per_batch)
            ]

            loss = self.maml.outer_step(tasks)
            losses.append(loss)

            if verbose and (it + 1) % 100 == 0:
                print(f"Iteration [{it+1}/{num_iterations}] "
                      f"Meta-loss: {loss:.4f}")

        return losses
