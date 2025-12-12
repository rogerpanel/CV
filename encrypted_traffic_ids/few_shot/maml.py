"""
Model-Agnostic Meta-Learning (MAML) for Encrypted Traffic Classification

Implements meta-learning through gradient-based optimization for fast adaptation
to new attack types with limited samples.

Key Concepts:
- Learn model initialization θ that enables fast adaptation
- Inner loop: Task-specific adaptation via gradient descent
- Outer loop: Meta-optimization across task distribution

References:
    Finn et al. (2017) - Model-Agnostic Meta-Learning for Fast Adaptation
    Paper Section 3.5 - Few-Shot Meta-Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import copy
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base import BaseModel


class MAMLModel(nn.Module):
    """
    Base model for MAML on encrypted traffic.

    Simple feedforward network that can be quickly adapted.
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dims: List[int] = [256, 128],
        num_classes: int = 2
    ):
        """
        Initialize MAML model.

        Args:
            input_dim: Number of input features per packet
            hidden_dims: Hidden layer dimensions
            num_classes: Number of output classes
        """
        super(MAMLModel, self).__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes

        # Build network
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch_size, seq_len, input_dim) or (batch_size, input_dim)

        Returns:
            Logits (batch_size, num_classes)
        """
        # If 3D input, average over sequence
        if len(x.shape) == 3:
            x = x.mean(dim=1)

        return self.network(x)

    def clone(self):
        """Create a deep copy of the model."""
        return copy.deepcopy(self)


class MAML:
    """
    MAML implementation for few-shot encrypted traffic classification.

    Meta-learns model initialization for fast adaptation to new attack types.
    """

    def __init__(
        self,
        model: MAMLModel,
        device: torch.device,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        inner_steps: int = 5,
        first_order: bool = False
    ):
        """
        Initialize MAML.

        Args:
            model: Base model to meta-train
            device: Compute device
            inner_lr: Inner loop learning rate (task adaptation)
            outer_lr: Outer loop learning rate (meta-optimization)
            inner_steps: Number of inner gradient steps
            first_order: Use first-order MAML (faster but less accurate)
        """
        self.model = model.to(device)
        self.device = device
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.first_order = first_order

        self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=outer_lr)
        self.criterion = nn.CrossEntropyLoss()

    def inner_loop(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        adapted_params: Optional[Dict] = None
    ) -> Tuple[Dict, float]:
        """
        Perform inner loop adaptation on support set.

        θ'_i = θ - α∇_θ L_T_i(f_θ)

        Args:
            support_x: Support set inputs
            support_y: Support set labels
            adapted_params: Starting parameters (None = use meta-parameters)

        Returns:
            Tuple of (adapted_params, final_loss)
        """
        # Clone model for task-specific adaptation
        if adapted_params is None:
            adapted_model = self.model.clone()
        else:
            adapted_model = self.model.clone()
            adapted_model.load_state_dict(adapted_params)

        adapted_model.train()

        # Inner loop optimization
        for step in range(self.inner_steps):
            # Forward pass
            logits = adapted_model(support_x)
            loss = self.criterion(logits, support_y)

            # Compute gradients
            grads = torch.autograd.grad(
                loss,
                adapted_model.parameters(),
                create_graph=not self.first_order
            )

            # Manual SGD update
            adapted_params = {}
            for (name, param), grad in zip(adapted_model.named_parameters(), grads):
                adapted_params[name] = param - self.inner_lr * grad

            # Update model
            adapted_model.load_state_dict(adapted_params)

        # Compute final loss after adaptation
        logits = adapted_model(support_x)
        final_loss = self.criterion(logits, support_y)

        return adapted_model.state_dict(), final_loss.item()

    def outer_loop(
        self,
        tasks: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> float:
        """
        Perform outer loop meta-update across task batch.

        θ = θ - β∇_θ Σ_i L_T_i(f_θ'_i)

        Args:
            tasks: List of (support_x, support_y, query_x, query_y) tuples

        Returns:
            Meta-loss across tasks
        """
        self.meta_optimizer.zero_grad()
        meta_loss = 0.0

        for support_x, support_y, query_x, query_y in tasks:
            # Move to device
            support_x = support_x.to(self.device)
            support_y = support_y.to(self.device)
            query_x = query_x.to(self.device)
            query_y = query_y.to(self.device)

            # Inner loop adaptation
            adapted_params, _ = self.inner_loop(support_x, support_y)

            # Create adapted model
            adapted_model = self.model.clone()
            adapted_model.load_state_dict(adapted_params)

            # Compute query loss with adapted model
            query_logits = adapted_model(query_x)
            query_loss = self.criterion(query_logits, query_y)

            meta_loss += query_loss

        # Meta-optimization
        meta_loss = meta_loss / len(tasks)
        meta_loss.backward()
        self.meta_optimizer.step()

        return meta_loss.item()

    def adapt(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        adaptation_steps: Optional[int] = None
    ) -> nn.Module:
        """
        Adapt model to new task using support set.

        Args:
            support_x: Support set inputs
            support_y: Support set labels
            adaptation_steps: Number of adaptation steps (uses inner_steps if None)

        Returns:
            Adapted model
        """
        if adaptation_steps is None:
            adaptation_steps = self.inner_steps

        # Store original inner_steps
        original_steps = self.inner_steps
        self.inner_steps = adaptation_steps

        # Adapt
        support_x = support_x.to(self.device)
        support_y = support_y.to(self.device)

        adapted_params, _ = self.inner_loop(support_x, support_y)

        # Create adapted model
        adapted_model = self.model.clone()
        adapted_model.load_state_dict(adapted_params)

        # Restore original steps
        self.inner_steps = original_steps

        return adapted_model


class MAMLTrainer:
    """
    Trainer for MAML on encrypted traffic datasets.
    """

    def __init__(
        self,
        maml: MAML,
        n_way: int = 5,
        k_shot: int = 5,
        n_query: int = 15,
        task_batch_size: int = 4
    ):
        """
        Initialize MAML trainer.

        Args:
            maml: MAML instance
            n_way: Number of classes per task
            k_shot: Number of support examples per class
            n_query: Number of query examples per class
            task_batch_size: Number of tasks per meta-update
        """
        self.maml = maml
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.task_batch_size = task_batch_size

        self.meta_train_losses = []
        self.meta_val_accuracies = []

    def sample_task(
        self,
        dataset: Dataset,
        n_way: int,
        k_shot: int,
        n_query: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a single task from dataset.

        Args:
            dataset: Dataset with .data and .labels
            n_way: Number of classes
            k_shot: Support examples per class
            n_query: Query examples per class

        Returns:
            Tuple of (support_x, support_y, query_x, query_y)
        """
        # Get unique classes
        all_classes = np.unique(dataset.labels)

        # Sample N classes
        task_classes = np.random.choice(all_classes, n_way, replace=False)

        support_x = []
        support_y = []
        query_x = []
        query_y = []

        for idx, class_id in enumerate(task_classes):
            # Get class examples
            class_indices = np.where(dataset.labels == class_id)[0]

            # Sample K+Q examples
            selected = np.random.choice(class_indices, k_shot + n_query, replace=False)

            # Split
            support_indices = selected[:k_shot]
            query_indices = selected[k_shot:]

            support_x.append(dataset.data[support_indices])
            support_y.extend([idx] * k_shot)

            query_x.append(dataset.data[query_indices])
            query_y.extend([idx] * n_query)

        # Convert to tensors
        support_x = torch.FloatTensor(np.concatenate(support_x, axis=0))
        support_y = torch.LongTensor(support_y)
        query_x = torch.FloatTensor(np.concatenate(query_x, axis=0))
        query_y = torch.LongTensor(query_y)

        return support_x, support_y, query_x, query_y

    def evaluate_task(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        query_y: torch.Tensor
    ) -> float:
        """
        Evaluate on a single task.

        Args:
            support_x, support_y: Support set
            query_x, query_y: Query set

        Returns:
            Task accuracy
        """
        # Adapt to task
        adapted_model = self.maml.adapt(support_x, support_y)
        adapted_model.eval()

        # Evaluate on query set
        with torch.no_grad():
            query_x = query_x.to(self.maml.device)
            query_y = query_y.to(self.maml.device)

            logits = adapted_model(query_x)
            predictions = logits.argmax(dim=1)
            accuracy = (predictions == query_y).float().mean().item()

        return accuracy

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        n_iterations: int = 10000,
        val_interval: int = 500,
        verbose: bool = True
    ) -> Dict:
        """
        Meta-train MAML.

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            n_iterations: Number of meta-training iterations
            val_interval: Validation frequency
            verbose: Print progress

        Returns:
            Training history
        """
        if verbose:
            print(f"\n{'=' * 80}")
            print(f"Meta-Training MAML ({self.n_way}-way {self.k_shot}-shot)")
            print(f"{'=' * 80}\n")

        best_val_accuracy = 0.0

        pbar = tqdm(range(n_iterations), desc="Meta-Training")

        for iteration in pbar:
            # Sample task batch
            tasks = []
            for _ in range(self.task_batch_size):
                task = self.sample_task(train_dataset, self.n_way, self.k_shot, self.n_query)
                tasks.append(task)

            # Meta-update
            meta_loss = self.maml.outer_loop(tasks)
            self.meta_train_losses.append(meta_loss)

            pbar.set_postfix({'meta_loss': meta_loss})

            # Validation
            if (iteration + 1) % val_interval == 0:
                val_accuracies = []
                for _ in range(100):  # 100 validation tasks
                    val_task = self.sample_task(val_dataset, self.n_way, self.k_shot, self.n_query)
                    val_acc = self.evaluate_task(*val_task)
                    val_accuracies.append(val_acc)

                mean_val_acc = np.mean(val_accuracies)
                std_val_acc = np.std(val_accuracies)
                self.meta_val_accuracies.append(mean_val_acc)

                if verbose:
                    print(f"\nIteration {iteration + 1}/{n_iterations}:")
                    print(f"  Meta Loss: {meta_loss:.4f}")
                    print(f"  Val Accuracy: {mean_val_acc * 100:.2f}% ± {std_val_acc * 100:.2f}%")

                if mean_val_acc > best_val_accuracy:
                    best_val_accuracy = mean_val_acc
                    if verbose:
                        print(f"  ✓ New best validation accuracy!")

        if verbose:
            print(f"\n{'=' * 80}")
            print(f"Meta-training completed!")
            print(f"Best validation accuracy: {best_val_accuracy * 100:.2f}%")
            print(f"{'=' * 80}\n")

        return {
            'meta_train_losses': self.meta_train_losses,
            'meta_val_accuracies': self.meta_val_accuracies,
            'best_val_accuracy': best_val_accuracy
        }
