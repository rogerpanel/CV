"""
Ensemble models for encrypted traffic intrusion detection

This module implements ensemble aggregation strategies that combine predictions
from multiple models (CNN-LSTM, Transformer, GNN) to achieve superior performance.

The paper reports ensemble achieving 99.92% accuracy on CICIDS2017 encrypted traffic,
improving upon individual models through prediction diversity.

Supported strategies:
- Hard voting: Majority vote
- Soft voting: Average probabilities
- Weighted voting: Learned weights based on validation performance
- Stacking: Meta-learner trained on base model predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
from .base import BaseModel


class EnsembleClassifier(BaseModel):
    """
    Ensemble classifier combining multiple base models.

    Achieves 99.92% accuracy on CICIDS2017 encrypted traffic by combining
    predictions from CNN-LSTM, Transformer, and GNN architectures.

    The ensemble leverages diversity in model architectures to improve
    robustness and accuracy beyond individual models.
    """

    def __init__(
        self,
        models: List[nn.Module],
        num_classes: int,
        voting_strategy: str = 'soft',
        model_weights: Optional[List[float]] = None,
        use_stacking: bool = False,
        meta_learner: Optional[nn.Module] = None
    ):
        """
        Initialize ensemble classifier.

        Args:
            models: List of base models to ensemble
            num_classes: Number of output classes
            voting_strategy: Voting strategy ('hard', 'soft', 'weighted')
            model_weights: Weights for weighted voting (must sum to 1)
            use_stacking: Whether to use stacking with meta-learner
            meta_learner: Meta-learner model for stacking (if use_stacking=True)
        """
        super(EnsembleClassifier, self).__init__()

        self.models = nn.ModuleList(models)
        self.num_classes = num_classes
        self.num_models = len(models)
        self.voting_strategy = voting_strategy
        self.use_stacking = use_stacking

        # Set model weights
        if model_weights is None:
            # Equal weights
            self.model_weights = [1.0 / self.num_models] * self.num_models
        else:
            assert len(model_weights) == self.num_models, \
                "Number of weights must match number of models"
            assert abs(sum(model_weights) - 1.0) < 1e-6, \
                "Model weights must sum to 1.0"
            self.model_weights = model_weights

        # Convert to tensor for efficient computation
        self.register_buffer('weights', torch.FloatTensor(self.model_weights))

        # Meta-learner for stacking
        if use_stacking:
            if meta_learner is None:
                # Default meta-learner: logistic regression
                self.meta_learner = nn.Linear(num_classes * self.num_models, num_classes)
            else:
                self.meta_learner = meta_learner
        else:
            self.meta_learner = None

    def forward(self, x: torch.Tensor, return_individual: bool = False) -> torch.Tensor:
        """
        Forward pass through ensemble.

        Args:
            x: Input tensor
            return_individual: If True, return individual model predictions

        Returns:
            Ensemble predictions (and optionally individual predictions)
        """
        # Get predictions from all models
        predictions = []
        for model in self.models:
            with torch.set_grad_enabled(self.training):
                logits = model(x)
                predictions.append(logits)

        # Stack predictions: (num_models, batch_size, num_classes)
        predictions = torch.stack(predictions)

        # Apply voting strategy
        if self.use_stacking and self.meta_learner is not None:
            # Stacking: concatenate all predictions and pass to meta-learner
            # (batch_size, num_models * num_classes)
            stacked = predictions.permute(1, 0, 2).reshape(predictions.size(1), -1)
            ensemble_logits = self.meta_learner(stacked)

        elif self.voting_strategy == 'hard':
            # Hard voting: majority vote on predicted classes
            # Convert logits to class predictions
            class_predictions = torch.argmax(predictions, dim=2)  # (num_models, batch_size)

            # Count votes for each class
            batch_size = class_predictions.size(1)
            ensemble_logits = torch.zeros(batch_size, self.num_classes, device=x.device)

            for i in range(batch_size):
                votes = class_predictions[:, i]
                for class_idx in range(self.num_classes):
                    ensemble_logits[i, class_idx] = (votes == class_idx).sum().float()

        elif self.voting_strategy == 'soft':
            # Soft voting: average predicted probabilities
            probabilities = F.softmax(predictions, dim=2)  # (num_models, batch_size, num_classes)
            ensemble_probs = probabilities.mean(dim=0)  # (batch_size, num_classes)
            # Convert back to logits
            ensemble_logits = torch.log(ensemble_probs + 1e-10)

        elif self.voting_strategy == 'weighted':
            # Weighted voting: weighted average with learned/specified weights
            probabilities = F.softmax(predictions, dim=2)  # (num_models, batch_size, num_classes)

            # Apply weights: (num_models, 1, 1) * (num_models, batch_size, num_classes)
            weights_expanded = self.weights.view(-1, 1, 1)
            weighted_probs = probabilities * weights_expanded

            # Sum over models
            ensemble_probs = weighted_probs.sum(dim=0)  # (batch_size, num_classes)

            # Convert back to logits
            ensemble_logits = torch.log(ensemble_probs + 1e-10)

        else:
            raise ValueError(f"Unknown voting strategy: {self.voting_strategy}")

        if return_individual:
            return ensemble_logits, predictions
        else:
            return ensemble_logits

    def optimize_weights(
        self,
        val_loader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> List[float]:
        """
        Optimize ensemble weights based on validation performance.

        Uses grid search to find optimal weights that maximize validation accuracy.

        Args:
            val_loader: Validation data loader
            device: Device to run evaluation on

        Returns:
            Optimized weights for each model
        """
        print("Optimizing ensemble weights...")

        # Get predictions from all models on validation set
        all_predictions = [[] for _ in range(self.num_models)]
        all_labels = []

        self.eval()
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    x, _, y = batch
                else:
                    x, y = batch

                x, y = x.to(device), y.to(device)

                for i, model in enumerate(self.models):
                    logits = model(x)
                    probs = F.softmax(logits, dim=1)
                    all_predictions[i].append(probs.cpu().numpy())

                all_labels.append(y.cpu().numpy())

        # Concatenate all batches
        all_predictions = [np.vstack(preds) for preds in all_predictions]
        all_labels = np.concatenate(all_labels)

        # Grid search for optimal weights
        best_accuracy = 0
        best_weights = [1.0 / self.num_models] * self.num_models

        # Simple grid search (can be improved with more sophisticated optimization)
        weight_options = np.arange(0, 1.1, 0.1)

        for w1 in weight_options:
            for w2 in weight_options:
                if self.num_models == 2:
                    weights = [w1, 1 - w1]
                elif self.num_models == 3:
                    w3 = 1 - w1 - w2
                    if w3 < 0 or w3 > 1:
                        continue
                    weights = [w1, w2, w3]
                else:
                    # For more models, use equal weights for remaining
                    remaining = 1 - w1 - w2
                    if remaining < 0:
                        continue
                    weights = [w1, w2] + [remaining / (self.num_models - 2)] * (self.num_models - 2)

                # Compute ensemble predictions with these weights
                ensemble_probs = sum(w * pred for w, pred in zip(weights, all_predictions))
                ensemble_preds = np.argmax(ensemble_probs, axis=1)

                # Compute accuracy
                accuracy = (ensemble_preds == all_labels).mean()

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_weights = weights

        print(f"Optimized weights: {best_weights}")
        print(f"Validation accuracy: {best_accuracy * 100:.2f}%")

        # Update model weights
        self.model_weights = best_weights
        self.weights = torch.FloatTensor(best_weights).to(self.weights.device)

        return best_weights

    def get_config(self) -> dict:
        """Get ensemble configuration."""
        return {
            'num_models': self.num_models,
            'num_classes': self.num_classes,
            'voting_strategy': self.voting_strategy,
            'model_weights': self.model_weights,
            'use_stacking': self.use_stacking
        }


class StackingMetaLearner(nn.Module):
    """
    Meta-learner for stacking ensemble.

    Learns to optimally combine base model predictions through a neural network.
    """

    def __init__(
        self,
        num_base_models: int,
        num_classes: int,
        hidden_dim: int = 128,
        dropout: float = 0.3
    ):
        """
        Initialize stacking meta-learner.

        Args:
            num_base_models: Number of base models in ensemble
            num_classes: Number of output classes
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
        """
        super(StackingMetaLearner, self).__init__()

        input_dim = num_base_models * num_classes

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Concatenated base model predictions (batch_size, num_base_models * num_classes)

        Returns:
            Meta-predictions (batch_size, num_classes)
        """
        return self.network(x)
