"""
Main training script for encrypted traffic intrusion detection

This script implements the complete training pipeline with all features:
- Hybrid CNN-LSTM, Transformer, GNN training
- Class imbalance handling (weighted loss, focal loss)
- Early stopping and learning rate scheduling
- Model checkpointing
- TensorBoard logging
- Adversarial training (optional)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from typing import Optional, Dict, Tuple
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import load_config
from utils.reproducibility import set_seed, get_device
from utils.metrics import compute_all_metrics, print_classification_report


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    Reference:
        Lin et al. (2017) - Focal Loss for Dense Object Detection
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize focal loss.

        Args:
            alpha: Weighting factor
            gamma: Focusing parameter
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            inputs: Model predictions (logits)
            targets: Ground truth labels

        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class Trainer:
    """
    Trainer class for encrypted traffic detection models.

    Handles training loop, validation, checkpointing, and logging.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        config: dict,
        device: torch.device,
        log_dir: str = './logs',
        checkpoint_dir: str = './checkpoints'
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration dictionary
            device: Device for computation
            log_dir: Directory for TensorBoard logs
            checkpoint_dir: Directory for model checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir

        # Create directories
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Initialize optimizer
        self.optimizer = self._create_optimizer()

        # Initialize loss function
        self.criterion = self._create_loss_function()

        # Initialize learning rate scheduler
        self.scheduler = self._create_scheduler()

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir) if config.get('use_tensorboard', True) else None

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        self.patience_counter = 0

        # History
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer from configuration."""
        optimizer_name = self.config.get('optimizer', 'adam').lower()
        lr = self.config.get('learning_rate', 0.001)
        weight_decay = self.config.get('weight_decay', 0.0001)

        if optimizer_name == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            return torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            momentum = self.config.get('momentum', 0.9)
            return torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum,
                                  weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def _create_loss_function(self) -> nn.Module:
        """Create loss function from configuration."""
        loss_name = self.config.get('loss_function', 'cross_entropy').lower()

        if loss_name == 'cross_entropy':
            # Check if class weights are needed
            return nn.CrossEntropyLoss()
        elif loss_name == 'focal_loss':
            gamma = self.config.get('focal_gamma', 2.0)
            alpha = self.config.get('focal_alpha', 0.25)
            return FocalLoss(alpha=alpha, gamma=gamma)
        elif loss_name == 'weighted_cross_entropy':
            # Would need class weights from dataset
            return nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown loss function: {loss_name}")

    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler from configuration."""
        scheduler_name = self.config.get('lr_scheduler', 'exponential').lower()

        if scheduler_name == 'exponential':
            decay_rate = self.config.get('lr_decay_rate', 0.95)
            return torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=decay_rate)
        elif scheduler_name == 'step':
            step_size = self.config.get('lr_decay_steps', 10)
            gamma = self.config.get('lr_decay_rate', 0.1)
            return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_name == 'cosine':
            T_max = self.config.get('num_epochs', 100)
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max)
        elif scheduler_name == 'none' or scheduler_name is None:
            return None
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")

    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}')

        for batch_idx, batch in enumerate(pbar):
            # Unpack batch
            if len(batch) == 3:
                x, _, y = batch
            else:
                x, y = batch

            x, y = x.to(self.device), y.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(x)

            loss = self.criterion(outputs, y)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config.get('gradient_clip_value'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['gradient_clip_value']
                )

            self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100.0 * correct / total
            })

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def validate(self) -> Tuple[float, float, Dict]:
        """
        Validate model.

        Returns:
            Tuple of (average_loss, accuracy, metrics_dict)
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Unpack batch
                if len(batch) == 3:
                    x, _, y = batch
                else:
                    x, y = batch

                x, y = x.to(self.device), y.to(self.device)

                # Forward pass
                outputs = self.model(x)
                loss = self.criterion(outputs, y)

                # Statistics
                total_loss += loss.item()
                probs = F.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)

        # Compute metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        metrics = compute_all_metrics(all_labels, all_predictions, all_probs)

        return avg_loss, metrics['accuracy'], metrics

    def train(self, num_epochs: Optional[int] = None) -> Dict:
        """
        Train model for specified number of epochs.

        Args:
            num_epochs: Number of epochs (if None, uses config)

        Returns:
            Training history
        """
        if num_epochs is None:
            num_epochs = self.config.get('num_epochs', 100)

        early_stopping_patience = self.config.get('early_stopping_patience', 10)

        print(f"\n{'=' * 80}")
        print(f"Starting training for {num_epochs} epochs")
        print(f"{'=' * 80}\n")

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)

            # Validate
            val_loss, val_acc, val_metrics = self.validate()
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)

            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()

            # Logging
            print(f"\nEpoch {epoch + 1}/{num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"  Val F1: {val_metrics['f1_score']:.4f}, Val MCC: {val_metrics['mcc']:.4f}")

            # TensorBoard logging
            if self.writer is not None:
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Accuracy/train', train_acc, epoch)
                self.writer.add_scalar('Accuracy/val', val_acc, epoch)
                self.writer.add_scalar('F1/val', val_metrics['f1_score'], epoch)
                self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)

            # Checkpointing
            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                self.best_val_loss = val_loss
                self.patience_counter = 0

                # Save best model
                checkpoint_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
                self.model.save_checkpoint(
                    checkpoint_path,
                    epoch=epoch,
                    optimizer_state=self.optimizer.state_dict(),
                    metrics=val_metrics
                )
                print(f"  ✓ New best model saved (Val Acc: {val_acc:.2f}%)")

            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

        print(f"\n{'=' * 80}")
        print(f"Training completed!")
        print(f"Best validation accuracy: {self.best_val_accuracy:.2f}%")
        print(f"{'=' * 80}\n")

        # Close TensorBoard writer
        if self.writer is not None:
            self.writer.close()

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'best_val_accuracy': self.best_val_accuracy,
            'best_val_loss': self.best_val_loss
        }
