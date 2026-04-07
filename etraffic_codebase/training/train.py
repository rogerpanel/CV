"""
Training pipeline for encrypted traffic intrusion detection models

Implements:
- FocalLoss for handling class imbalance (gamma=2.0, alpha=0.25)
- Full training loop with validation and early stopping
- Learning rate scheduling (exponential, step, cosine annealing)
- Gradient clipping and mixed precision training
- TensorBoard logging and checkpoint management

Reference: Paper Section 4.2 - Training Procedure
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from typing import Dict, Optional, List
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.metrics import compute_all_metrics
from utils.reproducibility import set_seed


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in encrypted traffic datasets.

    Downweights well-classified examples to focus training on hard,
    misclassified samples -- critical for imbalanced datasets where
    benign traffic vastly outnumbers attack traffic.

    L_FL = -alpha * (1 - p_t)^gamma * log(p_t)

    Reference:
        Lin et al. (2017) - Focal Loss for Dense Object Detection
        Paper Section 4.2 - Loss Function
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25,
                 weight: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.weight = weight

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(
            logits, targets, weight=self.weight, reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class Trainer:
    """
    Complete training pipeline for encrypted traffic models.

    Handles:
    - Model training with configurable optimizer and scheduler
    - Validation with comprehensive metrics
    - Early stopping based on validation loss
    - Checkpoint saving/loading
    - TensorBoard logging
    - Mixed precision training (AMP)
    """

    def __init__(self, model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: dict,
                 device: torch.device = None):
        """
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration dictionary
            device: Compute device
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model.to(self.device)

        # Loss function
        loss_fn = config.get('loss_function', 'focal_loss')
        if loss_fn == 'focal_loss':
            self.criterion = FocalLoss(
                gamma=config.get('focal_gamma', 2.0),
                alpha=config.get('focal_alpha', 0.25)
            )
        else:
            self.criterion = nn.CrossEntropyLoss(
                label_smoothing=config.get('label_smoothing', 0.0)
            )

        # Optimizer
        opt_name = config.get('optimizer', 'adam')
        lr = config.get('learning_rate', 0.001)
        wd = config.get('weight_decay', 1e-4)

        if opt_name == 'adam':
            self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == 'adamw':
            self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == 'sgd':
            self.optimizer = optim.SGD(
                model.parameters(), lr=lr, momentum=0.9, weight_decay=wd
            )
        else:
            self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        # LR scheduler
        sched_name = config.get('lr_scheduler', 'exponential')
        if sched_name == 'exponential':
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=config.get('lr_decay_rate', 0.95)
            )
        elif sched_name == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config.get('lr_decay_steps', 10),
                gamma=config.get('lr_decay_rate', 0.5)
            )
        elif sched_name == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.get('num_epochs', 100)
            )
        else:
            self.scheduler = None

        # Mixed precision
        self.use_amp = config.get('mixed_precision', False)
        self.scaler = GradScaler() if self.use_amp else None

        # Gradient clipping
        self.grad_clip = config.get('gradient_clip_value', 1.0)

        # Early stopping
        self.patience = config.get('early_stopping_patience', 10)
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # TensorBoard
        self.writer = None
        if config.get('use_tensorboard', False):
            try:
                from torch.utils.tensorboard import SummaryWriter
                log_dir = config.get('log_dir', './logs')
                self.writer = SummaryWriter(log_dir)
            except ImportError:
                pass

    def train(self, num_epochs: int = None) -> Dict[str, List]:
        """
        Full training loop.

        Args:
            num_epochs: Number of epochs (overrides config)

        Returns:
            Training history
        """
        if num_epochs is None:
            num_epochs = self.config.get('num_epochs', 100)

        history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
        }

        print(f"\nTraining on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print("=" * 60)

        for epoch in range(num_epochs):
            start_time = time.time()

            # Training phase
            train_metrics = self._train_epoch()

            # Validation phase
            val_metrics = self._validate()

            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Record history
            history['train_loss'].append(train_metrics['loss'])
            history['val_loss'].append(val_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['val_acc'].append(val_metrics['accuracy'])

            # TensorBoard logging
            if self.writer:
                self.writer.add_scalars('Loss', {
                    'train': train_metrics['loss'],
                    'val': val_metrics['loss']
                }, epoch)
                self.writer.add_scalars('Accuracy', {
                    'train': train_metrics['accuracy'],
                    'val': val_metrics['accuracy']
                }, epoch)

            elapsed = time.time() - start_time
            lr = self.optimizer.param_groups[0]['lr']

            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Train Loss: {train_metrics['loss']:.4f} "
                  f"Val Loss: {val_metrics['loss']:.4f} "
                  f"Val Acc: {val_metrics['accuracy']:.4f} "
                  f"LR: {lr:.6f} "
                  f"({elapsed:.1f}s)")

            # Early stopping
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                # Save best model
                if hasattr(self.model, 'save_checkpoint'):
                    self.model.save_checkpoint(
                        'checkpoints/best_model.pt', epoch,
                        optimizer_state=self.optimizer.state_dict(),
                        metrics=val_metrics
                    )
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break

        if self.writer:
            self.writer.close()

        return history

    def _train_epoch(self) -> Dict[str, float]:
        """Train one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in self.train_loader:
            if len(batch) == 3:
                x, _, y = batch
            else:
                x, y = batch

            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp and self.scaler:
                with autocast():
                    output = self.model(x)
                    loss = self.criterion(output, y)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(x)
                loss = self.criterion(output, y)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )
                self.optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = output.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        return {
            'loss': total_loss / total,
            'accuracy': correct / total
        }

    def _validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in self.val_loader:
                if len(batch) == 3:
                    x, _, y = batch
                else:
                    x, y = batch

                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                loss = self.criterion(output, y)

                total_loss += loss.item() * x.size(0)
                preds = output.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        return {
            'loss': total_loss / total,
            'accuracy': correct / total
        }
