"""
Training Script for PQ-IDPS

Orchestrates the complete training pipeline:
1. Load configuration
2. Initialize model, datasets, optimizer
3. Train with adversarial examples and quantum noise
4. Evaluate on validation set
5. Save checkpoints and logs

Usage:
    python training/train_pq_idps.py --config config/pq_idps_config.yaml
    python training/train_pq_idps.py --config config/pq_idps_config.yaml --resume checkpoints/last.pth
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import argparse
import yaml
import os
import sys
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
from tqdm import tqdm
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.pq_idps import PQIDPS, create_pq_idps
from data.pqc_datasets import get_pqc_dataloaders
from defense.adversarial_defense import (
    QuantumNoiseInjection,
    RandomizedSmoothing,
    AdversarialTrainer
)


class PQIDPSTrainer:
    """
    Trainer for PQ-IDPS model.

    Handles training loop, evaluation, checkpointing, and logging.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config['training']['device']
                                    if torch.cuda.is_available() else 'cpu')

        print(f"Using device: {self.device}")

        # Set random seeds for reproducibility
        self._set_seeds(config.get('seed', 42))

        # Create model
        print("Creating PQ-IDPS model...")
        self.model = create_pq_idps(config).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")

        # Load datasets
        print(f"\nLoading {config['data']['dataset']} dataset...")
        self.train_loader, self.val_loader, self.test_loader = get_pqc_dataloaders(
            dataset_name=config['data']['dataset'],
            data_dir=config['data']['data_dir'],
            batch_size=config['training']['batch_size'],
            num_workers=config['training']['num_workers'],
            max_packets=config['data']['max_packets']
        )

        # Create optimizer
        self.optimizer = self._create_optimizer(config['training'])

        # Create learning rate scheduler
        self.scheduler = self._create_scheduler(config['training'])

        # Defense mechanisms
        self.quantum_noise = QuantumNoiseInjection(
            **config['defense']['quantum_noise']
        ) if config['defense']['quantum_noise']['enable'] else None

        self.randomized_smoothing = RandomizedSmoothing(
            **config['defense']['randomized_smoothing']
        ) if config['defense']['randomized_smoothing']['enable'] else None

        self.adversarial_trainer = AdversarialTrainer(
            model=self.model,
            epsilon=config['adversarial']['epsilon'],
            alpha=config['adversarial']['alpha'],
            num_iterations=config['adversarial']['num_iterations'],
            grover_speedup=config['adversarial']['grover_speedup'],
            adversarial_ratio=config['adversarial']['adversarial_ratio']
        ) if config['adversarial']['enable'] else None

        # Logging
        self.writer = SummaryWriter(log_dir=config['logging']['tensorboard_dir'])

        # Checkpointing
        self.checkpoint_dir = Path(config['checkpoint']['save_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_acc = 0.0
        self.start_epoch = 0

    def _set_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        if self.config.get('deterministic', True):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _create_optimizer(self, config: Dict) -> torch.optim.Optimizer:
        """Create optimizer."""
        if config['optimizer'].lower() == 'adamw':
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=config['learning_rate'],
                weight_decay=config['weight_decay'],
                betas=(config['beta1'], config['beta2']),
                eps=config['epsilon']
            )
        elif config['optimizer'].lower() == 'adam':
            return torch.optim.Adam(
                self.model.parameters(),
                lr=config['learning_rate'],
                weight_decay=config['weight_decay']
            )
        elif config['optimizer'].lower() == 'sgd':
            return torch.optim.SGD(
                self.model.parameters(),
                lr=config['learning_rate'],
                momentum=0.9,
                weight_decay=config['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer: {config['optimizer']}")

    def _create_scheduler(self, config: Dict):
        """Create learning rate scheduler."""
        if config['scheduler'] == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config['epochs'],
                eta_min=config.get('min_lr', 1e-6)
            )
        elif config['scheduler'] == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.1
            )
        elif config['scheduler'] == 'linear':
            return torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.01,
                total_iters=config['epochs']
            )
        else:
            return None

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            packet_sequences = batch['packet_sequence'].to(self.device)
            statistical_features = batch['statistical_features'].to(self.device)
            labels = batch['label'].to(self.device)

            # Apply quantum noise injection
            if self.quantum_noise is not None:
                statistical_features = self.quantum_noise(statistical_features)

            # Apply randomized smoothing
            if self.randomized_smoothing is not None:
                packet_sequences = self.randomized_smoothing(packet_sequences)
                statistical_features = self.randomized_smoothing(statistical_features)

            # Adversarial training
            if self.adversarial_trainer is not None:
                metrics = self.adversarial_trainer.train_step(
                    packet_sequences,
                    statistical_features,
                    labels,
                    self.optimizer
                )
                loss = metrics['loss']
                acc = metrics['acc']

            else:
                # Standard training
                self.optimizer.zero_grad()

                logits, auxiliary = self.model(
                    packet_sequences,
                    statistical_features,
                    return_auxiliary=True
                )

                loss = F.cross_entropy(logits, labels)

                loss.backward()

                # Gradient clipping
                if self.config['training'].get('max_grad_norm', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['max_grad_norm']
                    )

                self.optimizer.step()

                # Compute accuracy
                preds = torch.argmax(logits, dim=-1)
                acc = (preds == labels).float().mean().item()

            # Update metrics
            total_loss += loss
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss:.4f}',
                'acc': f'{acc:.3f}'
            })

            # Log to TensorBoard
            if batch_idx % self.config['logging']['log_interval'] == 0:
                global_step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('train/loss', loss, global_step)
                self.writer.add_scalar('train/accuracy', acc, global_step)

                if 'auxiliary' in locals():
                    self.writer.add_scalar('train/classical_weight',
                                           auxiliary['classical_weight'], global_step)
                    self.writer.add_scalar('train/quantum_weight',
                                           auxiliary['quantum_weight'], global_step)

        avg_loss = total_loss / len(self.train_loader)
        avg_acc = correct / total

        return {
            'loss': avg_loss,
            'accuracy': avg_acc
        }

    @torch.no_grad()
    def evaluate(self, loader, split='val') -> Dict[str, float]:
        """Evaluate model."""
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        # Protocol-wise accuracy
        protocol_correct = {0: 0, 1: 0, 2: 0}
        protocol_total = {0: 0, 1: 0, 2: 0}

        # Pathway accuracies
        total_classical_acc = 0.0
        total_quantum_acc = 0.0

        pbar = tqdm(loader, desc=f"Evaluating {split}")

        for batch in pbar:
            packet_sequences = batch['packet_sequence'].to(self.device)
            statistical_features = batch['statistical_features'].to(self.device)
            labels = batch['label'].to(self.device)
            protocol_types = batch['protocol_type']

            # Forward pass
            logits, auxiliary = self.model(
                packet_sequences,
                statistical_features,
                return_auxiliary=True
            )

            loss = F.cross_entropy(logits, labels)

            # Predictions
            preds = torch.argmax(logits, dim=-1)

            # Update metrics
            total_loss += loss.item()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Protocol-wise accuracy
            for ptype in [0, 1, 2]:
                mask = protocol_types == ptype
                if mask.sum() > 0:
                    protocol_correct[ptype] += (preds[mask] == labels[mask]).sum().item()
                    protocol_total[ptype] += mask.sum().item()

            # Pathway accuracies
            pathway_accs = self.model.compute_pathway_accuracies(
                packet_sequences,
                statistical_features,
                labels
            )
            total_classical_acc += pathway_accs['classical_acc']
            total_quantum_acc += pathway_accs['quantum_acc']

        avg_loss = total_loss / len(loader)
        avg_acc = correct / total

        # Protocol-wise accuracies
        protocol_accs = {
            ptype: protocol_correct[ptype] / protocol_total[ptype]
            if protocol_total[ptype] > 0 else 0.0
            for ptype in [0, 1, 2]
        }

        # Average pathway accuracies
        avg_classical_acc = total_classical_acc / len(loader)
        avg_quantum_acc = total_quantum_acc / len(loader)

        return {
            'loss': avg_loss,
            'accuracy': avg_acc,
            'classical_acc': avg_classical_acc,
            'quantum_acc': avg_quantum_acc,
            'protocol_acc_classical': protocol_accs[0],
            'protocol_acc_hybrid': protocol_accs[1],
            'protocol_acc_pure_pqc': protocol_accs[2]
        }

    def train(self):
        """Complete training loop."""
        print("\nStarting training...")
        print(f"Total epochs: {self.config['training']['epochs']}")
        print(f"Batch size: {self.config['training']['batch_size']}")
        print(f"Learning rate: {self.config['training']['learning_rate']}")
        print()

        for epoch in range(self.start_epoch, self.config['training']['epochs']):
            # Train
            train_metrics = self.train_epoch(epoch)

            # Evaluate
            val_metrics = self.evaluate(self.val_loader, split='val')

            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Log to TensorBoard
            self.writer.add_scalar('epoch/train_loss', train_metrics['loss'], epoch)
            self.writer.add_scalar('epoch/train_accuracy', train_metrics['accuracy'], epoch)
            self.writer.add_scalar('epoch/val_loss', val_metrics['loss'], epoch)
            self.writer.add_scalar('epoch/val_accuracy', val_metrics['accuracy'], epoch)
            self.writer.add_scalar('epoch/val_classical_acc', val_metrics['classical_acc'], epoch)
            self.writer.add_scalar('epoch/val_quantum_acc', val_metrics['quantum_acc'], epoch)
            self.writer.add_scalar('epoch/learning_rate',
                                   self.optimizer.param_groups[0]['lr'], epoch)

            # Print summary
            print(f"\nEpoch {epoch}/{self.config['training']['epochs']}:")
            print(f"  Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.3f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.3f}")
            print(f"  Val Classical: {val_metrics['classical_acc']:.3f}, Quantum: {val_metrics['quantum_acc']:.3f}")
            print(f"  Val by Protocol - Classical: {val_metrics['protocol_acc_classical']:.3f}, "
                  f"Hybrid: {val_metrics['protocol_acc_hybrid']:.3f}, "
                  f"Pure PQC: {val_metrics['protocol_acc_pure_pqc']:.3f}")

            # Save checkpoint
            if (epoch + 1) % self.config['checkpoint']['save_interval'] == 0:
                self.save_checkpoint(epoch, val_metrics['accuracy'], is_best=False)

            # Save best model
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.save_checkpoint(epoch, val_metrics['accuracy'], is_best=True)
                print(f"  ✓ New best model! Val Acc: {self.best_val_acc:.3f}")

        print("\n" + "=" * 60)
        print("Training completed!")
        print(f"Best validation accuracy: {self.best_val_acc:.3f}")

        # Final evaluation on test set
        print("\nEvaluating on test set...")
        test_metrics = self.evaluate(self.test_loader, split='test')
        print(f"Test Accuracy: {test_metrics['accuracy']:.3f}")
        print(f"Test Classical: {test_metrics['classical_acc']:.3f}, Quantum: {test_metrics['quantum_acc']:.3f}")

        self.writer.close()

    def save_checkpoint(self, epoch: int, val_acc: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_acc': val_acc,
            'best_val_acc': self.best_val_acc,
            'config': self.config
        }

        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)

        # Keep only last N checkpoints
        if self.config['checkpoint'].get('keep_last_n', 0) > 0:
            checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_epoch_*.pth'))
            if len(checkpoints) > self.config['checkpoint']['keep_last_n']:
                for old_ckpt in checkpoints[:-self.config['checkpoint']['keep_last_n']]:
                    old_ckpt.unlink()


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description='Train PQ-IDPS')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("PQ-IDPS Training")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Experiment: {config['experiment']['name']}")
    print()

    # Create trainer
    trainer = PQIDPSTrainer(config)

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if trainer.scheduler and checkpoint['scheduler_state_dict']:
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.start_epoch = checkpoint['epoch'] + 1
        trainer.best_val_acc = checkpoint['best_val_acc']
        print(f"Resumed from epoch {trainer.start_epoch}")

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
