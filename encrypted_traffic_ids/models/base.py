"""
Base model class for encrypted traffic detection models

Provides common functionality for all model architectures including:
- Parameter counting
- Device management
- Save/load functionality
- Training mode management
"""

import torch
import torch.nn as nn
from typing import Dict, Any
from pathlib import Path


class BaseModel(nn.Module):
    """
    Base class for all intrusion detection models.

    Provides common functionality shared across all architectures.
    """

    def __init__(self):
        """Initialize base model."""
        super(BaseModel, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (to be implemented by subclasses).

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        raise NotImplementedError("Subclasses must implement forward()")

    def count_parameters(self) -> int:
        """
        Count total number of trainable parameters.

        Returns:
            Total number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_num_parameters(self, trainable_only: bool = True) -> Dict[str, int]:
        """
        Get detailed parameter count.

        Args:
            trainable_only: If True, count only trainable parameters

        Returns:
            Dictionary with parameter counts
        """
        if trainable_only:
            total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            total = sum(p.numel() for p in self.parameters())

        # Count by layer type
        counts_by_type = {}
        for name, module in self.named_modules():
            module_type = type(module).__name__
            if module_type not in counts_by_type:
                counts_by_type[module_type] = 0

            for param in module.parameters(recurse=False):
                if not trainable_only or param.requires_grad:
                    counts_by_type[module_type] += param.numel()

        return {
            'total': total,
            'by_type': counts_by_type
        }

    def save_checkpoint(self, path: str, epoch: int, optimizer_state: Dict = None,
                       metrics: Dict = None) -> None:
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
            epoch: Current epoch number
            optimizer_state: Optimizer state dict (optional)
            metrics: Performance metrics (optional)
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'model_config': self.get_config() if hasattr(self, 'get_config') else {},
        }

        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state

        if metrics is not None:
            checkpoint['metrics'] = metrics

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to: {path}")

    def load_checkpoint(self, path: str, device: torch.device = None) -> Dict[str, Any]:
        """
        Load model checkpoint.

        Args:
            path: Path to checkpoint file
            device: Device to load model to

        Returns:
            Dictionary containing checkpoint metadata
        """
        if device is None:
            device = next(self.parameters()).device

        checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])

        print(f"Checkpoint loaded from: {path}")
        if 'epoch' in checkpoint:
            print(f"  Epoch: {checkpoint['epoch']}")
        if 'metrics' in checkpoint:
            print(f"  Metrics: {checkpoint['metrics']}")

        return checkpoint

    def freeze_layers(self, layer_names: list = None) -> None:
        """
        Freeze specified layers (stop gradient computation).

        Args:
            layer_names: List of layer names to freeze.
                        If None, freezes all layers.
        """
        if layer_names is None:
            # Freeze all parameters
            for param in self.parameters():
                param.requires_grad = False
            print("All layers frozen")
        else:
            # Freeze specific layers
            for name, param in self.named_parameters():
                if any(layer_name in name for layer_name in layer_names):
                    param.requires_grad = False
                    print(f"Frozen: {name}")

    def unfreeze_layers(self, layer_names: list = None) -> None:
        """
        Unfreeze specified layers (enable gradient computation).

        Args:
            layer_names: List of layer names to unfreeze.
                        If None, unfreezes all layers.
        """
        if layer_names is None:
            # Unfreeze all parameters
            for param in self.parameters():
                param.requires_grad = True
            print("All layers unfrozen")
        else:
            # Unfreeze specific layers
            for name, param in self.named_parameters():
                if any(layer_name in name for layer_name in layer_names):
                    param.requires_grad = True
                    print(f"Unfrozen: {name}")

    def print_summary(self) -> None:
        """Print model summary including architecture and parameter count."""
        print("=" * 80)
        print("MODEL SUMMARY")
        print("=" * 80)
        print(f"Model: {self.__class__.__name__}")
        print(f"\nArchitecture:\n{self}")

        param_info = self.get_num_parameters()
        print(f"\nTotal trainable parameters: {param_info['total']:,}")

        print("\nParameters by layer type:")
        for layer_type, count in sorted(param_info['by_type'].items(),
                                       key=lambda x: x[1], reverse=True):
            if count > 0:
                percentage = count / param_info['total'] * 100
                print(f"  {layer_type}: {count:,} ({percentage:.1f}%)")

        print("=" * 80)
