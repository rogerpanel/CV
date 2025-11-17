"""
Data loader creation utilities

Implements efficient data loading with proper batching, shuffling,
and parallel loading for training encrypted traffic models.
"""

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from typing import Tuple, Optional
from .dataset import EncryptedTrafficDataset


def create_dataloaders(
    train_dataset: EncryptedTrafficDataset,
    val_dataset: EncryptedTrafficDataset,
    test_dataset: Optional[EncryptedTrafficDataset] = None,
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True,
    use_weighted_sampling: bool = False
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create train, validation, and test dataloaders.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset (optional)
        batch_size: Batch size for training
        num_workers: Number of parallel workers for data loading
        pin_memory: Whether to pin memory (speeds up GPU transfer)
        use_weighted_sampling: Whether to use weighted random sampling for training
                               (useful for imbalanced datasets)

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
        test_loader is None if test_dataset is not provided

    Example:
        >>> train_loader, val_loader, test_loader = create_dataloaders(
        ...     train_dataset, val_dataset, test_dataset,
        ...     batch_size=128, num_workers=4
        ... )
    """
    # Create weighted sampler for training if requested
    sampler = None
    shuffle_train = True

    if use_weighted_sampling:
        class_weights = train_dataset.get_class_weights()
        sample_weights = class_weights[train_dataset.labels.long()]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        shuffle_train = False  # Don't shuffle when using sampler

    # Create training dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop incomplete batches for stable training
    )

    # Create validation dataloader
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffling for validation
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    # Create test dataloader if provided
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )

    print(f"Dataloaders created:")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    if test_loader:
        print(f"  Test batches: {len(test_loader)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Weighted sampling: {use_weighted_sampling}")

    return train_loader, val_loader, test_loader


def create_inference_dataloader(
    dataset: EncryptedTrafficDataset,
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """
    Create dataloader for inference (prediction without labels).

    Args:
        dataset: Dataset for inference
        batch_size: Batch size
        num_workers: Number of parallel workers
        pin_memory: Whether to pin memory

    Returns:
        DataLoader configured for inference
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
