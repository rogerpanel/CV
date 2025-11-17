"""
Reproducibility utilities for ensuring deterministic results

This module implements functions to ensure reproducible experiments
across different runs, as required for Q1 journal publications.
"""

import random
import numpy as np
import torch
import os


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed (int): Random seed value (default: 42)
        deterministic (bool): If True, ensures deterministic behavior in CUDA operations
                             Note: May reduce performance

    References:
        This function follows best practices outlined in:
        - PyTorch reproducibility guide
        - "On the importance of reproducible research in machine learning"
    """
    # Python's built-in random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    if deterministic:
        # Ensure deterministic behavior in CUDA operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Set environment variable for additional determinism
        os.environ['PYTHONHASHSEED'] = str(seed)

        # For PyTorch >= 1.8, use deterministic algorithms
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(True)
    else:
        # Allow non-deterministic algorithms for better performance
        torch.backends.cudnn.benchmark = True

    print(f"Random seed set to {seed} (deterministic={deterministic})")


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Get the appropriate device for PyTorch operations.

    Args:
        prefer_cuda (bool): If True, use CUDA if available

    Returns:
        torch.device: Device to use for tensor operations
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        print("Using CPU device")

    return device


def print_system_info() -> None:
    """Print system and library version information for reproducibility."""
    import sys
    import platform

    print("=" * 80)
    print("SYSTEM INFORMATION")
    print("=" * 80)
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    print("=" * 80)
