"""
Reproducibility utilities for ensuring deterministic results

Ensures reproducible experiments across different runs,
as required for Q1 journal publications.
"""

import random
import numpy as np
import torch
import os
import sys
import platform


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value (default: 42)
        deterministic: If True, ensures deterministic CUDA behavior
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)
        if hasattr(torch, 'use_deterministic_algorithms'):
            try:
                torch.use_deterministic_algorithms(True)
            except RuntimeError:
                pass
    else:
        torch.backends.cudnn.benchmark = True

    print(f"Random seed set to {seed} (deterministic={deterministic})")


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Get the appropriate device for PyTorch operations."""
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def print_system_info() -> None:
    """Print system and library version information for reproducibility."""
    print("=" * 80)
    print("SYSTEM INFORMATION")
    print("=" * 80)
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"NumPy: {np.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN: {torch.backends.cudnn.version()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    print("=" * 80)
