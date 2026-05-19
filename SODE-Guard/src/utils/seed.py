"""Deterministic seeding helpers.

The five reproducibility seeds reported in the manuscript are
{42, 137, 271, 1729, 2026}. Each seed is passed through here so that the
Python ``random``, NumPy, PyTorch CPU + CUDA streams, and ``torchsde``
Brownian samplers all share the same root.
"""
from __future__ import annotations
import os
import random
import numpy as np
import torch


REPRODUCIBILITY_SEEDS: tuple[int, ...] = (42, 137, 271, 1729, 2026)


def set_global_seed(seed: int, *, deterministic_cudnn: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """``DataLoader`` ``worker_init_fn`` for reproducible shuffling."""
    s = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(s)
    random.seed(s)
