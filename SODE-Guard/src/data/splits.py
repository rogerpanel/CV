"""Stratified train/val/test split used throughout the manuscript."""
from __future__ import annotations
import numpy as np
from sklearn.model_selection import train_test_split


def stratified_split(X: np.ndarray, y: np.ndarray, *,
                     train: float = 0.70, val: float = 0.15, test: float = 0.15,
                     seed: int = 42):
    if abs(train + val + test - 1.0) > 1e-6:
        raise ValueError("split fractions must sum to 1")
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, train_size=train, stratify=y, random_state=seed,
    )
    rel_val = val / (val + test)
    X_va, X_te, y_va, y_te = train_test_split(
        X_tmp, y_tmp, train_size=rel_val, stratify=y_tmp, random_state=seed,
    )
    return (X_tr, y_tr), (X_va, y_va), (X_te, y_te)
