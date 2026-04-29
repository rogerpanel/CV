"""LULC change detection utilities."""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd


def lulc_change_matrix(
    earlier: np.ndarray,
    later: np.ndarray,
    class_labels: Sequence[str] | None = None,
) -> pd.DataFrame:
    """From-to transition matrix between two classified rasters."""
    if earlier.shape != later.shape:
        raise ValueError("earlier and later rasters must share the same shape")
    n = int(max(earlier.max(), later.max()) + 1)
    matrix = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        mask_i = earlier == i
        for j in range(n):
            matrix[i, j] = int(np.logical_and(mask_i, later == j).sum())
    if class_labels is None:
        class_labels = [str(i) for i in range(n)]
    return pd.DataFrame(matrix, index=class_labels, columns=class_labels)


def transition_areas(
    transition: pd.DataFrame,
    pixel_area_km2: float,
) -> pd.DataFrame:
    """Multiply a pixel-count transition matrix by the per-pixel area."""
    return transition * pixel_area_km2


def annual_class_areas(
    classified_stack: Mapping[int, np.ndarray],
    pixel_area_km2: float,
    class_labels: Sequence[str],
) -> pd.DataFrame:
    """Per-year, per-class area summary (km²)."""
    rows = []
    n_classes = len(class_labels)
    for year, raster in classified_stack.items():
        counts = np.bincount(raster.ravel(), minlength=n_classes)
        rows.append({
            "year": year,
            **{class_labels[i]: counts[i] * pixel_area_km2 for i in range(n_classes)},
        })
    return pd.DataFrame(rows).set_index("year")
