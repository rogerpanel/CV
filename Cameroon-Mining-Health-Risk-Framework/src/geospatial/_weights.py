"""Spatial weights matrices."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def distance_band_weights(coords: Sequence[tuple], threshold_km: float, binary: bool = True) -> np.ndarray:
    """Distance-band spatial weights using Haversine distance."""
    from math import asin, cos, radians, sin, sqrt

    n = len(coords)
    W = np.zeros((n, n))
    R = 6371.0
    for i in range(n):
        lat1, lon1 = radians(coords[i][0]), radians(coords[i][1])
        for j in range(i + 1, n):
            lat2, lon2 = radians(coords[j][0]), radians(coords[j][1])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
            c = 2 * asin(sqrt(a))
            d = R * c
            if d <= threshold_km and d > 0:
                W[i, j] = 1.0 if binary else 1.0 / d
                W[j, i] = W[i, j]
    return W


def row_standardise(W: np.ndarray) -> np.ndarray:
    """Row-standardise a weights matrix (rows sum to 1)."""
    rs = W.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return W / rs
