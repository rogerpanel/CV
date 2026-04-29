"""Inverse-distance weighting interpolation."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def inverse_distance_weighting(
    coords: Sequence[tuple],
    values: Sequence[float],
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    power: float = 2.0,
    smoothing: float = 1e-9,
) -> np.ndarray:
    """Interpolate ``values`` at ``coords`` onto the grid (grid_x, grid_y).

    Parameters
    ----------
    coords : Sequence[(lon, lat)]
        Control-point coordinates (any planar CRS — distances are Euclidean).
    values : Sequence[float]
        Observations at those control points.
    grid_x, grid_y : ndarray
        Output grid coordinates with the same shape.
    power : float, default 2
        IDW exponent.
    smoothing : float
        Avoids division by zero where a grid cell coincides with a sample.
    """
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float)
    out = np.empty_like(grid_x, dtype=float)

    flat_x = grid_x.ravel()
    flat_y = grid_y.ravel()
    flat_out = np.empty_like(flat_x)

    for k, (x, y) in enumerate(zip(flat_x, flat_y)):
        d = np.sqrt((coords[:, 0] - x) ** 2 + (coords[:, 1] - y) ** 2) + smoothing
        w = 1.0 / d ** power
        flat_out[k] = float(np.sum(w * values) / np.sum(w))

    return flat_out.reshape(grid_x.shape)
