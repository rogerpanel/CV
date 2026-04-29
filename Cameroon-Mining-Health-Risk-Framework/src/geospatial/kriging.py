"""Ordinary kriging via PyKrige.

Used to interpolate sparse heavy-metal samples (n=183) onto a regular grid for
spatial visualisation of PI / RI / HI surfaces.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def ordinary_kriging(
    coords: Sequence[tuple],
    values: Sequence[float],
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    variogram_model: str = "spherical",
    nlags: int = 6,
    enable_plotting: bool = False,
):
    """Ordinary kriging interpolation.

    Parameters
    ----------
    variogram_model : {"linear", "power", "gaussian", "spherical", "exponential"}
    """
    try:
        from pykrige.ok import OrdinaryKriging
    except ImportError as err:                                       # pragma: no cover
        raise ImportError("pykrige is required: pip install pykrige") from err

    x = np.asarray([c[0] for c in coords], dtype=float)
    y = np.asarray([c[1] for c in coords], dtype=float)
    v = np.asarray(values, dtype=float)

    OK = OrdinaryKriging(
        x, y, v,
        variogram_model=variogram_model,
        verbose=False,
        enable_plotting=enable_plotting,
        nlags=nlags,
    )
    z, ss = OK.execute("grid", np.unique(grid_x), np.unique(grid_y))
    return z, ss
