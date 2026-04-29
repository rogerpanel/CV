"""Local Indicators of Spatial Association (Anselin, 1995).

.. math::  I_i = z_i \\sum_j w_{ij} z_j

where :math:`z_i` are standardised values.  Permutation testing gives a
significance flag and a four-class label (HH / LL / HL / LH).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from ._weights import distance_band_weights, row_standardise


def _local_i(values, W):
    z = (values - values.mean()) / values.std(ddof=1)
    lag = W @ z
    Ii = z * lag
    return Ii, z, lag


def local_morans_i(
    values: Sequence[float],
    coords: Sequence[tuple],
    threshold_km: float = 5.0,
    permutations: int = 999,
    significance_level: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a DataFrame with local I, p-value, and HH/LL/HL/LH label per point."""
    values = np.asarray(values, dtype=float)
    W = row_standardise(distance_band_weights(coords, threshold_km))
    Ii, z, lag = _local_i(values, W)

    rng = np.random.default_rng(seed)
    n = len(values)
    sims = np.empty((permutations, n))
    for k in range(permutations):
        permuted = rng.permutation(values)
        z_perm = (permuted - permuted.mean()) / permuted.std(ddof=1)
        sims[k] = z_perm * (W @ z_perm)

    p_value = (np.sum(np.abs(sims) >= np.abs(Ii)[None, :], axis=0) + 1) / (permutations + 1)

    labels = []
    for zi, lj, p in zip(z, lag, p_value):
        if p > significance_level:
            labels.append("Not significant")
        elif zi > 0 and lj > 0:
            labels.append("High-High")
        elif zi < 0 and lj < 0:
            labels.append("Low-Low")
        elif zi > 0 and lj < 0:
            labels.append("High-Low")
        else:
            labels.append("Low-High")

    return pd.DataFrame({
        "value": values,
        "z_score": z,
        "spatial_lag": lag,
        "local_I": Ii,
        "p_value": p_value,
        "cluster": labels,
    })
