"""Global Moran's I.

.. math::

    I = \\frac{N}{S_0} \\cdot
        \\frac{\\sum_i \\sum_j w_{ij} (x_i-\\bar{x})(x_j-\\bar{x})}
              {\\sum_i (x_i-\\bar{x})^2},
    \\qquad S_0 = \\sum_i \\sum_j w_{ij}

A Monte Carlo permutation test (default 999 permutations) gives the p-value
under the null of spatial randomness.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ._weights import distance_band_weights, row_standardise


@dataclass
class MoransIResult:
    statistic: float
    expected: float
    z_score: float
    p_value: float
    permutations: int

    def to_dict(self) -> dict:
        return self.__dict__


def _morans_i(values: np.ndarray, W: np.ndarray) -> float:
    n = len(values)
    z = values - values.mean()
    s0 = W.sum()
    num = float((W * np.outer(z, z)).sum())
    den = float((z ** 2).sum())
    if den == 0 or s0 == 0:
        return 0.0
    return (n / s0) * (num / den)


def global_morans_i(
    values: Sequence[float],
    coords: Sequence[tuple],
    threshold_km: float = 5.0,
    permutations: int = 999,
    seed: int = 42,
) -> MoransIResult:
    """Compute Global Moran's I and a permutation p-value."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    W = row_standardise(distance_band_weights(coords, threshold_km))
    observed = _morans_i(values, W)

    rng = np.random.default_rng(seed)
    expected = -1.0 / (n - 1)
    sims = np.empty(permutations)
    for k in range(permutations):
        sims[k] = _morans_i(rng.permutation(values), W)

    z = (observed - sims.mean()) / sims.std(ddof=1)
    p = (np.sum(np.abs(sims) >= abs(observed)) + 1) / (permutations + 1)
    return MoransIResult(observed, expected, float(z), float(p), permutations)
