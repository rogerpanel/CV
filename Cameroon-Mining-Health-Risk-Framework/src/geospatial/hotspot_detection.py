"""Getis–Ord G* hotspot detection."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from ._weights import distance_band_weights


def getis_ord_g_star(
    values: Sequence[float],
    coords: Sequence[tuple],
    threshold_km: float = 5.0,
) -> pd.DataFrame:
    """Compute the Getis-Ord G* z-score per location.

    Significant positive z-scores → hotspot; significant negatives → coldspot.
    """
    x = np.asarray(values, dtype=float)
    n = len(x)
    W = distance_band_weights(coords, threshold_km, binary=True)
    np.fill_diagonal(W, 1.0)               # G* includes self in its window

    mean = x.mean()
    s = x.std(ddof=0)
    g_star = np.empty(n)
    z = np.empty(n)
    for i in range(n):
        wi = W[i]
        sum_w = wi.sum()
        sum_w2 = (wi ** 2).sum()
        num = (wi * x).sum() - mean * sum_w
        den = s * np.sqrt((n * sum_w2 - sum_w ** 2) / (n - 1))
        z[i] = num / den if den > 0 else 0.0
        g_star[i] = (wi * x).sum() / x.sum() if x.sum() > 0 else 0.0

    p = 2 * (1 - stats.norm.cdf(np.abs(z)))
    classification = np.where(z > 1.96, "Hot spot",
                       np.where(z < -1.96, "Cold spot", "Not significant"))
    return pd.DataFrame({
        "G_star": g_star,
        "z_score": z,
        "p_value": p,
        "class": classification,
    })
