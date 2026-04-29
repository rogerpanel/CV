"""Data quality assurance helpers used across the framework."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


def handle_below_detection_limit(
    series: pd.Series,
    detection_limit: float,
    method: str = "half_mdl",
) -> pd.Series:
    """Substitute below-MDL values per USEPA guidance.

    Parameters
    ----------
    series : pd.Series
        Concentrations; below-MDL values may be NaN, 0 or coded as ``"<MDL"``.
    detection_limit : float
    method : {"half_mdl", "mdl_over_sqrt2", "zero", "drop"}
        ``"half_mdl"`` is the USEPA default used in the manuscript.
    """
    s = pd.to_numeric(series, errors="coerce")
    mask = s.isna() | (s <= 0)
    if method == "half_mdl":
        s.loc[mask] = detection_limit / 2.0
    elif method == "mdl_over_sqrt2":
        s.loc[mask] = detection_limit / np.sqrt(2.0)
    elif method == "zero":
        s.loc[mask] = 0.0
    elif method == "drop":
        s = s.loc[~mask]
    else:
        raise ValueError(f"Unknown MDL handling method: {method}")
    return s


def shapiro_test_log(values: Iterable[float], alpha: float = 0.05):
    """Shapiro–Wilk test on raw and log-transformed data.

    Returns
    -------
    dict
        ``{"raw": (W, p), "log": (W, p), "lognormal": bool}``.
    """
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size < 3:
        return {"raw": (np.nan, np.nan), "log": (np.nan, np.nan), "lognormal": False}
    raw = stats.shapiro(arr)
    logd = stats.shapiro(np.log(arr))
    return {
        "raw": (float(raw.statistic), float(raw.pvalue)),
        "log": (float(logd.statistic), float(logd.pvalue)),
        "lognormal": logd.pvalue > alpha and raw.pvalue <= alpha,
    }
