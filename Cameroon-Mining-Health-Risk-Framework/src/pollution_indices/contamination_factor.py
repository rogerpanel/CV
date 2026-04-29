"""Contamination Factor (Håkanson, 1980).

.. math::  CF_i = C_i / C_{ref,i}
"""
from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd

from cmhr.utils import load_config
from ._classify import classify

ArrayLike = Union[float, np.ndarray, pd.Series]


def contamination_factor(measured: ArrayLike, background: float) -> ArrayLike:
    """Compute the contamination factor element-wise."""
    if background is None or background <= 0:
        raise ValueError("background must be positive")
    cf = np.asarray(measured, dtype=float) / background
    if isinstance(measured, pd.Series):
        return pd.Series(cf, index=measured.index, name=f"CF_{measured.name}")
    if np.isscalar(measured):
        return float(cf)
    return cf


def classify_cf(cf_values):
    """Classify CF using the four-tier Håkanson scheme."""
    rules = load_config("thresholds")["contamination_factor"]
    return classify(cf_values, rules)
