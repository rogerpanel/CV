"""Enrichment Factor (Sutherland, 2000; Zoller et al., 1974).

.. math::

    EF = \\frac{(C_{element}/C_{normaliser})_{sample}}
              {(C_{element}/C_{normaliser})_{background}}

Common normalisers are conservative lithogenic elements: Al, Fe or Sc.
"""
from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd

from cmhr.utils import load_config
from ._classify import classify

ArrayLike = Union[float, np.ndarray, pd.Series]


def enrichment_factor(
    sample_element: ArrayLike,
    sample_normaliser: ArrayLike,
    background_element: float,
    background_normaliser: float,
) -> ArrayLike:
    """Compute EF element-wise."""
    if background_element <= 0 or background_normaliser <= 0:
        raise ValueError("background concentrations must be positive")
    sample_ratio = np.divide(
        np.asarray(sample_element, dtype=float),
        np.asarray(sample_normaliser, dtype=float),
    )
    background_ratio = background_element / background_normaliser
    ef = sample_ratio / background_ratio
    if isinstance(sample_element, pd.Series):
        return pd.Series(ef, index=sample_element.index, name=f"EF_{sample_element.name}")
    if np.isscalar(sample_element):
        return float(ef)
    return ef


def classify_ef(ef_values):
    """Sutherland 7-class scheme."""
    rules = load_config("thresholds")["enrichment_factor"]
    return classify(ef_values, rules)
