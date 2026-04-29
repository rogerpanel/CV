"""Geoaccumulation Index (Müller, 1969).

.. math::

    I_{geo} = \\log_2\\!\\left(\\frac{C_n}{1.5 \\cdot B_n}\\right)

The 1.5 factor compensates for natural lithogenic variability of background
concentrations.  A seven-class scheme (0–6) is applied via
``classify_igeo``.
"""
from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd

from cmhr.utils import load_config
from ._classify import classify

ArrayLike = Union[float, np.ndarray, pd.Series]


def geoaccumulation_index(
    measured: ArrayLike,
    background: float,
    lithogenic_factor: float = 1.5,
) -> ArrayLike:
    """Compute :math:`I_{geo}` element-wise.

    Parameters
    ----------
    measured : array-like
        Sample concentrations ``Cₙ`` (mg/kg or mg/L; same unit as ``background``).
    background : float
        Regional geochemical background ``Bₙ`` (e.g. Mimba et al. 2018 for East
        Cameroon — see ``config/reference_doses.yaml :: regional_background``).
    lithogenic_factor : float, default 1.5
        Müller correction; rarely changed.

    Returns
    -------
    Same type as ``measured`` (scalar / np.ndarray / pd.Series).
    """
    if background is None or background <= 0:
        raise ValueError("background must be positive")
    ratio = np.divide(np.asarray(measured, dtype=float), lithogenic_factor * background)
    with np.errstate(divide="ignore", invalid="ignore"):
        igeo = np.log2(ratio)
    if isinstance(measured, pd.Series):
        return pd.Series(igeo, index=measured.index, name=f"Igeo_{measured.name}")
    if np.isscalar(measured):
        return float(igeo)
    return igeo


def classify_igeo(igeo_values):
    """Apply Müller's seven-class scheme."""
    rules = load_config("thresholds")["geoaccumulation_index"]
    return classify(igeo_values, rules)
