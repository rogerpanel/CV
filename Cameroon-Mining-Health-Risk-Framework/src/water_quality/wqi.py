"""Water Quality Index (Tiwari & Mishra, 1985; Pesce & Wunderlin, 2000).

Weighted-arithmetic implementation:

.. math::

    Q_i = 100 \\cdot \\frac{C_i - I_i}{S_i - I_i}, \\qquad
    W_i = \\frac{w_i}{\\sum w_i}, \\qquad
    WQI = \\sum W_i Q_i

with :math:`w_i = K / S_i` and :math:`K = 1 / \\sum (1/S_i)`.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from cmhr.utils import load_config


def _ideal_value(parameter: str) -> float:
    """Ideal-water concentration ``Iᵢ`` (USPHS / WHO)."""
    if parameter.lower() == "ph":
        return 7.0
    return 0.0


def water_quality_index(
    df: pd.DataFrame,
    standards: Mapping[str, float] | None = None,
) -> pd.Series:
    """Per-sample WQI; assumes ``df`` columns share names with ``standards``.

    Parameters
    ----------
    df : DataFrame
        Wide-format physico-chemical measurements (mg/L, except pH).
    standards : mapping, optional
        Element / parameter → permissible limit ``Sᵢ``.  Defaults to
        ``config/reference_doses.yaml :: who_water_standards``.
    """
    if standards is None:
        std_cfg = load_config("reference_doses")["who_water_standards"]
        standards = {k: v for k, v in std_cfg.items() if isinstance(v, (int, float))}

    common = [p for p in standards if p in df.columns and standards[p] > 0]
    if not common:
        raise ValueError("No matching parameters between df and standards")

    s = np.array([standards[p] for p in common])
    k = 1.0 / np.sum(1.0 / s)
    weights = (k / s)
    weights /= weights.sum()

    sub_indices = []
    for parameter, std in zip(common, s):
        ideal = _ideal_value(parameter)
        q = 100.0 * (df[parameter].astype(float) - ideal) / (std - ideal)
        sub_indices.append(q.values)
    sub = np.column_stack(sub_indices)
    wqi = np.nansum(sub * weights, axis=1)
    return pd.Series(wqi, index=df.index, name="WQI")
