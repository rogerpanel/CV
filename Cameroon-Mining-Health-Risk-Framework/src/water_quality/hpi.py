"""Heavy-metal Pollution Index (Mohan et al., 1996).

.. math::

    HPI = \\frac{\\sum W_i Q_i}{\\sum W_i}, \\qquad
    Q_i = \\frac{|M_i - I_i|}{S_i - I_i} \\times 100, \\qquad
    W_i = K / S_i

The critical pollution value is 100.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from cmhr.utils import load_config


def heavy_metal_pollution_index(
    df: pd.DataFrame,
    standards: Mapping[str, float] | None = None,
    ideal_values: Mapping[str, float] | None = None,
) -> pd.Series:
    """Compute HPI per sample (row).

    ``standards`` defaults to WHO drinking-water guidelines; ``ideal_values``
    defaults to zero for all metals (as recommended for trace metals).
    """
    if standards is None:
        cfg = load_config("reference_doses")["who_water_standards"]
        standards = {k: v for k, v in cfg.items()
                     if isinstance(v, (int, float)) and k not in {"pH", "TDS", "turbidity_NTU", "hardness"}}

    if ideal_values is None:
        ideal_values = {k: 0.0 for k in standards}

    common = [m for m in standards if m in df.columns and standards[m] > 0]
    if not common:
        raise ValueError("No matching metals between df and standards")

    s = np.array([standards[m] for m in common])
    weights = 1.0 / s

    qi_arr = []
    for metal, std in zip(common, s):
        ideal = ideal_values.get(metal, 0.0)
        qi = np.abs(df[metal].astype(float) - ideal) / (std - ideal) * 100.0
        qi_arr.append(qi.values)
    qi_mat = np.column_stack(qi_arr)
    num = np.nansum(qi_mat * weights, axis=1)
    den = np.nansum(np.broadcast_to(weights, qi_mat.shape), axis=1)
    hpi = num / np.where(den > 0, den, np.nan)
    return pd.Series(hpi, index=df.index, name="HPI")
