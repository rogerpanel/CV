"""Contamination Degree, ``Cd`` (Backman et al., 1998; Edet & Offiong, 2002).

.. math::  Cd = \\sum_{i=1}^{n} \\Bigl(\\frac{C_i}{C_{N,i}} - 1\\Bigr)

where :math:`C_{N,i}` is the upper permissible limit for parameter ``i``.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from cmhr.utils import load_config


def contamination_degree(
    df: pd.DataFrame,
    standards: Mapping[str, float] | None = None,
) -> pd.Series:
    """Edet & Offiong contamination degree per sample.

    Classification (Backman et al. 1998):
    - Cd < 1   : Low
    - 1 ≤ Cd < 3 : Medium
    - Cd ≥ 3   : High
    """
    if standards is None:
        cfg = load_config("reference_doses")["who_water_standards"]
        standards = {k: v for k, v in cfg.items() if isinstance(v, (int, float))}
    common = [m for m in standards if m in df.columns and standards[m] > 0]
    if not common:
        raise ValueError("No matching parameters between df and standards")
    s = np.array([standards[m] for m in common])
    matrix = df[common].astype(float).values
    cd = np.nansum(matrix / s - 1.0, axis=1)
    return pd.Series(cd, index=df.index, name="Cd")
