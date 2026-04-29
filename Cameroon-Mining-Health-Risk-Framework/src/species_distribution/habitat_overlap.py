"""Spatial overlay analyses for SDM × mining footprint, SDM × protected areas."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def mining_habitat_overlap(
    suitability: np.ndarray,
    mining_mask: np.ndarray,
    pixel_area_km2: float,
    threshold: float = 0.6,
) -> dict:
    """Quantify habitat–mining overlap (km² and %)."""
    if suitability.shape != mining_mask.shape:
        raise ValueError("suitability and mining_mask must share shape")
    high = suitability >= threshold
    overlap = np.logical_and(high, mining_mask)
    overlap_area = float(overlap.sum() * pixel_area_km2)
    mining_area = float(mining_mask.sum() * pixel_area_km2)
    suitable_area = float(high.sum() * pixel_area_km2)
    return {
        "overlap_km2": overlap_area,
        "mining_area_km2": mining_area,
        "suitable_habitat_km2": suitable_area,
        "fraction_of_mining": overlap_area / mining_area if mining_area else 0.0,
        "fraction_of_habitat": overlap_area / suitable_area if suitable_area else 0.0,
    }


def protected_area_overlap(
    mining_mask: np.ndarray,
    distance_to_pa_km: np.ndarray,
    buffer_distances_km: Sequence[float],
    pixel_area_km2: float,
) -> pd.DataFrame:
    """Mining area falling inside each buffer of a protected area."""
    rows = []
    for d in buffer_distances_km:
        within = (distance_to_pa_km <= d) & mining_mask
        rows.append({
            "buffer_km": d,
            "mining_within_buffer_km2": float(within.sum() * pixel_area_km2),
        })
    return pd.DataFrame(rows)
