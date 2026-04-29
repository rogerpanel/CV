"""Mining-proximity classification for DHS clusters."""
from __future__ import annotations

from math import asin, cos, radians, sin, sqrt
from typing import Sequence

import numpy as np
import pandas as pd

from cmhr.utils import load_config


def _haversine_km(lat1, lon1, lat2, lon2):
    lat1, lat2, lon1, lon2 = map(radians, (lat1, lat2, lon1, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 6371.0 * 2 * asin(sqrt(a))


def distance_to_mining_km(
    cluster_coords: Sequence[tuple],
    mining_coords: Sequence[tuple],
) -> np.ndarray:
    """Minimum great-circle distance from each cluster to the nearest mining pixel/centroid."""
    out = np.empty(len(cluster_coords))
    for i, (lat, lon) in enumerate(cluster_coords):
        out[i] = min(_haversine_km(lat, lon, mlat, mlon)
                     for mlat, mlon in mining_coords)
    return out


def classify_proximity(distances_km, classes_cfg=None) -> pd.Series:
    """Apply ``study_area.yaml :: mining_proximity_classes_km``."""
    if classes_cfg is None:
        classes_cfg = load_config("study_area")["mining_proximity_classes_km"]
    arr = np.asarray(distances_km, dtype=float)
    out = np.empty(arr.shape, dtype=object)
    out[:] = "unclassified"
    for label, (lo, hi) in classes_cfg.items():
        hi = float(hi) if hi != float("inf") else np.inf
        mask = (arr >= lo) & (arr < hi)
        out[mask] = label
    return pd.Series(out, name="proximity_class")
