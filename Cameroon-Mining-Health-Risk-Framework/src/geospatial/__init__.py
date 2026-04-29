"""Spatial analysis: autocorrelation, hotspots, and interpolation.

* ``morans_i`` — Global Moran's I
* ``lisa`` — Local Indicators of Spatial Association
* ``kriging`` — ordinary kriging with variogram fitting (PyKrige)
* ``idw`` — inverse-distance weighting
* ``hotspot_detection`` — Getis-Ord G* hotspots
"""
from .morans_i import global_morans_i
from .lisa import local_morans_i
from .kriging import ordinary_kriging
from .idw import inverse_distance_weighting
from .hotspot_detection import getis_ord_g_star

__all__ = [
    "global_morans_i",
    "local_morans_i",
    "ordinary_kriging",
    "inverse_distance_weighting",
    "getis_ord_g_star",
]
