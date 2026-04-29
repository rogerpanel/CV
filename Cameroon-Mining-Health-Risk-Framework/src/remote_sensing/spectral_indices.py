"""Sentinel-2 / Landsat spectral indices used in the manuscript.

All inputs are surface-reflectance arrays (or float-valued ``ee.Image``
bands) on the same scale.
"""
from __future__ import annotations

import numpy as np

ArrayLike = float | np.ndarray


def ndvi(nir: ArrayLike, red: ArrayLike) -> ArrayLike:
    """Normalised Difference Vegetation Index.

    .. math::  NDVI = (\\rho_{NIR} - \\rho_{Red}) / (\\rho_{NIR} + \\rho_{Red})
    """
    return (nir - red) / (nir + red + 1e-10)


def evi(nir: ArrayLike, red: ArrayLike, blue: ArrayLike,
        G: float = 2.5, C1: float = 6.0, C2: float = 7.5, L: float = 1.0) -> ArrayLike:
    """Enhanced Vegetation Index (Huete et al., 2002)."""
    return G * (nir - red) / (nir + C1 * red - C2 * blue + L)


def savi(nir: ArrayLike, red: ArrayLike, L: float = 0.5) -> ArrayLike:
    """Soil-Adjusted Vegetation Index (Huete, 1988)."""
    return ((nir - red) / (nir + red + L)) * (1 + L)


def bsi(swir1: ArrayLike, red: ArrayLike, nir: ArrayLike, blue: ArrayLike) -> ArrayLike:
    """Bare Soil Index (Rikimaru et al., 2002)."""
    return ((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue) + 1e-10)


def ndwi(green: ArrayLike, nir: ArrayLike) -> ArrayLike:
    """Normalised Difference Water Index (McFeeters, 1996)."""
    return (green - nir) / (green + nir + 1e-10)


def mndwi(green: ArrayLike, swir1: ArrayLike) -> ArrayLike:
    """Modified NDWI (Xu, 2006)."""
    return (green - swir1) / (green + swir1 + 1e-10)


def ndmi(nir: ArrayLike, swir1: ArrayLike) -> ArrayLike:
    """Normalised Difference Moisture Index."""
    return (nir - swir1) / (nir + swir1 + 1e-10)


def ndbi(swir1: ArrayLike, nir: ArrayLike) -> ArrayLike:
    """Normalised Difference Built-up Index (Zha et al., 2003)."""
    return (swir1 - nir) / (swir1 + nir + 1e-10)


def all_indices(bands: dict) -> dict:
    """Compute all manuscript indices from a band dictionary.

    ``bands`` should expose ``blue, green, red, nir, swir1`` arrays/images.
    """
    return {
        "NDVI": ndvi(bands["nir"], bands["red"]),
        "EVI":  evi(bands["nir"], bands["red"], bands["blue"]),
        "SAVI": savi(bands["nir"], bands["red"]),
        "BSI":  bsi(bands["swir1"], bands["red"], bands["nir"], bands["blue"]),
        "NDWI": ndwi(bands["green"], bands["nir"]),
        "MNDWI": mndwi(bands["green"], bands["swir1"]),
        "NDMI": ndmi(bands["nir"], bands["swir1"]),
    }
