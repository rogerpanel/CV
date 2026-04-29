"""Google Earth Engine helpers replicating the manuscript's preprocessing.

Run ``earthengine authenticate`` once before importing this module.
"""
from __future__ import annotations

from typing import Sequence


def _ensure_ee():
    try:
        import ee
    except ImportError as err:                                       # pragma: no cover
        raise ImportError(
            "earthengine-api is required: pip install earthengine-api"
        ) from err
    if not ee.data._initialized:                                     # type: ignore[attr-defined]
        ee.Initialize()
    return ee


def cloud_mask_sentinel2(image):
    """Mask clouds and shadows using the Scene Classification Layer.

    SCL classes kept: 4 (vegetation), 5 (bare soil), 6 (water), 7 (unclassified).
    """
    ee = _ensure_ee()
    scl = image.select("SCL")
    mask = (scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7))
            .And(image.select("MSK_CLDPRB").lt(20)))
    return image.updateMask(mask).divide(10000).copyProperties(image, image.propertyNames())


def build_sentinel2_collection(
    aoi,
    start: str,
    end: str,
    cloud_pct_max: int = 30,
):
    """Return a cloud-masked Sentinel-2 SR collection (COPERNICUS/S2_SR_HARMONIZED).

    Parameters
    ----------
    aoi : ee.Geometry
    start, end : ISO strings
    cloud_pct_max : int, default 30
        Matches the manuscript's <30% scene-level cloud cover criterion.
    """
    ee = _ensure_ee()
    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct_max))
        .map(cloud_mask_sentinel2)
    )


def build_landsat_collection(
    aoi,
    start: str,
    end: str,
    sensors: Sequence[str] = ("LC08", "LC09", "LT05"),
):
    """Combined Landsat 5/8/9 Surface Reflectance Collection 2 Level-2."""
    ee = _ensure_ee()
    band_map = {
        "LT05": {"blue": "SR_B1", "green": "SR_B2", "red": "SR_B3", "nir": "SR_B4",
                 "swir1": "SR_B5", "swir2": "SR_B7"},
        "LC08": {"blue": "SR_B2", "green": "SR_B3", "red": "SR_B4", "nir": "SR_B5",
                 "swir1": "SR_B6", "swir2": "SR_B7"},
        "LC09": {"blue": "SR_B2", "green": "SR_B3", "red": "SR_B4", "nir": "SR_B5",
                 "swir1": "SR_B6", "swir2": "SR_B7"},
    }
    common_bands = ["blue", "green", "red", "nir", "swir1", "swir2"]
    collections = []
    for sensor in sensors:
        cid = f"LANDSAT/{sensor}/C02/T1_L2"
        coll = (ee.ImageCollection(cid)
                .filterBounds(aoi)
                .filterDate(start, end)
                .filter(ee.Filter.lt("CLOUD_COVER", 30))
                .map(lambda img: img
                     .select([band_map[sensor][b] for b in common_bands], common_bands)
                     .multiply(2.75e-5).add(-0.2)
                     .copyProperties(img, img.propertyNames())))
        collections.append(coll)
    merged = collections[0]
    for c in collections[1:]:
        merged = merged.merge(c)
    return merged


def annual_composite(collection, year: int, reducer: str = "median"):
    """Build a single-year composite (default median)."""
    ee = _ensure_ee()
    start = f"{year}-12-01"            # dry-season window per manuscript
    end = f"{year + 1}-02-28"
    yr = collection.filterDate(start, end)
    if reducer == "median":
        return yr.median()
    if reducer == "mean":
        return yr.mean()
    raise ValueError(f"Unsupported reducer {reducer!r}")
