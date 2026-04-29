"""Google Earth Engine batch export script.

Exports annual median composites and a Random-Forest LULC classification for
each year in ``epochs`` to Google Drive.

Run from the project root with a working `earthengine authenticate` session::

    python -m scripts.gee_export --epochs 2000 2010 2020 2024
"""
from __future__ import annotations

import argparse

from cmhr.utils import load_config


def export_one(year: int, aoi_geojson: dict, drive_folder: str = "cmhr_exports"):
    import ee
    from cmhr.remote_sensing import (
        build_landsat_collection, build_sentinel2_collection,
        annual_composite, all_indices,
    )

    aoi = ee.Geometry(aoi_geojson)
    if year >= 2015:
        coll = build_sentinel2_collection(aoi, f"{year}-01-01", f"{year}-12-31")
    else:
        coll = build_landsat_collection(aoi, f"{year}-01-01", f"{year}-12-31")
    composite = annual_composite(coll, year)

    # Add manuscript spectral indices
    bands = {
        "blue": composite.select("B2") if year >= 2015 else composite.select("blue"),
        "green": composite.select("B3") if year >= 2015 else composite.select("green"),
        "red": composite.select("B4") if year >= 2015 else composite.select("red"),
        "nir": composite.select("B8") if year >= 2015 else composite.select("nir"),
        "swir1": composite.select("B11") if year >= 2015 else composite.select("swir1"),
    }
    indices = all_indices(bands)
    for k, img in indices.items():
        composite = composite.addBands(img.rename(k))

    task = ee.batch.Export.image.toDrive(
        image=composite,
        description=f"cmhr_composite_{year}",
        folder=drive_folder,
        fileNamePrefix=f"cmhr_composite_{year}",
        region=aoi,
        scale=10 if year >= 2015 else 30,
        maxPixels=1e13,
    )
    task.start()
    print(f"started export for {year}: task id {task.id}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", nargs="*", type=int, default=[2000, 2010, 2020, 2024])
    p.add_argument("--folder", default="cmhr_exports")
    args = p.parse_args()

    bbox = load_config("study_area")["bounding_box"]
    aoi = {
        "type": "Polygon",
        "coordinates": [[
            [bbox["min_lon"], bbox["min_lat"]],
            [bbox["max_lon"], bbox["min_lat"]],
            [bbox["max_lon"], bbox["max_lat"]],
            [bbox["min_lon"], bbox["max_lat"]],
            [bbox["min_lon"], bbox["min_lat"]],
        ]],
    }
    for year in args.epochs:
        export_one(year, aoi, args.folder)


if __name__ == "__main__":
    main()
