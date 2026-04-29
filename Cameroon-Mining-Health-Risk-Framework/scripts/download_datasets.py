"""Open-data download helpers.

Examples
--------
::

    python -m scripts.download_datasets --gbif --bbox 3.5 6.5 12 16
    python -m scripts.download_datasets --worldclim 2.1 --variables bio_1 bio_12
    python -m scripts.download_datasets --srtm --aoi config/study_area.yaml

Authenticated sources (DHS, USGS Earth Explorer for some products) require
credentials and cannot be fully automated; the script prints the required URL
and registration steps.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"


def gbif_download(species_list, bbox, out_path: Path):
    """Issue a GBIF occurrence download via pygbif."""
    try:
        from pygbif import occurrences as occ
    except ImportError:                                              # pragma: no cover
        print("Install pygbif: pip install pygbif", file=sys.stderr)
        return
    south, north, west, east = bbox
    rows = []
    for species in species_list:
        results = occ.search(
            scientificName=species,
            decimalLatitude=f"{south},{north}",
            decimalLongitude=f"{west},{east}",
            hasCoordinate=True, hasGeospatialIssue=False,
            limit=10000,
        )
        rows.extend(results.get("results", []))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved {len(rows)} GBIF records → {out_path}")


def dhs_instructions():
    print("DHS Cameroon 2018 datasets are restricted-access. Steps:")
    print("  1. Register at https://dhsprogram.com/data/")
    print("  2. Request access to the Cameroon 2018 datasets")
    print("  3. Download CMGE71FL.zip, CMHR71DT.ZIP, CMKR71DT.ZIP, CMIR71DT.ZIP")
    print("  4. Place them in data/health/dhs_2018/")


def worldclim_download(version: str, variables, out_dir: Path, resolution: str = "30s"):
    """WorldClim 2.1 GeoTIFFs."""
    import requests
    base = f"https://biogeo.ucdavis.edu/data/worldclim/v{version}/base/wc{version}_{resolution}_"
    out_dir.mkdir(parents=True, exist_ok=True)
    for v in variables:
        url = f"{base}{v}.zip"
        target = out_dir / f"{v}.zip"
        if target.exists():
            print(f"skip (exists): {target}")
            continue
        print(f"GET {url}")
        r = requests.get(url, stream=True, timeout=300)
        r.raise_for_status()
        with target.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    print(f"WorldClim → {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gbif", action="store_true")
    p.add_argument("--bbox", nargs=4, type=float, default=[3.5, 6.5, 12.0, 16.0],
                   help="south north west east")
    p.add_argument("--worldclim", type=str, help="WorldClim version, e.g. 2.1")
    p.add_argument("--variables", nargs="*", default=["bio"])
    p.add_argument("--dhs", action="store_true", help="Print DHS access instructions")
    args = p.parse_args()

    if args.gbif:
        species = [
            "Loxodonta cyclotis",
            "Gorilla gorilla gorilla",
            "Pan troglodytes troglodytes",
            "Panthera pardus",
            "Osteolaemus tetraspis",
        ]
        gbif_download(species, args.bbox, DATA_ROOT / "biodiversity" / "gbif_cameroon_east.csv")
    if args.worldclim:
        worldclim_download(args.worldclim, args.variables,
                           DATA_ROOT / "remote_sensing" / "worldclim")
    if args.dhs:
        dhs_instructions()


if __name__ == "__main__":
    main()
