"""Reproduce a Moran's I + LISA analysis on the bundled template."""
from __future__ import annotations

from cmhr.utils import load_config, load_geochem_table
from cmhr.pollution_indices import pollution_load_index
from cmhr.geospatial import global_morans_i, local_morans_i


def main():
    df = load_geochem_table("data/geochemical/sample_template.csv")
    bg = load_config("reference_doses")["regional_background"]
    pli = pollution_load_index(df, bg)
    coords = list(zip(df["lat"], df["lon"]))
    moran = global_morans_i(pli.values, coords, threshold_km=50.0)
    lisa = local_morans_i(pli.values, coords, threshold_km=50.0)
    print("Global Moran's I:", moran.to_dict())
    print(lisa)


if __name__ == "__main__":
    main()
