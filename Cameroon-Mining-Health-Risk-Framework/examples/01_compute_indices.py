"""Minimum working example: compute pollution indices on the bundled template.

Run with the project root on your PYTHONPATH (or after `pip install -e .`)::

    python examples/01_compute_indices.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from cmhr.utils import load_config, load_geochem_table
from cmhr.pollution_indices import (
    geoaccumulation_index, classify_igeo,
    pollution_load_index, contamination_factor,
    ecological_risk_index, classify_ri,
)


HERE = Path(__file__).resolve().parents[1]
SAMPLE = HERE / "data" / "geochemical" / "sample_template.csv"


def main():
    df = load_geochem_table(SAMPLE)
    bg = load_config("reference_doses")["regional_background"]
    print(f"Loaded {len(df)} samples from {SAMPLE.relative_to(HERE)}")

    igeo_pb = geoaccumulation_index(df["Pb_mg_kg"].dropna(), bg["Pb"])
    print("\nIgeo (Pb):", igeo_pb.tolist())
    print("classes  :", classify_igeo(igeo_pb).tolist())

    pli = pollution_load_index(df.dropna(subset=["Pb_mg_kg"]), bg)
    print("\nPLI per sample:\n", pli)

    ri = ecological_risk_index(df.dropna(subset=["Pb_mg_kg"]), bg)
    print("\nRI per sample:\n", pd.DataFrame({
        "RI": ri,
        "class": classify_ri(ri),
    }))


if __name__ == "__main__":
    main()
