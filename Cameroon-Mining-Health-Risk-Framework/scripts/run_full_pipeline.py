"""End-to-end command-line runner.

Reproduces the manuscript's principal tables for any geochemical input file
that follows the schema in ``data/geochemical/sample_template.csv``.

Run from the project root::

    python -m scripts.run_full_pipeline \
        --geochem data/geochemical/sample_template.csv \
        --output  outputs/

The script writes:
* `pollution_indices.csv`  — per-sample Igeo, CF, EF, PLI, RI, mCdeg, Nemerow PI
* `health_risk.csv`        — per-element HQ/HI/ILCR (Monte Carlo p5/50/95)
* `spatial_autocorrelation.csv` — Moran's I + LISA classes for the PLI field
* `summary.json`           — overall metrics
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cmhr.utils import load_config, load_geochem_table, save_results
from cmhr.pollution_indices import (
    geoaccumulation_index, classify_igeo,
    contamination_factor, classify_cf,
    pollution_load_index,
    ecological_risk_index, classify_ri,
    enrichment_factor, classify_ef,
    nemerow_pollution_index, classify_nemerow,
    modified_contamination_degree,
)
from cmhr.health_risk import MonteCarloRiskAssessment
from cmhr.geospatial import global_morans_i, local_morans_i


CARCINOGENIC = {"As", "Cd", "Cr", "Pb", "Ni"}


def _backgrounds():
    return load_config("reference_doses")["regional_background"]


def pollution_indices_table(df: pd.DataFrame) -> pd.DataFrame:
    bg = _backgrounds()
    rows = df[["sample_id", "district", "sub_district", "lat", "lon"]].copy()

    for el, bn in bg.items():
        col = f"{el}_mg_kg"
        if col in df.columns:
            rows[f"Igeo_{el}"] = geoaccumulation_index(df[col], bn)
            rows[f"CF_{el}"] = contamination_factor(df[col], bn)

    rows["Igeo_class_max"] = rows.filter(like="Igeo_").max(axis=1).pipe(classify_igeo)
    rows["CF_class_max"] = rows.filter(like="CF_").max(axis=1).pipe(classify_cf)

    if "Al_mg_kg" in df.columns and bg.get("Al"):
        for el in ("As", "Pb", "Cd", "Hg", "Cr", "Cu", "Zn"):
            col = f"{el}_mg_kg"
            if col in df.columns:
                rows[f"EF_{el}"] = enrichment_factor(df[col], df["Al_mg_kg"], bg[el], bg["Al"])
        rows["EF_class_max"] = rows.filter(like="EF_").max(axis=1).pipe(classify_ef)

    rows["PLI"] = pollution_load_index(df, bg)
    rows["mCdeg"] = modified_contamination_degree(df, bg)
    try:
        rows["RI"] = ecological_risk_index(df, bg)
        rows["RI_class"] = classify_ri(rows["RI"])
    except ValueError:
        pass

    if any(c.endswith("_mg_l") for c in df.columns):
        std = load_config("reference_doses")["who_water_standards"]
        std = {k: v for k, v in std.items() if isinstance(v, (int, float))}
        water_df = df.rename(columns=lambda c: c.replace("_mg_l", "")).copy()
        try:
            rows["Nemerow_PI"] = nemerow_pollution_index(water_df, std)
            rows["Nemerow_class"] = classify_nemerow(rows["Nemerow_PI"])
        except ValueError:
            pass
    return rows


def health_risk_table(df: pd.DataFrame, receptors=("adults", "children")) -> pd.DataFrame:
    out = []
    for receptor in receptors:
        mc = MonteCarloRiskAssessment(receptor=receptor)
        for el in ("As", "Cd", "Pb", "Hg", "Cr", "Mn", "Fe", "Zn"):
            col_w = f"{el}_mg_l"
            if col_w not in df.columns or df[col_w].dropna().empty:
                continue
            c = float(df[col_w].mean())
            try:
                hq = mc.run_ingestion(c, el, "water", carcinogenic=False)
                row = hq.summary()
                row["metric"] = "HQ_ingestion_water"
                out.append(row)
            except KeyError:
                pass
            if el in CARCINOGENIC:
                try:
                    cr = mc.run_ingestion(c, el, "water", carcinogenic=True)
                    row = cr.summary()
                    row["metric"] = "CR_ingestion_water"
                    out.append(row)
                except KeyError:
                    pass
    return pd.DataFrame(out)


def spatial_table(df: pd.DataFrame, indices: pd.DataFrame) -> pd.DataFrame:
    if {"lat", "lon"} - set(df.columns):
        return pd.DataFrame()
    coords = list(zip(df["lat"], df["lon"]))
    valid = indices["PLI"].notna()
    if valid.sum() < 5:
        return pd.DataFrame()
    coords = [c for c, ok in zip(coords, valid) if ok]
    values = indices["PLI"].dropna().values
    moran = global_morans_i(values, coords)
    lisa = local_morans_i(values, coords)
    lisa["sample_id"] = df.loc[valid, "sample_id"].values
    return lisa, moran


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--geochem", required=True, help="CSV input")
    p.add_argument("--output", default="outputs", help="Output directory")
    p.add_argument("--receptors", nargs="*", default=["adults", "children"])
    args = p.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    df = load_geochem_table(args.geochem)

    indices = pollution_indices_table(df)
    save_results(indices, out / "pollution_indices.csv")

    risk = health_risk_table(df, receptors=args.receptors)
    save_results(risk, out / "health_risk.csv")

    summary = {
        "n_samples": int(len(df)),
        "districts": df["district"].dropna().unique().tolist(),
        "PLI_max": float(np.nanmax(indices.get("PLI", [np.nan]))),
        "PLI_mean": float(np.nanmean(indices.get("PLI", [np.nan]))),
    }
    try:
        lisa_df, moran = spatial_table(df, indices)
        save_results(lisa_df, out / "spatial_autocorrelation.csv")
        summary["morans_I"] = moran.statistic
        summary["morans_p"] = moran.p_value
    except Exception as err:                                         # pragma: no cover
        print(f"[warn] spatial analysis skipped: {err}")

    save_results(summary, out / "summary.json")
    print(f"Done → {out.resolve()}")


if __name__ == "__main__":
    main()
