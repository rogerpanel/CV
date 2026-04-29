"""DHS data ingestion helpers.

The Cameroon DHS 2018 cluster shapefile (``CMGE71FL.zip``) and household
recodes (``CMHR71DT.ZIP``) are not redistributable — register at
https://dhsprogram.com/data/ to access them.  The helpers below assume those
files have been placed in ``data/health/dhs_2018/``.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_dhs_clusters(geo_path: str | Path) -> pd.DataFrame:
    """Read the DHS GE shapefile and keep East-region clusters only."""
    try:
        import geopandas as gpd
    except ImportError as err:                                       # pragma: no cover
        raise ImportError("geopandas is required for DHS processing") from err
    gdf = gpd.read_file(geo_path)
    gdf = gdf.rename(columns={"DHSCLUST": "cluster_id",
                              "LATNUM": "lat", "LONGNUM": "lon",
                              "DHSREGNA": "region", "URBAN_RURA": "residence"})
    east = gdf[gdf["region"].str.contains("Est|East", case=False, na=False)].copy()
    east = east[east["lat"] != 0]                                    # remove unmapped clusters
    return east.drop(columns="geometry", errors="ignore")


def prepare_outcome_table(
    individual_recode: pd.DataFrame,
    cluster_table: pd.DataFrame,
    outcomes: list[str],
) -> pd.DataFrame:
    """Aggregate individual-level outcomes to cluster prevalence."""
    grouped = individual_recode.groupby("cluster_id")[outcomes].mean().reset_index()
    return cluster_table.merge(grouped, on="cluster_id", how="left")
