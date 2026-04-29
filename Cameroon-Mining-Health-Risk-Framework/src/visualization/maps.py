"""Map figures (sample locations, LULC, LISA clusters)."""
from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_LISA_PALETTE = {
    "High-High": "#d62728",
    "Low-Low":   "#1f77b4",
    "High-Low":  "#ff7f0e",
    "Low-High":  "#9467bd",
    "Not significant": "#cccccc",
}


def plot_sample_map(samples: pd.DataFrame, value_col: str, title: str = "Sample map",
                    cmap: str = "viridis", ax=None):
    """Scatter plot of sample points coloured by ``value_col``.

    Parameters
    ----------
    samples : DataFrame
        Must include ``lat`` and ``lon`` columns.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(samples["lon"], samples["lat"], c=samples[value_col],
                    s=60, edgecolor="k", cmap=cmap)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title(title)
    plt.colorbar(sc, ax=ax, label=value_col)
    ax.set_aspect("equal", adjustable="datalim")
    return ax


def plot_lisa_map(lisa_df: pd.DataFrame, samples: pd.DataFrame, ax=None):
    """Plot LISA cluster classes for each sample point."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))
    df = samples.copy()
    df["cluster"] = lisa_df["cluster"].values
    for cls, sub in df.groupby("cluster"):
        ax.scatter(sub["lon"], sub["lat"],
                   color=_LISA_PALETTE.get(cls, "#999"),
                   label=cls, s=70, edgecolor="k")
    ax.legend(title="LISA cluster", loc="best")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title("Local indicators of spatial association")
    ax.set_aspect("equal", adjustable="datalim")
    return ax


def plot_lulc_change_map(early: np.ndarray, late: np.ndarray,
                         class_labels: Sequence[str],
                         ax=None):
    """Side-by-side LULC maps (uses categorical cmap)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    cmap = plt.cm.get_cmap("tab10", len(class_labels))
    for img, ax_, title in zip([early, late], axes, ["Earliest year", "Latest year"]):
        im = ax_.imshow(img, cmap=cmap, vmin=0, vmax=len(class_labels) - 1)
        ax_.set_title(title)
        ax_.axis("off")
    cbar = fig.colorbar(im, ax=axes, ticks=range(len(class_labels)), shrink=0.8)
    cbar.ax.set_yticklabels(class_labels)
    return fig
