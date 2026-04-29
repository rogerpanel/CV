"""Core analytical plots (distributions, density, tornado, confusion matrix)."""
from __future__ import annotations

from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_pollution_index_distribution(df: pd.DataFrame, columns: Sequence[str], ax=None):
    """Boxplot of one or more pollution-index columns."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    df[list(columns)].boxplot(ax=ax)
    ax.set_ylabel("Index value")
    ax.set_title("Pollution index distributions")
    return ax


def plot_monte_carlo_density(samples: np.ndarray, threshold: float = 1.0,
                             title: str = "Monte Carlo HQ", ax=None):
    """Probability density of a Monte Carlo HQ/CR sample with threshold line."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    ax.hist(samples, bins=80, density=True, alpha=0.6, color="#1f77b4")
    p5, p50, p95 = np.percentile(samples, [5, 50, 95])
    for p, c in zip([p5, p50, p95], ["#999", "#000", "#d62728"]):
        ax.axvline(p, color=c, linestyle="--", linewidth=1)
    ax.axvline(threshold, color="red", linewidth=2, label=f"Threshold = {threshold}")
    ax.set_xlabel(title)
    ax.set_ylabel("Density")
    ax.legend()
    return ax


def plot_sensitivity_tornado(df: pd.DataFrame, parameter_col: str = "parameter",
                             value_col: str = "spearman_r", ax=None):
    """Horizontal tornado plot of sensitivity coefficients."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, max(3, 0.4 * len(df))))
    df_sorted = df.iloc[df[value_col].abs().argsort()]
    ax.barh(df_sorted[parameter_col], df_sorted[value_col],
            color=np.where(df_sorted[value_col] > 0, "#1f77b4", "#d62728"))
    ax.axvline(0, color="k", linewidth=0.8)
    ax.set_xlabel(value_col)
    ax.set_title("Sensitivity tornado plot")
    return ax


def plot_confusion_matrix(cm: pd.DataFrame, ax=None, cmap: str = "Blues"):
    """Heatmap of a confusion matrix."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm.values, cmap=cmap)
    ax.set_xticks(range(len(cm.columns)))
    ax.set_yticks(range(len(cm.index)))
    ax.set_xticklabels(cm.columns, rotation=45, ha="right")
    ax.set_yticklabels(cm.index)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm.iat[i, j]), ha="center", va="center", color="black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Reference")
    ax.set_title("Confusion matrix")
    plt.colorbar(im, ax=ax)
    return ax
