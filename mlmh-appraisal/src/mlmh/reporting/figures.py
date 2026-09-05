"""Reliability curves, ROC curves and E1 inflation plots as PDF."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from ..evaluation.metrics import reliability_curve

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})


def _mark(fig, synthetic: bool):
    if synthetic:
        fig.text(0.5, 0.5, "SYNTHETIC FIXTURE", fontsize=28, color="red", alpha=0.18, ha="center", va="center", rotation=25)


def reliability_plot(preds: dict[str, pd.DataFrame], path: Path, title: str = "", synthetic: bool = False) -> Path:
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=0.8, label="perfect")
    for name, pred in preds.items():
        rc = reliability_curve(pred["y"], pred["p"])
        ax.plot(rc["p_mean"], rc["y_mean"], marker="o", ms=3, lw=1, label=name)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed proportion")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=6)
    _mark(fig, synthetic)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return Path(path)


def roc_plot(preds: dict[str, pd.DataFrame], path: Path, title: str = "", synthetic: bool = False) -> Path:
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=0.8)
    for name, pred in preds.items():
        if pred["y"].nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(pred["y"], pred["p"])
        ax.plot(fpr, tpr, lw=1, label=name)
    ax.set_xlabel("1 - specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=6)
    _mark(fig, synthetic)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return Path(path)


def inflation_plot(table: pd.DataFrame, path: Path, metric: str = "auroc", synthetic: bool = False) -> Path:
    """Dot plot: subject-wise vs record-wise estimate per model and cohort."""
    cohorts = sorted(table["cohort"].unique())
    fig, axes = plt.subplots(1, len(cohorts), figsize=(2.6 * len(cohorts), 3.0), sharey=True, squeeze=False)
    for ax, cohort in zip(axes[0], cohorts):
        sub = table[table["cohort"] == cohort]
        ys = np.arange(len(sub))
        ax.hlines(ys, sub[f"{metric}_subject_wise"], sub[f"{metric}_record_wise"], color="lightgrey", lw=3)
        ax.plot(sub[f"{metric}_subject_wise"], ys, "o", color="#1f4e5f", label="subject-wise")
        ax.plot(sub[f"{metric}_record_wise"], ys, "o", color="#8c2f1e", label="record-wise")
        ax.set_yticks(ys)
        ax.set_yticklabels(sub["model"])
        ax.set_title(cohort)
        ax.set_xlabel(metric.upper())
        ax.set_xlim(0.4, 1.0)
    axes[0][0].legend(frameon=False, fontsize=6, loc="lower right")
    _mark(fig, synthetic)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return Path(path)
