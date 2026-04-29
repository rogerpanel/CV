"""Classification accuracy assessment.

* Confusion matrix and per-class user / producer accuracies.
* Cohen's kappa with the standard six-class qualitative scale.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from cmhr.utils import load_config
from cmhr.pollution_indices._classify import classify


def confusion_metrics(
    y_true: Sequence,
    y_pred: Sequence,
    class_labels: Sequence | None = None,
) -> dict:
    """Return overall, kappa, and per-class accuracies."""
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    n = cm.sum()
    overall = cm.trace() / n if n else float("nan")

    producer_acc = np.divide(cm.diagonal(), cm.sum(axis=1, where=cm.sum(axis=1) > 0),
                             where=cm.sum(axis=1) > 0,
                             out=np.full(cm.shape[0], np.nan))
    user_acc = np.divide(cm.diagonal(), cm.sum(axis=0, where=cm.sum(axis=0) > 0),
                         where=cm.sum(axis=0) > 0,
                         out=np.full(cm.shape[0], np.nan))

    return {
        "confusion_matrix": pd.DataFrame(cm, index=class_labels, columns=class_labels),
        "overall_accuracy": float(overall),
        "kappa": cohen_kappa(y_true, y_pred),
        "producer_accuracy": dict(zip(class_labels or range(len(producer_acc)), producer_acc)),
        "user_accuracy": dict(zip(class_labels or range(len(user_acc)), user_acc)),
    }


def cohen_kappa(y_true: Sequence, y_pred: Sequence) -> float:
    """Cohen's :math:`\\kappa = (p_o - p_e) / (1 - p_e)`."""
    cm = confusion_matrix(y_true, y_pred)
    n = cm.sum()
    if n == 0:
        return float("nan")
    po = cm.trace() / n
    pe = (cm.sum(axis=0) * cm.sum(axis=1)).sum() / (n * n)
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def classify_kappa(value: float) -> str:
    """Apply the Landis & Koch (1977) qualitative scale."""
    rules = load_config("thresholds")["cohen_kappa"]
    return classify([value], rules)[0]
