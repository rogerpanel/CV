"""Detection-quality metrics (paper §6)."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    brier_score_loss,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


@dataclass
class DetectionReport:
    f1: float
    precision: float
    recall: float
    mcc: float
    kappa: float
    brier: float
    ece: float


class DetectionMetrics:
    @staticmethod
    def expected_calibration_error(
        probs: np.ndarray, labels: np.ndarray, num_bins: int = 15
    ) -> float:
        conf = probs.max(axis=-1)
        pred = probs.argmax(axis=-1)
        correct = (pred == labels).astype(np.float32)
        bins = np.linspace(0.0, 1.0, num_bins + 1)
        ece = 0.0
        for i in range(num_bins):
            mask = (conf >= bins[i]) & (conf < bins[i + 1] + (i == num_bins - 1))
            if mask.sum() == 0:
                continue
            bin_acc = correct[mask].mean()
            bin_conf = conf[mask].mean()
            ece += (mask.mean()) * abs(bin_acc - bin_conf)
        return float(ece)

    @staticmethod
    def report(
        probabilities: np.ndarray, labels: np.ndarray,
    ) -> DetectionReport:
        if probabilities.ndim == 1:
            probabilities = np.stack([1 - probabilities, probabilities], axis=-1)
        pred = probabilities.argmax(axis=-1)
        try:
            brier = brier_score_loss(
                (labels == 1).astype(int), probabilities[:, 1]
            )
        except Exception:
            brier = float("nan")
        return DetectionReport(
            f1=float(f1_score(labels, pred, average="macro", zero_division=0)),
            precision=float(precision_score(labels, pred, average="macro", zero_division=0)),
            recall=float(recall_score(labels, pred, average="macro", zero_division=0)),
            mcc=float(matthews_corrcoef(labels, pred)) if len(set(labels)) > 1 else 0.0,
            kappa=float(cohen_kappa_score(labels, pred)),
            brier=float(brier),
            ece=DetectionMetrics.expected_calibration_error(probabilities, labels),
        )

    @staticmethod
    def as_dict(report: DetectionReport) -> dict:
        return asdict(report)
