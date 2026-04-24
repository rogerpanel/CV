"""Conformal-prediction evaluation: coverage, WSC, SSCV, bootstrap CI."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class ConformalReport:
    empirical_coverage: float
    worst_slab_coverage: float
    sscv: float
    mean_set_size: float
    bootstrap_ci: tuple[float, float]


class ConformalMetrics:
    @staticmethod
    def empirical_coverage(
        scores: np.ndarray, labels: np.ndarray, q_hat: float
    ) -> float:
        accepted = scores <= q_hat
        correct = (accepted & (labels == 1)) | (~accepted)
        return float(correct.mean())

    @staticmethod
    def worst_slab(
        scores: np.ndarray, labels: np.ndarray, q_hat: float, slabs: int = 5
    ) -> float:
        order = np.argsort(scores)
        buckets = np.array_split(order, slabs)
        vals = []
        for b in buckets:
            if len(b) == 0:
                continue
            accepted = scores[b] <= q_hat
            correct = (accepted & (labels[b] == 1)) | (~accepted)
            vals.append(float(correct.mean()))
        return min(vals) if vals else 0.0

    @staticmethod
    def sscv(
        scores: np.ndarray, labels: np.ndarray, q_hat: float, alpha: float,
        strata: int = 5,
    ) -> float:
        """Maximum deviation from 1-α across size strata."""
        order = np.argsort(scores)
        buckets = np.array_split(order, strata)
        worst = 0.0
        for b in buckets:
            if len(b) == 0:
                continue
            accepted = scores[b] <= q_hat
            correct = (accepted & (labels[b] == 1)) | (~accepted)
            worst = max(worst, abs(float(correct.mean()) - (1 - alpha)))
        return worst

    @staticmethod
    def bootstrap_ci(
        scores: np.ndarray, labels: np.ndarray, q_hat: float,
        n_bootstraps: int = 200, seed: int = 0,
    ) -> tuple[float, float]:
        rng = np.random.default_rng(seed)
        n = len(scores)
        bs = []
        for _ in range(n_bootstraps):
            idx = rng.integers(0, n, size=n)
            bs.append(
                ConformalMetrics.empirical_coverage(scores[idx], labels[idx], q_hat)
            )
        lo, hi = np.quantile(bs, [0.025, 0.975])
        return float(lo), float(hi)

    @staticmethod
    def report(
        scores: np.ndarray, labels: np.ndarray, q_hat: float, alpha: float,
    ) -> ConformalReport:
        return ConformalReport(
            empirical_coverage=ConformalMetrics.empirical_coverage(scores, labels, q_hat),
            worst_slab_coverage=ConformalMetrics.worst_slab(scores, labels, q_hat),
            sscv=ConformalMetrics.sscv(scores, labels, q_hat, alpha),
            mean_set_size=float(((scores <= q_hat).astype(float)).mean() + 1.0),
            bootstrap_ci=ConformalMetrics.bootstrap_ci(scores, labels, q_hat),
        )

    @staticmethod
    def as_dict(report: ConformalReport) -> dict:
        d = asdict(report)
        d["bootstrap_ci"] = list(d["bootstrap_ci"])
        return d
