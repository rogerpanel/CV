"""Statistical tests used in the paper's cross-dataset comparisons.

* Friedman test + Nemenyi post-hoc (multi-model, multi-dataset)
* Wilcoxon signed-rank test (pairwise)
* Bonferroni correction helper
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import friedmanchisquare, rankdata, wilcoxon


@dataclass
class FriedmanResult:
    statistic: float
    p_value: float
    avg_ranks: np.ndarray


class StatisticalTests:
    # ------------------------------------------------------------------ #
    @staticmethod
    def friedman(scores: np.ndarray) -> FriedmanResult:
        """scores: (n_datasets, n_models) — larger is better."""
        if scores.ndim != 2:
            raise ValueError("Expected 2D score matrix.")
        stat, p = friedmanchisquare(*[scores[:, j] for j in range(scores.shape[1])])
        # Ranks per dataset (1 = best), averaged per model.
        ranks = np.array([rankdata(-row) for row in scores])
        return FriedmanResult(
            statistic=float(stat),
            p_value=float(p),
            avg_ranks=ranks.mean(axis=0),
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def nemenyi_critical_difference(
        avg_ranks: np.ndarray, n_datasets: int, q_alpha: float = 2.569,
    ) -> float:
        """CD = q_α · sqrt(k(k+1)/(6n))."""
        k = len(avg_ranks)
        return float(q_alpha * np.sqrt(k * (k + 1) / (6.0 * n_datasets)))

    # ------------------------------------------------------------------ #
    @staticmethod
    def wilcoxon_pair(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
        try:
            stat, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
            return float(stat), float(p)
        except ValueError:
            return 0.0, 1.0

    # ------------------------------------------------------------------ #
    @staticmethod
    def bonferroni(p_values: np.ndarray) -> np.ndarray:
        p = np.asarray(p_values, dtype=np.float64)
        return np.clip(p * len(p), 0.0, 1.0)
