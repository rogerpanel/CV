"""Statistical tests used by the manuscript.

The TNNLS paper reports a Friedman χ²(9) = 58.7 across the 9 baselines plus
SODE-Guard (10 systems), and McNemar against the strongest internal model
SDE-TGNN with p < 10⁻⁸.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.contingency_tables import mcnemar


def friedman_test(score_matrix: np.ndarray) -> dict:
    """``score_matrix`` shape (n_methods, n_datasets_or_seeds)."""
    stat, p = friedmanchisquare(*score_matrix)
    return {"statistic": float(stat), "p_value": float(p),
            "df": score_matrix.shape[0] - 1, "n": score_matrix.shape[1]}


def mcnemar_test(pred_a: np.ndarray, pred_b: np.ndarray,
                 y_true: np.ndarray, exact: bool = False) -> dict:
    a_correct = (pred_a == y_true); b_correct = (pred_b == y_true)
    table = [[int((a_correct & b_correct).sum()),  int((a_correct & ~b_correct).sum())],
             [int((~a_correct & b_correct).sum()), int((~a_correct & ~b_correct).sum())]]
    res = mcnemar(table, exact=exact, correction=True)
    return {"statistic": float(res.statistic), "p_value": float(res.pvalue),
            "contingency": table}


def wilcoxon_test(scores_a: np.ndarray, scores_b: np.ndarray) -> dict:
    stat, p = wilcoxon(scores_a, scores_b)
    return {"statistic": float(stat), "p_value": float(p)}


def bootstrap_ci(values: np.ndarray, *, iters: int = 1000, ci: float = 0.95,
                 seed: int = 42) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    boots = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        boots[i] = rng.choice(values, size=values.size, replace=True).mean()
    lo = float(np.quantile(boots, (1 - ci) / 2))
    hi = float(np.quantile(boots, 1 - (1 - ci) / 2))
    return float(values.mean()), lo, hi
