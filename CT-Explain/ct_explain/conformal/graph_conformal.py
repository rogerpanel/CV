"""Graph conformal prediction (Theorem 1 + Equation 7).

    s(v_t, G_exec(t)) = [1 − p_safe(v_t)]
                       + λ_cascade · max_{u ∈ N_k(v_t)} Inf(u→v_t) · (1 − p_safe(u))

Under workflow exchangeability (Assumption 1), building the prediction set
C(v_t) = "Safe" iff s ≤ q̂_{1−α} yields

    P[∀ v ∈ V_action^(n+1): C(v)=Safe ⟹ v is safe]  ≥  1 − α.

This module provides:

* `NonconformityScorer`   – pure Eq. 7 implementation.
* `GraphConformalCalibrator` – split-conformal procedure that computes
  q̂_{1−α} from a calibration set, plus worst-slab / size-stratified
  coverage diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from ct_explain.conformal.execution_graph import DynamicExecutionGraph


# --------------------------------------------------------------------- #
# Nonconformity score (Eq. 7)
# --------------------------------------------------------------------- #

@dataclass
class NonconformityScorer:
    """Composite conformal score for actions in a dynamic execution graph."""

    k_hops: int = 2
    lambda_cascade: float = 0.5

    def score(
        self,
        node_id: str,
        exec_graph: DynamicExecutionGraph,
        p_safe: Callable[[str], float],
    ) -> float:
        local = 1.0 - float(p_safe(node_id))
        neighbours = exec_graph.k_hop_neighbours(node_id, self.k_hops)
        cascade = 0.0
        for u in neighbours:
            try:
                inf = exec_graph.influence(u, node_id)
            except Exception:  # pragma: no cover
                inf = 0.0
            cascade = max(cascade, inf * (1.0 - float(p_safe(u))))
        return local + self.lambda_cascade * cascade


# --------------------------------------------------------------------- #
# Split-conformal calibrator
# --------------------------------------------------------------------- #

@dataclass
class ConformalQuantile:
    q_hat: float
    alpha: float
    n_calibration: int


class GraphConformalCalibrator:
    def __init__(
        self,
        scorer: NonconformityScorer,
        alpha: float = 0.05,
    ) -> None:
        self.scorer = scorer
        self.alpha = alpha
        self._quantile: Optional[ConformalQuantile] = None

    # ------------------------------------------------------------------ #
    def calibrate(
        self,
        calibration: Sequence[tuple[str, DynamicExecutionGraph, Callable[[str], float], int]],
    ) -> ConformalQuantile:
        """``calibration`` holds tuples (node_id, G_exec, p_safe_fn, y_true).

        Per the manuscript we compute the (1 − α)-quantile over nonconformity
        scores observed on *truly safe* calibration points — the prediction
        set then guarantees P[miscoverage] ≤ α on future exchangeable
        workflows (Theorem 1).
        """
        scores = [
            self.scorer.score(nid, g, p_safe)
            for nid, g, p_safe, y in calibration
            if int(y) == 1    # only safe actions contribute
        ]
        if not scores:
            raise ValueError("No safe examples found in calibration set.")
        n = len(scores)
        rank = int(np.ceil((1 - self.alpha) * (n + 1))) - 1
        rank = max(0, min(rank, n - 1))
        q_hat = float(np.sort(scores)[rank])
        self._quantile = ConformalQuantile(q_hat=q_hat, alpha=self.alpha, n_calibration=n)
        return self._quantile

    # ------------------------------------------------------------------ #
    @property
    def q_hat(self) -> float:
        if self._quantile is None:
            raise RuntimeError("Calibrate before querying q_hat.")
        return self._quantile.q_hat

    def predict(
        self,
        node_id: str,
        exec_graph: DynamicExecutionGraph,
        p_safe: Callable[[str], float],
    ) -> dict:
        score = self.scorer.score(node_id, exec_graph, p_safe)
        verdict = "Safe" if score <= self.q_hat else "Abstain"
        return {"score": score, "q_hat": self.q_hat, "verdict": verdict}

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #
    def evaluate_coverage(
        self,
        test_set: Sequence[tuple[str, DynamicExecutionGraph, Callable[[str], float], int]],
    ) -> dict:
        """Empirical coverage + worst-slab and size-stratified coverage."""
        scores: list[float] = []
        truths: list[int] = []
        for nid, g, p_safe, y in test_set:
            scores.append(self.scorer.score(nid, g, p_safe))
            truths.append(int(y))
        scores_a = np.array(scores)
        truths_a = np.array(truths)
        accepted = scores_a <= self.q_hat
        # "Prediction set = Safe" is correct when the ground-truth label == 1 (safe).
        coverage = float(((accepted & (truths_a == 1)) | (~accepted)).mean())

        # Worst-Slab Coverage over 5 score-quantile bins.
        order = np.argsort(scores_a)
        bins = np.array_split(order, 5)
        wsc = min(
            float(((accepted[b] & (truths_a[b] == 1)) | (~accepted[b])).mean())
            for b in bins if len(b) > 0
        )

        # Size-stratified — approximated by score bucket size.
        ssc_violation = max(
            abs((1 - self.alpha) - (
                float(((accepted[b] & (truths_a[b] == 1)) | (~accepted[b])).mean())
            ))
            for b in bins if len(b) > 0
        )
        return {
            "coverage": coverage,
            "worst_slab_coverage": wsc,
            "ssc_violation": float(ssc_violation),
            "abstention_rate": float((~accepted).mean()),
        }
