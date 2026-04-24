"""ConformalGuard orchestrator.

Wires together the graph-conformal calibrator (Theorem 1) and the E-value
martingale (Ville's inequality) so a SOC integrator gets a single object
with three methods:

    guard.calibrate(calibration_set)
    verdict = guard.certify(action_id, exec_graph, p_safe_fn)
    status  = guard.monitor(action_id, exec_graph, p_safe_fn)

The two statistical layers are complementary:

* `certify` performs the fixed-horizon guarantee (P[miscoverage] ≤ α
  per workflow).
* `monitor` adds the anytime-valid guarantee over streaming workflows
  (P[∃ t : false-accept] ≤ α).

A workflow is *terminated* when either `Abstain` on any action or the
martingale trigger fires. That is the contract consumed by the SOC
dashboard (`ct_explain.soc.dashboard`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from ct_explain.conformal.e_value import EValueMartingale
from ct_explain.conformal.execution_graph import DynamicExecutionGraph
from ct_explain.conformal.graph_conformal import (
    GraphConformalCalibrator,
    NonconformityScorer,
)


@dataclass
class ConformalVerdict:
    verdict: str                     # "Safe" | "Abstain" | "Escalate"
    score: float
    q_hat: float
    e_value: float
    triggered: bool
    reason: str


class ConformalGuard:
    def __init__(
        self,
        alpha: float = 0.05,
        k_hops: int = 2,
        lambda_cascade: float = 0.5,
        martingale_lr: float = 0.1,
    ) -> None:
        self.alpha = alpha
        self.scorer = NonconformityScorer(k_hops=k_hops, lambda_cascade=lambda_cascade)
        self.calibrator = GraphConformalCalibrator(scorer=self.scorer, alpha=alpha)
        self._martingale: Optional[EValueMartingale] = None
        self._martingale_lr = martingale_lr

    # ------------------------------------------------------------------ #
    # Calibration
    # ------------------------------------------------------------------ #
    def calibrate(
        self,
        calibration: Sequence[tuple[str, DynamicExecutionGraph, Callable[[str], float], int]],
    ) -> float:
        res = self.calibrator.calibrate(calibration)
        self._martingale = EValueMartingale(
            q_hat=res.q_hat, alpha=self.alpha, lr=self._martingale_lr
        )
        return res.q_hat

    # ------------------------------------------------------------------ #
    # Fixed-horizon certify
    # ------------------------------------------------------------------ #
    def certify(
        self,
        node_id: str,
        exec_graph: DynamicExecutionGraph,
        p_safe_fn: Callable[[str], float],
    ) -> ConformalVerdict:
        if self._martingale is None:
            raise RuntimeError("Call .calibrate() before .certify().")
        result = self.calibrator.predict(node_id, exec_graph, p_safe_fn)
        verdict = result["verdict"]
        reason = (
            "nonconformity below calibrated threshold"
            if verdict == "Safe"
            else "nonconformity exceeds q̂_{1-α}"
        )
        return ConformalVerdict(
            verdict=verdict,
            score=result["score"],
            q_hat=result["q_hat"],
            e_value=self._martingale.e_value,
            triggered=self._martingale.triggered,
            reason=reason,
        )

    # ------------------------------------------------------------------ #
    # Anytime-valid monitor
    # ------------------------------------------------------------------ #
    def monitor(
        self,
        node_id: str,
        exec_graph: DynamicExecutionGraph,
        p_safe_fn: Callable[[str], float],
    ) -> ConformalVerdict:
        if self._martingale is None:
            raise RuntimeError("Call .calibrate() before .monitor().")
        score = self.scorer.score(node_id, exec_graph, p_safe_fn)
        update = self._martingale.update(score)

        if update["triggered"]:
            verdict = "Escalate"
            reason = (
                f"E-value {update['E']:.2f} exceeded 1/α threshold ({1/self.alpha:.1f})"
            )
        elif score > self.calibrator.q_hat:
            verdict = "Abstain"
            reason = "nonconformity exceeds q̂_{1-α}"
        else:
            verdict = "Safe"
            reason = "nonconformity below calibrated threshold"

        return ConformalVerdict(
            verdict=verdict,
            score=score,
            q_hat=self.calibrator.q_hat,
            e_value=update["E"],
            triggered=update["triggered"],
            reason=reason,
        )

    # ------------------------------------------------------------------ #
    def reset_monitor(self) -> None:
        if self._martingale is not None:
            self._martingale.reset()
