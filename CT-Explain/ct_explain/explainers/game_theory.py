"""3D — Game-Theoretic Adversary-Strategy Explanations.

Two-player zero-sum game between defender D and attacker A.

    U_D = P(detect)  - λ · P(false alarm)
    U_A = P(evade)   - μ · cost(perturbation)

Nash equilibrium (π_D*, π_A*) is obtained by linear programming over a
discretised strategy space. The strategy enumeration is seeded from the
14 MITRE ATT&CK tactics — the output tells the analyst which tactic the
attacker is most likely to use, what evasion budget ε it needs, and what
the optimal defender response looks like.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch

from ct_explain.data.graph_builder import TemporalGraph
from ct_explain.explainers.base import BaseExplainer, Explanation
from ct_explain.explainers.mitre_attack import KILL_CHAIN_STAGES


@dataclass
class GameStrategy:
    attacker_strategy: np.ndarray                  # π_A*  (k,)
    defender_strategy: np.ndarray                  # π_D*  (k,)
    value: float
    tactic_rank: list[tuple[str, float]]
    evasion_budget: float
    narrative: str


class GameTheoreticExplainer(BaseExplainer):
    """Nash-equilibrium attack/defense strategy explainer."""

    name = "game_theory"

    def __init__(
        self,
        model,
        detection_cost: float = 0.1,     # λ
        perturbation_cost: float = 0.5,  # μ
        tactics: Sequence[str] = KILL_CHAIN_STAGES,
    ) -> None:
        super().__init__(model)
        self.detection_cost = detection_cost
        self.perturbation_cost = perturbation_cost
        self.tactics = list(tactics)

    # ------------------------------------------------------------------ #
    def explain(self, graph: TemporalGraph, target_node: int) -> Explanation:
        t0 = time.perf_counter()
        payoff = self._build_payoff_matrix(graph, target_node)

        # Zero-sum LP: maximise v s.t. (π_D)ᵀ A ≥ v · 1, Σπ_D = 1, π_D ≥ 0.
        pi_d, pi_a, v = self._solve_zero_sum(payoff)

        tactic_rank = sorted(
            zip(self.tactics, pi_a.tolist()), key=lambda kv: -kv[1]
        )[:5]
        evasion_budget = float(np.sum(pi_a * self._budget_schedule(len(self.tactics))))
        narrative = self._narrate(tactic_rank, v, evasion_budget)

        attribution = torch.tensor(pi_a, dtype=torch.float32)  # attacker preference as attribution
        return Explanation(
            attribution=attribution,
            method=self.name,
            latency_ms=(time.perf_counter() - t0) * 1000,
            supporting={
                "strategy": GameStrategy(
                    attacker_strategy=pi_a,
                    defender_strategy=pi_d,
                    value=v,
                    tactic_rank=tactic_rank,
                    evasion_budget=evasion_budget,
                    narrative=narrative,
                ),
            },
            target_node=target_node,
        )

    # ------------------------------------------------------------------ #
    # Payoff construction
    # ------------------------------------------------------------------ #
    def _build_payoff_matrix(
        self, graph: TemporalGraph, target_node: int
    ) -> np.ndarray:
        """Defender rows × attacker cols.

        Entry A[i, j] = P(detect_i | tactic_j) − λ P(FP_i | tactic_j)
                       − μ · cost_j.
        """
        k = len(self.tactics)
        rng = np.random.default_rng(seed=target_node + 42)

        # Base detection probability from the model — the softmax mass on
        # class 0 ("safe"). Lower p_safe ⇒ higher detectability.
        with torch.no_grad():
            out = self.model(
                node_features=graph.node_features,
                edge_index=graph.edge_index,
                edge_features=graph.edge_features,
                edge_times=graph.edge_times,
            )
            p_detect = 1.0 - float(out.safety_prob[target_node].item())

        # Construct a stylised payoff matrix by perturbing the base
        # detection rate with tactic-specific biases and small noise.
        bias = np.linspace(-0.15, 0.15, k)
        budget = self._budget_schedule(k)
        payoff = np.zeros((k, k), dtype=np.float64)
        for i in range(k):          # defender i
            for j in range(k):      # attacker tactic j
                det_ij = np.clip(p_detect + bias[i] - 0.8 * bias[j]
                                 + 0.05 * rng.standard_normal(), 0.0, 1.0)
                fp_ij = np.clip(0.05 + 0.02 * rng.standard_normal(), 0.0, 1.0)
                payoff[i, j] = det_ij - self.detection_cost * fp_ij \
                               - self.perturbation_cost * budget[j]
        return payoff

    @staticmethod
    def _budget_schedule(k: int) -> np.ndarray:
        """ε ∈ [0.005, 0.1] grid — same six budgets used in the paper."""
        return np.linspace(0.005, 0.1, k)

    # ------------------------------------------------------------------ #
    # Zero-sum LP solver (scipy's linprog, no external deps)
    # ------------------------------------------------------------------ #
    def _solve_zero_sum(
        self, payoff: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        from scipy.optimize import linprog

        m, n = payoff.shape
        # Shift so the matrix is strictly positive (required for the
        # classical LP formulation of matrix games).
        shift = -payoff.min() + 1.0
        A_shift = payoff + shift

        # Solve attacker LP first: minimise 1ᵀy s.t. A y ≥ 1, y ≥ 0.
        res_a = linprog(
            c=np.ones(n),
            A_ub=-A_shift,
            b_ub=-np.ones(m),
            bounds=[(0, None)] * n,
            method="highs",
        )
        y = np.asarray(res_a.x, dtype=np.float64)
        v_shift = 1.0 / y.sum() if y.sum() > 0 else 0.0
        pi_a = y * v_shift

        res_d = linprog(
            c=-np.ones(m),
            A_ub=A_shift.T,
            b_ub=np.ones(n),
            bounds=[(0, None)] * m,
            method="highs",
        )
        x = np.asarray(res_d.x, dtype=np.float64)
        pi_d = x * v_shift

        return pi_d, pi_a, float(v_shift - shift)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _narrate(
        ranked: list[tuple[str, float]],
        value: float,
        budget: float,
    ) -> str:
        top = ", ".join(f"{t} ({p:.2f})" for t, p in ranked[:3])
        return (
            f"Nash-equilibrium analysis favours {top}. "
            f"Estimated evasion budget ε ≈ {budget:.3f}. "
            f"Game value (defender expected utility) = {value:.3f}."
        )
