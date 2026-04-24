"""E-value martingale for anytime-valid sequential monitoring (Equation 8).

    E_t = ∏_{i=1}^t [1 + λ_i · (s(v_i, G_exec(i)) − q̂_{1−α})]

with λ_i ∈ [0, 1/(1 − q̂_{1−α})] chosen by online gradient descent on
−log E_t (equivalent to the KLY "betting" framework). By Ville's
inequality,

    P[∃ t : E_t ≥ 1/α]  ≤  α.

The workflow is terminated / escalated when E_t exceeds 1/α. The
martingale is built to be reusable across arbitrarily long trajectories —
it never expires.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EValueState:
    t: int = 0
    log_E: float = 0.0         # log-space for numerical stability
    lam: float = 0.0           # current bet size
    triggered: bool = False
    history: list[float] = field(default_factory=list)   # log-E at each step


class EValueMartingale:
    def __init__(
        self,
        q_hat: float,
        alpha: float = 0.05,
        lr: float = 0.1,
        lam_init: float = 0.0,
    ) -> None:
        self.q_hat = float(q_hat)
        self.alpha = float(alpha)
        self.lr = lr
        self.threshold = 1.0 / alpha
        self.state = EValueState(lam=lam_init)
        self._lam_upper = 1.0 / max(1.0 - self.q_hat, 1e-6)

    # ------------------------------------------------------------------ #
    # Update
    # ------------------------------------------------------------------ #
    def update(self, score: float) -> dict:
        """Consume the next nonconformity score s(v_t, G_exec(t))."""
        s = float(score)
        r = s - self.q_hat                                # residual
        factor = 1.0 + self.state.lam * r
        if factor <= 0:
            factor = 1e-9
        self.state.log_E += __import__("math").log(factor)
        self.state.t += 1
        self.state.history.append(self.state.log_E)

        # Online gradient-descent update on λ.
        # d (log E_t) / dλ = r / (1 + λ r)
        grad = r / factor
        new_lam = self.state.lam + self.lr * grad
        self.state.lam = float(min(max(new_lam, 0.0), self._lam_upper))

        if not self.state.triggered:
            import math
            if self.state.log_E >= math.log(self.threshold):
                self.state.triggered = True

        return {
            "t": self.state.t,
            "log_E": self.state.log_E,
            "E": __import__("math").exp(self.state.log_E),
            "triggered": self.state.triggered,
            "threshold": self.threshold,
            "lambda": self.state.lam,
        }

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    @property
    def triggered(self) -> bool:
        return self.state.triggered

    @property
    def e_value(self) -> float:
        import math
        return math.exp(self.state.log_E)

    def reset(self) -> None:
        self.state = EValueState(lam=0.0)
