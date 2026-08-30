"""Maurer PAC-Bayes-kl bound used by Theorem 4 of the revised manuscript.

Reviewer 1 asked that the PAC-Bayes certificate be connected to the
reported calibration errors instead of appearing as an unused
preliminary. This module returns the kl-inversion certificate:

    R(Q) ≤ kl^{-1}( R̂_S(Q) , [KL(Q‖P) + log(2√n / δ)] / n ).

We use the Reeb-Seldin closed-form kl-inversion (Reeb & Seldin, 2014,
Thm. 8) that is safe and monotone. The returned certificate is
combined with the empirical calibration error in
``docs/response_to_reviewers/`` and reported in Table 5 of the
revised manuscript.
"""
from __future__ import annotations
import math


def _kl_bernoulli(p: float, q: float) -> float:
    p = min(max(p, 1e-12), 1 - 1e-12)
    q = min(max(q, 1e-12), 1 - 1e-12)
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def _kl_inverse_upper(p_hat: float, bound: float, tol: float = 1e-9) -> float:
    """Largest q ∈ [p_hat, 1] with kl(p_hat‖q) ≤ bound (binary search)."""
    lo, hi = p_hat, 1.0 - 1e-12
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _kl_bernoulli(p_hat, mid) > bound:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return hi


def pac_bayes_bound(empirical_risk: float,
                    kl_divergence: float,
                    n_samples: int,
                    delta: float = 0.05) -> dict:
    """Return the Maurer PAC-Bayes-kl certificate as a dict.

    Parameters
    ----------
    empirical_risk : mean 0/1 loss on the training or validation split
        (must lie in [0, 1]).
    kl_divergence : KL(Q ‖ P) between posterior (trained SODE-Guard) and
        the prior fixed at the diffusion floor.
    n_samples : number of i.i.d. training examples used to compute
        ``empirical_risk``.
    delta : failure probability of the certificate (default 0.05).
    """
    if not 0.0 <= empirical_risk <= 1.0:
        raise ValueError("empirical_risk must be in [0, 1]")
    if kl_divergence < 0:
        raise ValueError("kl_divergence must be non-negative")
    bound_rhs = (kl_divergence + math.log(2.0 * math.sqrt(n_samples) / delta)) / n_samples
    risk_upper = _kl_inverse_upper(empirical_risk, bound_rhs)
    return {
        "empirical_risk": float(empirical_risk),
        "kl_divergence": float(kl_divergence),
        "n_samples": int(n_samples),
        "delta": float(delta),
        "kl_rhs": float(bound_rhs),
        "population_risk_upper": float(risk_upper),
        "gap": float(risk_upper - empirical_risk),
    }
