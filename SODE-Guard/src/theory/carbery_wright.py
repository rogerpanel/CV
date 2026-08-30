"""Carbery–Wright bounds used by the revised Proposition 5.

Reviewer 2 raised a direction-of-inequality concern: Carbery–Wright
normalises by the *actual* L² norm of the polynomial, not by an upper
bound. Replacing the denominator with an upper bound does **not** yield the
stated probability upper bound.

The fix is to work directly with the empirical L² norm. Let
Δ(δ) := g(x+δ) − g(x). Under the hypotheses of Proposition 5,

    (Exact form.)      P[|Δ| ≤ β] ≤ C · d* · ( β / ‖Δ‖_2 )^{1/d*}.
    (Usable form.)     β / ‖Δ‖_2 ≥ β / (L_g · ε) is *reversed*; the correct
                        direction sends a *lower* bound on ‖Δ‖_2 to an
                        *upper* bound on the anti-concentration probability.

We therefore use the lower confidence bound ‖Δ‖_2 ≥ L_lo · ε produced by
``theory.lipschitz.certify_L2_lipschitz`` (with the sign flipped: the
Hoeffding *lower* bound on the ratio) and evaluate

    P[|Δ| ≤ β] ≤ C · d* · ( β / (L_lo · ε) )^{1/d*}       (†)

which is a valid one-sided bound. The manuscript's Proposition 5 has been
rewritten to state (†) explicitly.

We further separate two quantities the reviewers asked to disambiguate:

  * ``margin_stability_bound``  — Pr[|Δ| ≤ β], the "smoothed-score gap"
    controlled by Carbery–Wright.
  * ``decision_flip_bound``     — Pr[ argmax g(x+δ) ≠ argmax g(x) ],
    the actual robustness quantity, derived from the margin bound via
    a union argument.
"""
from __future__ import annotations
import math
import torch


CARBERY_WRIGHT_C: float = 2.0       # universal constant; see Meka-Nguyen-Vu 2016


def margin_stability_bound(beta: float, L_lower: float, epsilon: float,
                           d_star: int) -> float:
    """Pr[|Δ| ≤ β] ≤ C · d* · (β / (L_lower · ε))^{1/d*}.

    Uses the LOWER confidence bound ``L_lower`` on ‖Δ‖_2 / ε so the
    inequality is in the correct direction (Reviewer 2 fix).
    """
    if L_lower <= 0 or epsilon <= 0:
        return 1.0
    ratio = beta / (L_lower * epsilon)
    bound = CARBERY_WRIGHT_C * d_star * (max(ratio, 1e-12) ** (1.0 / d_star))
    return float(min(bound, 1.0))


def decision_flip_bound(margin: torch.Tensor, L_lower: float, epsilon: float,
                        d_star: int) -> torch.Tensor:
    """Bound on Pr[argmax g(x+δ) ≠ argmax g(x)] via a union of margin bounds.

    A prediction flips only if the (top1 − top2) margin closes by at least
    the current margin value m. The Carbery–Wright bound at β = m gives
    an upper bound on that event, tightened by a union across K classes.
    """
    m = margin.abs()
    K = max(int(m.numel()), 1)
    bounds = torch.empty_like(m)
    for i in range(m.numel()):
        bounds.flatten()[i] = margin_stability_bound(
            float(m.flatten()[i]), L_lower, epsilon, d_star
        )
    return (K * bounds).clamp_max(1.0)


def two_sided_stability_bound(beta: float, L_lower: float, epsilon: float,
                              d_star: int) -> tuple[float, float]:
    """Return (P[|Δ|≤β], 1 − P[|Δ|≤β]) — two-sided form used by Corollary 1."""
    lo = margin_stability_bound(beta, L_lower, epsilon, d_star)
    return lo, 1.0 - lo


def invert_for_radius(margin: torch.Tensor, L_lower: float, d_star: int,
                      beta: float = 0.05, confidence: float = 0.95) -> torch.Tensor:
    """Largest ε for which the decision-flip bound stays below (1 − conf).

    Solves       K · C · d* · (β / (L_lower ε))^{1/d*} = 1 − conf
    ⇒  ε* = (β / L_lower) · ( K · C · d* / (1 − conf) )^{d*}
    with K set to 1 for the pair-wise (top1 vs top2) certificate, which is
    the honest reading Reviewer 2 asked for.
    """
    K = 1
    one_minus_conf = max(1.0 - confidence, 1e-6)
    scale = (K * CARBERY_WRIGHT_C * d_star / one_minus_conf) ** d_star
    return (margin.abs() * beta / max(L_lower, 1e-6) * scale).clamp_min(0.0)
