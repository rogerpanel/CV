"""Statistical analysis utilities (Wilson score interval, curve crossings)."""

from __future__ import annotations

import math
from typing import Tuple

# z-scores for common two-sided confidence levels
_Z = {0.90: 1.6448536269514722, 0.95: 1.959963984540054, 0.99: 2.5758293035489004}


def z_for(confidence: float) -> float:
    if confidence in _Z:
        return _Z[confidence]
    # inverse-normal via rational approximation (Acklam) for arbitrary level
    p = 1.0 - (1.0 - confidence) / 2.0
    return _norm_ppf(p)


def wilson_interval(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Parameters
    ----------
    k : number of successes
    n : number of trials
    confidence : two-sided confidence level (default 0.95)

    Returns
    -------
    (low, high) clamped to [0, 1].
    """
    if n == 0:
        return (float("nan"), float("nan"))
    z = z_for(confidence)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def _norm_ppf(p: float) -> float:  # pragma: no cover - fallback path
    """Acklam's inverse normal CDF approximation."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
