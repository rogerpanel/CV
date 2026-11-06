"""Cohen-Rosenfeld-Kolter (2019) randomised-smoothing certificate.

S&P 2027 audit (Part C.8): the paper previously claimed a smoothing
certificate without actually implementing one. This module adds the
canonical two-stage procedure:

    1. Draw ``n0`` noisy samples of ``f(x + N(0, sigma^2 I))`` and let
       ``cA`` be the most-frequent predicted class.
    2. Draw ``n`` fresh noisy samples and count ``k`` = votes for cA.
       Compute the one-sided Clopper-Pearson lower bound ``pA_lower``
       on Pr[f(x + eta) = cA] at confidence ``1 - alpha``.
    3. If ``pA_lower <= 1/2`` return ABSTAIN; else return cA with
       certified radius ``sigma * Phi^{-1}(pA_lower)``.

References
----------
Cohen, Rosenfeld & Kolter, "Certified Adversarial Robustness via
Randomised Smoothing", ICML 2019.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch

try:
    from scipy.stats import beta as _beta_dist
    from scipy.stats import norm as _norm_dist
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover - fall back to erfinv-based Phi^-1.
    _HAS_SCIPY = False


ABSTAIN: int = -1


# ------------------------------------------------------------------ #
def _clopper_pearson_lower(k: int, n: int, alpha: float) -> float:
    """One-sided lower Clopper-Pearson bound on a Binomial proportion.

    Returns the largest ``p`` such that ``Pr[Bin(n, p) >= k] <= alpha``,
    i.e. ``p_lower = BetaInv(alpha; k, n - k + 1)``.
    """
    if k <= 0:
        return 0.0
    if k >= n:
        # The lower bound collapses to alpha^{1/n}.
        return float(alpha ** (1.0 / max(n, 1)))
    if _HAS_SCIPY:
        return float(_beta_dist.ppf(alpha, k, n - k + 1))
    # Bisection fallback using the regularised incomplete-beta identity
    # Pr[Bin(n, p) >= k] = I_p(k, n - k + 1).
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        # Direct binomial tail via cumulative formula; stable for small n.
        logp = math.log(mid) if mid > 0 else -1e30
        log1mp = math.log1p(-mid) if mid < 1 else -1e30
        tail = 0.0
        term = math.exp(k * logp + (n - k) * log1mp)
        # log C(n, k)
        from math import lgamma
        logC = lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)
        # Direct survival function via regularised incomplete beta is
        # cleaner, but a cumulative sum over tail suffices for n <= 1e5.
        tail = math.exp(logC + k * logp + (n - k) * log1mp)
        for i in range(k + 1, n + 1):
            logCi = lgamma(n + 1) - lgamma(i + 1) - lgamma(n - i + 1)
            tail += math.exp(logCi + i * logp + (n - i) * log1mp)
        if tail > alpha:
            hi = mid
        else:
            lo = mid
    return lo


def _phi_inv(p: float) -> float:
    """Inverse standard-normal CDF."""
    if _HAS_SCIPY:
        return float(_norm_dist.ppf(p))
    # Fallback using the erf-based identity Phi^{-1}(p) = sqrt(2) * erf^{-1}(2p - 1).
    return math.sqrt(2.0) * math.erfinv(2.0 * p - 1.0)


# ------------------------------------------------------------------ #
@dataclass
class SmoothingResult:
    predicted_class: int
    radius: float
    n0: int
    n: int
    sigma: float
    pA_lower: float
    abstain: bool

    def as_dict(self) -> dict:
        return {
            "predicted_class": int(self.predicted_class),
            "radius": float(self.radius),
            "n0": int(self.n0),
            "n": int(self.n),
            "sigma": float(self.sigma),
            "pA_lower": float(self.pA_lower),
            "abstain": bool(self.abstain),
        }


# ------------------------------------------------------------------ #
def _predict_noisy(
    model: Callable[..., Any],
    x: Mapping[str, torch.Tensor] | torch.Tensor,
    key: str | None,
    sigma: float,
    n: int,
    num_classes: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Return a ``(num_classes,)`` vote histogram over n noisy samples."""
    counts = torch.zeros(num_classes, dtype=torch.long, device=device)
    remaining = n
    model_eval = getattr(model, "eval", lambda: None)
    model_eval()
    while remaining > 0:
        b = min(batch_size, remaining)
        remaining -= b
        with torch.no_grad():
            if isinstance(x, torch.Tensor):
                noise = torch.randn((b, *x.shape), device=device) * sigma
                xb = x.unsqueeze(0).expand(b, *x.shape) + noise
                out = model(xb)
            else:
                # Dict batch: perturb payload embedding `key`; repeat along batch dim.
                k = key or "p"
                base = x[k]
                noise = torch.randn((b, *base.shape), device=device) * sigma
                xb = base.unsqueeze(0).expand(b, *base.shape) + noise
                # Flatten (b, A, L, d) -> (b*A, L, d) but keep per-sample votes.
                new_batch = {}
                for kk, vv in x.items():
                    if kk == k:
                        continue
                    new_batch[kk] = vv.unsqueeze(0).expand(b, *vv.shape).reshape(
                        b * vv.shape[0], *vv.shape[1:]
                    )
                new_batch[k] = xb.reshape(b * base.shape[0], *base.shape[1:])
                out = model(new_batch)
                logits = out["logits"] if isinstance(out, dict) else out
                # Collapse per-agent to a single class vote by taking the
                # argmax of the pooled logits over agents in each copy.
                A = base.shape[0]
                logits = logits.view(b, A, num_classes).mean(dim=1)
                preds = logits.argmax(dim=-1)
                counts.scatter_add_(
                    0, preds.to(device), torch.ones_like(preds, device=device)
                )
                continue
            logits = out["logits"] if isinstance(out, dict) else out
            preds = logits.argmax(dim=-1)
            counts.scatter_add_(
                0, preds.to(device), torch.ones_like(preds, device=device)
            )
    return counts


def cohen_certify(
    model: Callable[..., Any],
    x: Mapping[str, torch.Tensor] | torch.Tensor,
    sigma: float,
    n0: int,
    n: int,
    alpha: float,
    num_classes: int,
    batch_size: int = 64,
    input_key: str | None = None,
) -> SmoothingResult:
    """Two-stage randomised-smoothing certificate (Cohen et al. 2019).

    Parameters
    ----------
    model : callable
        Base classifier; either ``model(tensor)`` or ``model(dict)``.
    x : tensor or mapping
        Clean input. For a mapping, ``input_key`` names the field whose
        values are perturbed (default: ``"p"``).
    sigma : float
        Standard deviation of the Gaussian smoothing noise.
    n0 : int
        Number of noise samples used to select the top class.
    n : int
        Number of noise samples used to bound pA.
    alpha : float
        Clopper-Pearson confidence failure level.
    num_classes : int
        Size of the label alphabet.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    if n0 <= 0 or n <= 0:
        raise ValueError("n0, n must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    device = _infer_device(x)

    counts0 = _predict_noisy(
        model, x, input_key, sigma, n0, num_classes, batch_size, device
    )
    cA = int(counts0.argmax().item())
    counts = _predict_noisy(
        model, x, input_key, sigma, n, num_classes, batch_size, device
    )
    k_a = int(counts[cA].item())
    pA_lower = _clopper_pearson_lower(k_a, n, alpha)
    if pA_lower <= 0.5:
        return SmoothingResult(
            predicted_class=ABSTAIN,
            radius=0.0,
            n0=n0,
            n=n,
            sigma=sigma,
            pA_lower=float(pA_lower),
            abstain=True,
        )
    radius = float(sigma) * _phi_inv(pA_lower)
    return SmoothingResult(
        predicted_class=cA,
        radius=float(radius),
        n0=n0,
        n=n,
        sigma=sigma,
        pA_lower=float(pA_lower),
        abstain=False,
    )


def _infer_device(x: Mapping[str, torch.Tensor] | torch.Tensor) -> torch.device:
    if isinstance(x, torch.Tensor):
        return x.device
    for v in x.values():
        if isinstance(v, torch.Tensor):
            return v.device
    return torch.device("cpu")


# ------------------------------------------------------------------ #
def certified_accuracy_curve(
    results: list[SmoothingResult],
    labels: list[int],
    radii: list[float],
) -> list[dict]:
    """Return CertAcc(r) = fraction of samples certified AND correct at r."""
    out: list[dict] = []
    N = len(labels)
    for r in radii:
        ok = 0
        for res, y in zip(results, labels):
            if res.abstain:
                continue
            if res.predicted_class != int(y):
                continue
            if res.radius >= r:
                ok += 1
        out.append({"radius": float(r), "cert_acc": float(ok / max(N, 1))})
    return out
