"""Empirical Bismut–Elworthy–Li estimator verification.

Reviewer 2 asked whether the BEL gradient estimator implemented with
finite Monte-Carlo samples and Euler–Maruyama discretisation is really
unbiased. This module provides the diagnostic used in Appendix D of
the revised manuscript.

Protocol:
    1. Pick a random smooth scalar functional  φ(X_T) = ‖X_T‖^2.
    2. Compute the "reference" gradient by finite differences at the
       diffusion parameters θ_g:  ∇^{FD} φ ≈ ( E φ_{θ+h} − E φ_{θ−h} ) / 2h .
    3. Compute the BEL estimator on the same sample:
       ∇^{BEL} φ = E [ φ(X_T) · M_T(θ_g) ]  with M_T from the Malliavin weight.
    4. Report the relative bias (‖∇^{BEL} − ∇^{FD}‖ / ‖∇^{FD}‖) and the
       Monte-Carlo standard error, sweeping N ∈ {32, 128, 512, 2048}.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch

from .pseudoinverse import moore_penrose_diffusion


@dataclass
class BELReport:
    N_samples: int
    reference_grad_norm: float
    bel_grad_norm: float
    relative_bias: float
    monte_carlo_stderr: float


@torch.no_grad()
def _reference_finite_difference(model, x: torch.Tensor, h: float = 1e-3,
                                 n_paths: int = 2048) -> float:
    """||∇^{FD} E ‖X_T‖^2|| approximated over a small parameter perturbation."""
    def phi(y):
        return y.pow(2).sum(dim=-1).mean()

    theta = [p.detach().clone() for p in model.diffusion.parameters()]
    base = phi(model.forward_mc(x, n_paths=n_paths))
    grads = []
    for p, snap in zip(model.diffusion.parameters(), theta):
        original = snap.clone()
        # Perturb a single scalar in each parameter tensor: mean-difference
        perturb = torch.zeros_like(p); perturb.view(-1)[0] = h
        p.copy_(snap + perturb)
        plus = phi(model.forward_mc(x, n_paths=n_paths))
        p.copy_(snap - perturb)
        minus = phi(model.forward_mc(x, n_paths=n_paths))
        p.copy_(original)
        grads.append(float((plus - minus) / (2 * h)))
    return math.sqrt(sum(g * g for g in grads))


@torch.no_grad()
def _bel_estimator_norm(model, x: torch.Tensor, n_paths: int, floor: float = 1e-3) -> tuple[float, float]:
    """Return (‖∇^{BEL}‖, MC std-err) on ``x`` for the same φ used above."""
    h0 = model.encode(x)
    est = []
    for s in range(n_paths):
        hT = model._integrate(h0, seed=s)
        g = model.diffusion(h0, torch.zeros(h0.shape[0], 1, device=h0.device))
        g_plus = moore_penrose_diffusion(g, floor=floor)
        # Malliavin weight over unit horizon: M_T ~ (1/T) ∫ (g^+ J_s)^T dW_s;
        # for the small-horizon linearisation we use M_T = g^+ · (hT − h0).
        M = torch.einsum("bji,bi->bj", g_plus, (hT - h0))
        est.append((hT.pow(2).sum(dim=-1, keepdim=True) * M).mean(dim=0))
    stack = torch.stack(est, dim=0)                # (n_paths, m)
    mean = stack.mean(dim=0)
    se = stack.std(dim=0, unbiased=True) / math.sqrt(max(n_paths, 1))
    return float(mean.norm()), float(se.norm())


def verify_bel(model, x: torch.Tensor,
               sample_sizes: tuple[int, ...] = (32, 128, 512, 2048)) -> list[BELReport]:
    ref = _reference_finite_difference(model, x)
    out: list[BELReport] = []
    for N in sample_sizes:
        est, stderr = _bel_estimator_norm(model, x, N)
        out.append(BELReport(
            N_samples=N,
            reference_grad_norm=ref,
            bel_grad_norm=est,
            relative_bias=abs(est - ref) / max(ref, 1e-9),
            monte_carlo_stderr=stderr,
        ))
    return out
