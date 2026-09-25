"""Regularised Bismut–Elworthy–Li estimator of input gradients on the EM grid.

For X_{k+1} = X_k + f(X_k, t_k) Δt + g(X_k, t_k) ΔW_k, a direction v and the
tangent J_k v = ∂X_k/∂X_0 · v, the discrete Malliavin weight is

    M_v = (1/T) Σ_k ⟨ g⁺_λ(X_k) J_k v , ΔW_k ⟩,   g⁺_λ = (gᵀg + λ I_m)⁻¹ gᵀ,

and the estimator is ∂_v E φ(X_T) ≈ E[φ(X_T) M_v]. It needs no derivative
of φ. It is exact (as λ → 0 and Δt → 0) when m = d and g is invertible
(Elworthy & Li, 1994). With m < d or λ > 0 it estimates the derivative
only along the part of J_k v inside range g(X_k), so it is biased otherwise
(Proposition 1 of the paper). SODE-Guard trains θ_g with pathwise
gradients; this estimator is used only for ∇_x.
"""
from __future__ import annotations
import math
from typing import Callable

import torch
from torch.func import jvp, vmap

from .pseudoinverse import moore_penrose_diffusion

Drift = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
Diff = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def bel_gradient(f: Drift, g: Diff, x0: torch.Tensor, phi: Callable[[torch.Tensor], torch.Tensor], *,
                 T: float = 1.0, dt: float = 0.05, n_paths: int = 1024, floor: float = 1e-3,
                 generator: torch.Generator | None = None) -> torch.Tensor:
    """BEL estimate of ∇_{x0} E φ(X_T) for every row of x0 (shape (B, d))."""
    B, d = x0.shape
    n_steps = int(round(T / dt))
    x = x0.repeat_interleave(n_paths, dim=0)                              # (B·N, d)
    tangents = torch.eye(d, dtype=x0.dtype, device=x0.device).unsqueeze(1).expand(d, x.shape[0], d).contiguous()
    weight = torch.zeros(x.shape[0], d, dtype=x0.dtype, device=x0.device)
    for k in range(n_steps):
        t = torch.full((x.shape[0], 1), k * dt, dtype=x0.dtype, device=x0.device)
        gk = g(x, t)
        m = gk.shape[-1]
        dw = torch.randn(x.shape[0], m, generator=generator, dtype=x0.dtype, device=x0.device) * math.sqrt(dt)
        g_plus = moore_penrose_diffusion(gk, floor=floor)                   # (B·N, m, d)
        weight += torch.einsum("jnm,nm->nj", torch.einsum("nmd,jnd->jnm", g_plus, tangents), dw)

        def step(z, t=t, dw=dw):
            return z + f(z, t) * dt + torch.einsum("nij,nj->ni", g(z, t), dw)

        x_next = step(x)
        tangents = vmap(lambda v: jvp(step, (x,), (v,))[1])(tangents)
        x = x_next
    est = phi(x).unsqueeze(-1) * weight / T                                # (B·N, d)
    return est.view(B, n_paths, d).mean(dim=1)


def bel_input_gradient(model, x: torch.Tensor, phi: Callable[[torch.Tensor], torch.Tensor], *,
                       n_paths: int = 1024) -> torch.Tensor:
    """∇_x E φ(X_T) for SODE-Guard: BEL through the SDE, autograd through the encoder."""
    x = x.detach().requires_grad_(True)
    h0 = model.encode(x)
    floor = model.cfg.ellipticity_floor
    root = math.sqrt(floor)

    def g_floor(z, t):
        gz = model.diffusion(z, t)
        k = min(gz.shape[-2], gz.shape[-1])
        eye = torch.zeros_like(gz)
        idx = torch.arange(k, device=gz.device)
        eye[..., idx, idx] = root
        return gz + eye

    with torch.no_grad():
        grad_h0 = bel_gradient(model.drift, g_floor, h0.detach(), phi, T=model.cfg.horizon,
                               dt=model.cfg.dt, n_paths=n_paths, floor=floor)
    return torch.autograd.grad(h0, x, grad_outputs=grad_h0)[0]
