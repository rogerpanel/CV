"""Adjoint-sensitivity helpers.

When `torchdiffeq` is available we simply call `odeint_adjoint` — nothing
more to do. This module supplies a *manual* adjoint when the library is
missing, matching Equation 6's backward ODE

    dλ/dt = −(∂f_θ/∂h)^T λ

so that the counterfactual explainer (§3C) continues to produce
gradient signals on minimal installations.
"""
from __future__ import annotations

import torch

from ct_explain.utils.compat import HAS_TORCHDIFFEQ
from ct_explain.utils.logging import get_logger

log = get_logger(__name__)


def adjoint_backward(
    ode_func, h_final: torch.Tensor, loss_grad: torch.Tensor,
    t_span: torch.Tensor, steps: int = 16,
) -> torch.Tensor:
    """Integrate dλ/dt = −(∂f/∂h)ᵀ λ backwards from t_end to 0.

    Returns λ(0) which equals ∂L/∂h(0) for the Neural-ODE layer.
    """
    if HAS_TORCHDIFFEQ:
        # Prefer the library-integrated adjoint in production.
        from torchdiffeq import odeint_adjoint  # noqa: F401
        log.info("torchdiffeq available — use odeint_adjoint directly.")

    lam = loss_grad.clone()
    h = h_final.clone()
    t0, t1 = t_span[0], t_span[-1]
    dt = -(t1 - t0) / steps                 # backward step

    for k in range(steps):
        t = t1 + k * dt
        h.requires_grad_(True)
        f = ode_func(t, h)
        jvp = torch.autograd.grad(
            outputs=f, inputs=h, grad_outputs=lam,
            retain_graph=False, create_graph=False,
        )[0]
        lam = lam - dt * jvp
        h = h - dt * f.detach()
    return lam
