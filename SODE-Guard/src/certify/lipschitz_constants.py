"""Lipschitz constant L of Theorem A, computed from the trained weights.

For the Euler–Maruyama scheme with step Δt and horizon T, the synchronous
coupling of the paths started at h(x) and h(x+δ) gives (Lemma 1)

    E||X_T^{x+δ} − X_T^x||² ≤ L_h² ||δ||² exp((2 K_f + K_g² + K_f² Δt) T),

so the mean logits F(x) = E[P_B(ψ(X_T))] are L-Lipschitz with

    L = ||W_ψ||_2 · L_h · exp((K_f + K_g²/2 + K_f² Δt / 2) T).

K_f, K_g, L_h are products of exact per-layer spectral norms and activation
constants. We compute them with an exact SVD rather than trusting the
one-step power iteration used by spectral normalisation during training.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, asdict

import torch

from ..models.egraphsage import spectral_norm_exact


@dataclass
class LipschitzReport:
    L_h: float          # encoder
    K_f: float          # drift, in the state
    K_g: float          # diffusion, Frobenius norm of the (d, m) output
    head_norm: float    # ||W_ψ||_2
    dt: float
    T: float
    L_state: float      # sqrt of the mean-square growth factor, incl. L_h
    L: float            # Lipschitz constant of the mean logits F

    def as_dict(self) -> dict:
        return asdict(self)


@torch.no_grad()
def model_lipschitz(model) -> LipschitzReport:
    was_training = model.training
    model.eval()
    try:
        L_h = model.encoder.lipschitz_bound()
        K_f = model.drift.lipschitz_bound()
        K_g = model.diffusion.lipschitz_bound()
        head_norm = spectral_norm_exact(model.head.weight)
    finally:
        model.train(was_training)
    dt, T = model.cfg.dt, model.cfg.horizon
    L_state = L_h * math.exp((K_f + 0.5 * K_g ** 2 + 0.5 * K_f ** 2 * dt) * T)
    return LipschitzReport(L_h=L_h, K_f=K_f, K_g=K_g, head_norm=head_norm,
                           dt=dt, T=T, L_state=L_state, L=head_norm * L_state)
