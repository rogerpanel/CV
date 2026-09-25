"""SODE-Guard: end-to-end model.

Pipeline:
    flow features (B, 83) → E-GraphSAGE encoder → h_0 ∈ ℝ¹²⁸
    h_0 → Euler–Maruyama integration of dX = f dt + g dW to T = 1 → X_T
    X_T → linear head ψ → logits ∈ ℝ^K

The mean predictor is F(x) = E[ψ(X_T) | X_0 = h(x)] and the class is
argmax F(x) (softmax of the mean logits, Eq. (6) of the paper). Deployment
estimates F with ``mc_paths_eval`` paths; certification uses many more paths
with fresh randomness (``src/certify``).
"""
from __future__ import annotations
import secrets
from dataclasses import dataclass
from typing import Optional, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm

from .egraphsage import EGraphSAGE
from .drift_diffusion import DriftNet, DiffusionNet
from ..sde.integrator import EulerMaruyama, EMConfig


@dataclass
class SODEGuardConfig:
    feature_dim: int = 83
    hidden_dim: int = 128
    num_classes: int = 34
    drift_hidden: int = 256
    diff_hidden: int = 256
    drift_layers: int = 3
    diff_layers: int = 3
    noise_dim: int = 16
    horizon: float = 1.0
    dt: float = 0.05
    ellipticity_floor: float = 1.0e-3
    spectral_norm: bool = True
    mc_paths_eval: int = 8
    encoder_layers: int = 3
    encoder_dropout: float = 0.10
    activation: str = "gelu"
    virtual_brownian: bool = True
    certifiable: bool = False   # Lipschitz encoder + spectral-normalised head


def fresh_seeds(n: int) -> list[int]:
    """Seeds drawn from the OS CSPRNG, so path samples are not predictable."""
    return [secrets.randbits(31) for _ in range(n)]


class SODEGuard(nn.Module):
    def __init__(self, cfg: Optional[SODEGuardConfig] = None):
        super().__init__()
        self.cfg = cfg or SODEGuardConfig()
        c = self.cfg

        self.encoder = EGraphSAGE(
            edge_features=c.feature_dim,
            hidden_dim=c.hidden_dim,
            num_layers=c.encoder_layers,
            dropout=c.encoder_dropout,
            lipschitz=c.certifiable,
        )
        self.drift = DriftNet(
            dim=c.hidden_dim, hidden=c.drift_hidden,
            num_layers=c.drift_layers, spectral=c.spectral_norm,
            activation=c.activation,
        )
        self.diffusion = DiffusionNet(
            dim=c.hidden_dim, hidden=c.diff_hidden,
            num_layers=c.diff_layers, noise_dim=c.noise_dim,
            spectral=c.spectral_norm, activation=c.activation,
        )
        head = nn.Linear(c.hidden_dim, c.num_classes)
        self.head = spectral_norm(head) if c.certifiable else head

        self._em = EulerMaruyama(EMConfig(
            t0=0.0, t1=c.horizon, dt=c.dt,
            noise_dim=c.noise_dim,
            ellipticity_floor=c.ellipticity_floor,
            use_virtual_brownian=c.virtual_brownian,
            save_trajectory=False,
        ))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def _integrate(self, h0: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
        return self._em(h0, self.drift, self.diffusion, seed=seed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Single-path logits (seed 0). Used inside training losses."""
        return self.head(self._integrate(self.encode(x), seed=0))

    def sample_logits(self, x: torch.Tensor, seeds: Sequence[int]) -> torch.Tensor:
        """Per-path logits, shape (B, N, K); differentiable in x and θ.

        Each seed drives one Brownian path per example (the virtual Brownian
        tree draws independent increments for every row of the batch).
        """
        h0 = self.encode(x)
        return torch.stack([self.head(self._integrate(h0, seed=s)) for s in seeds], dim=1)

    def forward_mean(self, x: torch.Tensor, n_paths: Optional[int] = None,
                     seeds: Optional[Sequence[int]] = None) -> torch.Tensor:
        """Monte-Carlo estimate of the mean logits F(x); differentiable.

        With ``seeds=None`` the paths use fresh CSPRNG seeds, which is what an
        expectation-over-transformation attacker should differentiate through.
        """
        if seeds is None:
            seeds = fresh_seeds(n_paths or self.cfg.mc_paths_eval)
        return self.sample_logits(x, seeds).mean(dim=1)

    @torch.no_grad()
    def forward_mc(self, x: torch.Tensor, n_paths: Optional[int] = None) -> torch.Tensor:
        """Deployed predictor: softmax of the mean logits over fixed seeds 0..N-1."""
        n = n_paths or self.cfg.mc_paths_eval
        return F.softmax(self.forward_mean(x, seeds=range(n)), dim=-1)

    def forward_with_paths(self, x: torch.Tensor, n_paths: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-path logits and terminal states for the anti-concentration regulariser."""
        h0 = self.encode(x)
        logits_list, states = [], []
        for s in range(n_paths):
            hT = self._integrate(h0, seed=s)
            states.append(hT)
            logits_list.append(self.head(hT))
        return torch.stack(logits_list, dim=1), torch.stack(states, dim=1)
