"""SODE-Guard: end-to-end model.

Pipeline:
    flow features (B, 83) → E-GraphSAGE encoder → h_0 ∈ ℝ¹²⁸
    h_0 → Euler–Maruyama integrate dX = f dt + g dW to T=1 → h_T (or path)
    h_T → linear classifier → logits ∈ ℝ^K
    For inference we average softmax over N_mc Monte-Carlo paths.

The model exposes hooks so that downstream code can:
    * train the AC regulariser jointly (``forward_with_paths``),
    * compute certified radii (``certified_score``), and
    * be plugged into ``torchattacks`` style PGD without modification
      (``forward(x)`` accepts the raw 83-dim vector and returns logits).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

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


class _SDEModule(nn.Module):
    """Wrap drift & diffusion for the optional torchsde adjoint path."""
    noise_type = "general"
    sde_type = "ito"

    def __init__(self, drift: DriftNet, diffusion: DiffusionNet):
        super().__init__()
        self.drift = drift
        self.diffusion = diffusion

    def f(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        return self.drift(x, t.view(-1, 1))

    def g(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        return self.diffusion(x, t.view(-1, 1))


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
        self.sde_module = _SDEModule(self.drift, self.diffusion)
        self.head = nn.Linear(c.hidden_dim, c.num_classes)

        self._em = EulerMaruyama(EMConfig(
            t0=0.0, t1=c.horizon, dt=c.dt,
            noise_dim=c.noise_dim,
            ellipticity_floor=c.ellipticity_floor,
            use_virtual_brownian=c.virtual_brownian,
            save_trajectory=False,
        ))

    # ---------------------------------------------------------------
    # Forward variants
    # ---------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def _integrate(self, h0: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
        return self._em(h0, self.drift, self.diffusion, seed=seed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Single-path forward (training and gradient-based attacks)."""
        h0 = self.encode(x)
        hT = self._integrate(h0)
        return self.head(hT)

    @torch.no_grad()
    def forward_mc(self, x: torch.Tensor, n_paths: Optional[int] = None) -> torch.Tensor:
        """Evaluation forward: average softmax across n_paths samples."""
        n = n_paths or self.cfg.mc_paths_eval
        h0 = self.encode(x)
        probs = 0
        for s in range(n):
            hT = self._integrate(h0, seed=s)
            probs = probs + F.softmax(self.head(hT), dim=-1)
        return probs / n

    def forward_with_paths(self, x: torch.Tensor, n_paths: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return logits stack and terminal-state stack for the AC regulariser.

        Used by ``regularizers.anti_concentration.AntiConcentrationLoss``.
        """
        h0 = self.encode(x)
        logits_list, states = [], []
        for s in range(n_paths):
            hT = self._integrate(h0, seed=s)
            states.append(hT)
            logits_list.append(self.head(hT))
        return torch.stack(logits_list, dim=1), torch.stack(states, dim=1)

    # ---------------------------------------------------------------
    # Smoothed prediction + certified score
    # ---------------------------------------------------------------

    @torch.no_grad()
    def certified_score(self, x: torch.Tensor, n_paths: int = 256,
                        margin_threshold: float = 0.05) -> torch.Tensor:
        """Compute the margin of the smoothed classifier.

        ``returns`` the gap between top-1 and top-2 averaged softmax. The
        anti-concentration certificate (``regularizers.anti_concentration``)
        converts this margin into a robust radius.
        """
        probs = self.forward_mc(x, n_paths=n_paths)
        top2, _ = torch.topk(probs, k=2, dim=-1)
        margin = top2[:, 0] - top2[:, 1]
        return margin
