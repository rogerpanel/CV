"""Drift f_θ and diffusion g_θ networks used by SODE-Guard.

Both are 3-layer MLPs with GELU activations and spectral normalisation on each
linear layer. The diffusion network outputs a (d, m) matrix and is paired at
integration time with an ellipticity floor λ_0 (see ``sde/integrator.py``).

``lipschitz_bound()`` returns an upper bound on the Lipschitz constant in the
state x (the time input is excluded). For the diffusion the bound is in the
Frobenius norm of the (d, m) output, which is what the Itô isometry needs.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm

from .egraphsage import GELU_LIPSCHITZ, spectral_norm_exact

_ACT_LIPSCHITZ = {"gelu": GELU_LIPSCHITZ, "silu": 1.0998, "relu": 1.0}


def _linear(in_f: int, out_f: int, sn: bool) -> nn.Linear:
    layer = nn.Linear(in_f, out_f)
    return spectral_norm(layer) if sn else layer


def _time_concat(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    if t.ndim == x.ndim - 1:
        t = t.unsqueeze(-1)
    if t.shape[0] == 1:
        t = t.expand(x.shape[0], -1)
    return torch.cat([x, t], dim=-1)


class _TimeMLP(nn.Module):
    def __init__(self, dim: int, out_dim: int, hidden: int, num_layers: int,
                 spectral: bool, activation: str):
        super().__init__()
        self.dim = dim
        self.activation = activation
        act = {"gelu": nn.GELU, "silu": nn.SiLU, "relu": nn.ReLU}[activation]
        layers: list[nn.Module] = [_linear(dim + 1, hidden, spectral), act()]
        for _ in range(num_layers - 2):
            layers += [_linear(hidden, hidden, spectral), act()]
        layers += [_linear(hidden, out_dim, spectral)]
        self.net = nn.Sequential(*layers)

    def lipschitz_bound(self) -> float:
        linears = [m for m in self.net if isinstance(m, nn.Linear)]
        bound = spectral_norm_exact(linears[0].weight[:, : self.dim])
        for lin in linears[1:]:
            bound *= _ACT_LIPSCHITZ[self.activation] * spectral_norm_exact(lin.weight)
        return bound


class DriftNet(_TimeMLP):
    """f_θ : (x, t) → ℝ^d."""

    def __init__(self, dim: int = 128, hidden: int = 256, num_layers: int = 3,
                 spectral: bool = True, activation: str = "gelu"):
        super().__init__(dim, dim, hidden, num_layers, spectral, activation)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(_time_concat(x, t))


class DiffusionNet(_TimeMLP):
    """g_θ : (x, t) → ℝ^{d×m}."""

    def __init__(self, dim: int = 128, hidden: int = 256, num_layers: int = 3,
                 noise_dim: int = 16, spectral: bool = True, activation: str = "gelu"):
        super().__init__(dim, dim * noise_dim, hidden, num_layers, spectral, activation)
        self.m = noise_dim

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        flat = self.net(_time_concat(x, t))
        return flat.view(x.shape[0], self.dim, self.m)
