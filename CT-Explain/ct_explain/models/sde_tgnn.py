"""SDE-TGNN (Equation 3).

    dh_v(t) = f_θ(...) dt + σ_ψ(h_v(t)) dW_t

The diffusion coefficient σ_ψ is implemented as a state-dependent
heteroscedastic head. Integration uses `torchsde.sdeint` (Stratonovich
or Itô form) when available; otherwise we fall back to Euler–Maruyama so
that downstream components (`UncertaintyAttributionExplainer`) can still
call `.sample()` for Monte-Carlo variance estimation.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ct_explain.models.ct_tgnn import CTTemporalGNN, CTTGNNOutput
from ct_explain.models.ode_func import GraphSnapshot
from ct_explain.utils.compat import HAS_TORCHSDE
from ct_explain.utils.logging import get_logger

log = get_logger(__name__)


class DiffusionHead(nn.Module):
    """σ_ψ(h) : ℝ^{d_h} → ℝ^{d_h} (diagonal state-dependent noise scale)."""

    def __init__(self, hidden_dim: int, floor: float = 1e-3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
        )
        self.floor = floor

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h) + self.floor


class SDETemporalGNN(CTTemporalGNN):
    """CT-TGNN with an SDE-style diffusion term for uncertainty modelling."""

    def __init__(self, *args, diffusion_scale: float = 0.1, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.diffusion = DiffusionHead(self.hidden_dim)
        self.diffusion_scale = diffusion_scale
        # torchsde convention hooks
        self.noise_type = "diagonal"
        self.sde_type = "ito"

    # torchsde.sdeint drift / diffusion interface ------------------- #
    def f(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:   # drift
        return self.ode_func(t, y)

    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:   # diffusion
        return self.diffusion_scale * self.diffusion(y)

    # ------------------------------------------------------------------ #
    # Monte-Carlo sampling of terminal hidden state
    # ------------------------------------------------------------------ #
    def sample(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        edge_times: torch.Tensor,
        num_samples: int = 32,
        query_time: Optional[float] = None,
    ) -> torch.Tensor:
        """Draw K Monte-Carlo terminal states h(T) from the SDE.

        Returns tensor of shape (num_samples, N, d_h).
        """
        t_end = float(query_time if query_time is not None else self.integration_window)
        num_nodes = node_features.size(0)
        h0 = self.input_proj(node_features)

        snapshot = GraphSnapshot(
            edge_index=edge_index,
            edge_features=edge_features,
            delta_t=t_end - edge_times,
            num_nodes=num_nodes,
        )
        self.ode_func.bind(snapshot)
        t_span = torch.tensor([0.0, t_end], device=h0.device, dtype=h0.dtype)

        samples = []
        if HAS_TORCHSDE:
            import torchsde
            for _ in range(num_samples):
                traj = torchsde.sdeint(
                    self, h0, t_span, method="srk",
                    dt=max((t_span[-1] - t_span[0]).item() / 32, 1e-3),
                )
                samples.append(traj[-1])
        else:
            log.warning("torchsde unavailable — Euler–Maruyama fallback.")
            samples = [self._euler_maruyama(h0, t_span) for _ in range(num_samples)]

        return torch.stack(samples)

    # ------------------------------------------------------------------ #
    # Euler–Maruyama fallback
    # ------------------------------------------------------------------ #
    def _euler_maruyama(
        self,
        h0: torch.Tensor,
        t_span: torch.Tensor,
        steps: int = 32,
    ) -> torch.Tensor:
        t0, t1 = t_span[0], t_span[-1]
        dt = (t1 - t0) / steps
        sqrt_dt = dt.abs().sqrt()
        h = h0
        for k in range(steps):
            t = t0 + k * dt
            drift = self.f(t, h)
            diff = self.g(t, h)
            noise = torch.randn_like(h)
            h = h + drift * dt + diff * noise * sqrt_dt
        return h

    # ------------------------------------------------------------------ #
    # MC-wrapped predict with variance
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        edge_times: torch.Tensor,
        num_samples: int = 32,
    ) -> dict:
        self.eval()
        h_samples = self.sample(node_features, edge_index, edge_features, edge_times,
                                num_samples=num_samples)
        logits = self.classifier(h_samples)                   # (K, N, C)
        probs = torch.softmax(logits, dim=-1)
        mean = probs.mean(0)
        var = probs.var(0)
        return {
            "mean": mean,
            "variance": var,
            "hidden_mean": h_samples.mean(0),
            "hidden_var": h_samples.var(0),
            "safety_prob": mean[:, 0],
            "prediction": mean.argmax(-1),
        }
