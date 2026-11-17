"""CT-TGNN: Continuous-Time Temporal Graph Neural Network.

Implements the detection backbone of Equation 1 with adaptive Dormand–Prince
integration and optional adjoint-sensitivity backpropagation. Produces
per-node hidden trajectories plus graph-level and node-level classification
heads (safety probability p_safe(v) consumed by ConformalGuard).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ct_explain.models.ode_func import GraphSnapshot, NeuralODEFunc
from ct_explain.utils.compat import HAS_TORCHDIFFEQ
from ct_explain.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class CTTGNNOutput:
    hidden: torch.Tensor                 # (N, d_h) at T
    trajectory: Optional[torch.Tensor]   # (S, N, d_h) at each solver step
    attention: Optional[torch.Tensor]    # (H, E) last computed attn
    logits: Optional[torch.Tensor] = None        # (N, num_classes)
    safety_prob: Optional[torch.Tensor] = None   # (N,)
    extras: dict = field(default_factory=dict)


class CTTemporalGNN(nn.Module):
    """Continuous-time detection backbone.

    Parameters
    ----------
    node_feat_dim
        Dimension of x_v (packet / flow descriptor).
    edge_feat_dim
        Dimension of e_{vu}.
    hidden_dim
        ODE state size d_h.
    num_classes
        Number of output classes. The *first* class is reserved for "safe" /
        "benign" — ConformalGuard consumes softmax[:, 0] as p_safe.
    """

    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        hidden_dim: int = 128,
        num_classes: int = 2,
        num_heads: int = 4,
        time_dim: int = 32,
        dropout: float = 0.1,
        ode_solver: str = "dopri5",
        rtol: float = 1e-3,
        atol: float = 1e-4,
        adjoint: bool = True,
        integration_window: float = 1.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.integration_window = integration_window
        self.ode_solver = ode_solver
        self.rtol = rtol
        self.atol = atol
        self.adjoint = adjoint

        self.input_proj = nn.Linear(node_feat_dim, hidden_dim)
        self.ode_func = NeuralODEFunc(
            hidden_dim=hidden_dim,
            edge_dim=edge_feat_dim,
            num_heads=num_heads,
            time_dim=time_dim,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        node_features: torch.Tensor,          # (N, node_feat_dim)
        edge_index: torch.Tensor,             # (2, E)
        edge_features: torch.Tensor,          # (E, edge_feat_dim)
        edge_times: torch.Tensor,             # (E,) arrival times
        query_time: Optional[float] = None,
        return_trajectory: bool = False,
    ) -> CTTGNNOutput:
        num_nodes = node_features.size(0)
        h0 = self.input_proj(node_features)
        t_end = float(query_time if query_time is not None else self.integration_window)
        delta_t = t_end - edge_times                                 # (E,)

        snapshot = GraphSnapshot(
            edge_index=edge_index,
            edge_features=edge_features,
            delta_t=delta_t,
            num_nodes=num_nodes,
        )
        self.ode_func.bind(snapshot)

        t_span = torch.tensor([0.0, t_end], device=h0.device, dtype=h0.dtype)
        h_final, traj = self._integrate(h0, t_span, return_trajectory)

        logits = self.classifier(h_final)
        safe = F.softmax(logits, dim=-1)[:, 0]
        return CTTGNNOutput(
            hidden=h_final,
            trajectory=traj,
            attention=self.ode_func.last_attention,
            logits=logits,
            safety_prob=safe,
        )

    # ------------------------------------------------------------------ #
    # ODE integration (adjoint if available, fallback Euler)
    # ------------------------------------------------------------------ #
    def _integrate(
        self,
        h0: torch.Tensor,
        t_span: torch.Tensor,
        return_trajectory: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if HAS_TORCHDIFFEQ:
            from torchdiffeq import odeint, odeint_adjoint

            solver = odeint_adjoint if (self.adjoint and self.training) else odeint
            traj = solver(
                self.ode_func,
                h0,
                t_span,
                method=self.ode_solver,
                rtol=self.rtol,
                atol=self.atol,
            )
            h_final = traj[-1]
            return h_final, traj if return_trajectory else None

        log.warning("torchdiffeq unavailable — falling back to fixed-step Euler.")
        return self._euler(h0, t_span, return_trajectory)

    def _euler(
        self,
        h0: torch.Tensor,
        t_span: torch.Tensor,
        return_trajectory: bool,
        steps: int = 16,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        t0, t1 = t_span[0], t_span[-1]
        dt = (t1 - t0) / steps
        h = h0
        traj = [h0] if return_trajectory else None
        for k in range(steps):
            t = t0 + k * dt
            h = h + dt * self.ode_func(t, h)
            if return_trajectory:
                traj.append(h)
        return h, torch.stack(traj) if return_trajectory else None

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        edge_times: torch.Tensor,
    ) -> dict:
        self.eval()
        out = self.forward(node_features, edge_index, edge_features, edge_times)
        probs = F.softmax(out.logits, dim=-1)
        return {
            "probabilities": probs,
            "safety_prob": out.safety_prob,
            "prediction": probs.argmax(-1),
            "hidden": out.hidden,
            "attention": out.attention,
        }
