"""
Temporal Adaptive Batch Normalization Neural ODE (TA-BN-ODE)
============================================================
Implements Section IV of the paper:
  - TA-BN layer with time-dependent γ(t), β(t), μ(t), σ²(t)  (Eq. 6–8)
  - ODE vector field with ELU activation                       (Eq. 5)
  - Multi-scale parallel ODE branches with learned time        (Eq. 9)
    constants {10⁻⁶, 10⁻³, 1, 3600} s
  - Adjoint-method training via torchdiffeq
  - Stability regularisation (Theorem 1)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint, odeint
from typing import Optional, List, Tuple


# ---------------------------------------------------------------------------
# Temporal Adaptive Batch Normalization  (Eq. 6-8)
# ---------------------------------------------------------------------------
class TABNLayer(nn.Module):
    """Time-dependent batch normalisation (TA-BN).

    Parameters are functions of integration time *t* via small MLPs with
    sinusoidal time embeddings, following Salvi et al. (NeurIPS 2024).
    """

    def __init__(self, dim: int, mlp_hidden: int = 64, omega: float = 1.0,
                 eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.omega = omega

        # Time embedding dim: [t, sin(ωt), cos(ωt)]
        time_dim = 3

        # MLPs for γ(t) and β(t)  — two hidden layers of 64 units each
        self.gamma_net = nn.Sequential(
            nn.Linear(time_dim, mlp_hidden), nn.ELU(),
            nn.Linear(mlp_hidden, mlp_hidden), nn.ELU(),
            nn.Linear(mlp_hidden, dim),
        )
        self.beta_net = nn.Sequential(
            nn.Linear(time_dim, mlp_hidden), nn.ELU(),
            nn.Linear(mlp_hidden, mlp_hidden), nn.ELU(),
            nn.Linear(mlp_hidden, dim),
        )

        # Running statistics (EMA)
        self.register_buffer("running_mean", torch.zeros(dim))
        self.register_buffer("running_var", torch.ones(dim))

        # Initialise γ → 1, β → 0
        nn.init.constant_(self.gamma_net[-1].weight, 0.0)
        nn.init.constant_(self.gamma_net[-1].bias, 1.0)
        nn.init.constant_(self.beta_net[-1].weight, 0.0)
        nn.init.constant_(self.beta_net[-1].bias, 0.0)

    def _time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """[t, sin(ωt), cos(ωt)]"""
        t_scalar = t.float().reshape(1)
        return torch.stack([
            t_scalar,
            torch.sin(self.omega * t_scalar),
            torch.cos(self.omega * t_scalar),
        ], dim=-1)  # (1, 3)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, dim)
            t: scalar integration time
        """
        t_emb = self._time_embedding(t).to(x.device)

        gamma = self.gamma_net(t_emb).squeeze(0)  # (dim,)
        beta = self.beta_net(t_emb).squeeze(0)

        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean)
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * var)
        else:
            mean = self.running_mean
            var = self.running_var

        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return gamma * x_norm + beta

    def regularisation_loss(self, t_span: torch.Tensor) -> torch.Tensor:
        """∫₀ᵀ ‖γ(t)‖² + ‖β(t)‖² dt  (Theorem 1 stability reg.)"""
        loss = torch.tensor(0.0, device=t_span.device)
        for t in t_span:
            t_emb = self._time_embedding(t).to(t_span.device)
            gamma = self.gamma_net(t_emb).squeeze(0)
            beta = self.beta_net(t_emb).squeeze(0)
            loss = loss + gamma.pow(2).sum() + beta.pow(2).sum()
        return loss / len(t_span)


# ---------------------------------------------------------------------------
# ODE vector field  (Eq. 5)
# ---------------------------------------------------------------------------
class ODEFunc(nn.Module):
    """f_θ(h, t) = ELU(TA-BN(W₂ · ELU(TA-BN(W₁ h, t)), t))"""

    def __init__(self, hidden_dim: int, mlp_hidden: int = 64):
        super().__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.tabn1 = TABNLayer(hidden_dim, mlp_hidden=mlp_hidden)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.tabn2 = TABNLayer(hidden_dim, mlp_hidden=mlp_hidden)

        # NFE counter for diagnostics
        self.nfe = 0

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        self.nfe += 1
        out = self.W1(h)
        out = self.tabn1(out, t)
        out = F.elu(out)
        out = self.W2(out)
        out = self.tabn2(out, t)
        out = F.elu(out)
        return out

    def regularisation_loss(self, t_span: torch.Tensor) -> torch.Tensor:
        return (self.tabn1.regularisation_loss(t_span)
                + self.tabn2.regularisation_loss(t_span))


# ---------------------------------------------------------------------------
# Single TA-BN-ODE Block (Algorithm 1)
# ---------------------------------------------------------------------------
class TABNODEBlock(nn.Module):
    """One continuous-depth block: ODESolve(f_θ, h₀, [t₀, t₁])."""

    def __init__(self, hidden_dim: int, solver: str = "dopri5",
                 rtol: float = 1e-3, atol: float = 1e-4,
                 use_adjoint: bool = True):
        super().__init__()
        self.ode_func = ODEFunc(hidden_dim)
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self._integrator = odeint_adjoint if use_adjoint else odeint

    def forward(self, h0: torch.Tensor,
                t_span: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h0: (batch, hidden_dim)
            t_span: 1-D time grid, e.g. linspace(0, 1, 10)
        Returns:
            h_T: (batch, hidden_dim) at final time
        """
        self.ode_func.nfe = 0
        # h_traj shape: (len(t_span), batch, hidden_dim)
        h_traj = self._integrator(
            self.ode_func, h0, t_span,
            method=self.solver, rtol=self.rtol, atol=self.atol,
        )
        return h_traj[-1]  # final state

    def trajectory(self, h0: torch.Tensor,
                   t_span: torch.Tensor) -> torch.Tensor:
        """Return full trajectory for uncertainty sampling."""
        self.ode_func.nfe = 0
        return self._integrator(
            self.ode_func, h0, t_span,
            method=self.solver, rtol=self.rtol, atol=self.atol,
        )

    @property
    def nfe(self) -> int:
        return self.ode_func.nfe

    def regularisation_loss(self, t_span: torch.Tensor) -> torch.Tensor:
        return self.ode_func.regularisation_loss(t_span)


# ---------------------------------------------------------------------------
# Multi-Scale TA-BN-ODE  (Eq. 9)
# ---------------------------------------------------------------------------
class MultiScaleTABNODE(nn.Module):
    """Parallel ODE branches with learned time constants τ_s.

    dh/dt = Σ_s α_s · f_{θ_s}(t / τ_s)

    Default time constants span 8 orders of magnitude:
        {10⁻⁶, 10⁻³, 1, 3600} seconds
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256,
                 n_ode_blocks: int = 2,
                 time_constants: Optional[List[float]] = None,
                 solver: str = "dopri5",
                 rtol: float = 1e-3, atol: float = 1e-4,
                 use_adjoint: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim

        if time_constants is None:
            time_constants = [1e-6, 1e-3, 1.0, 3600.0]
        self.register_buffer(
            "time_constants",
            torch.tensor(time_constants, dtype=torch.float32),
        )
        n_scales = len(time_constants)

        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Scale-specific ODE blocks (stacked)
        self.scale_blocks = nn.ModuleList()
        for _ in range(n_scales):
            blocks = nn.ModuleList([
                TABNODEBlock(hidden_dim, solver, rtol, atol, use_adjoint)
                for _ in range(n_ode_blocks)
            ])
            self.scale_blocks.append(blocks)

        # Learned attention weights α_s
        self.scale_attention = nn.Parameter(torch.ones(n_scales) / n_scales)

        # Event-driven state update  u_ψ(x_i, k_i)
        self.event_update = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor,
                t_span: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, input_dim) event features
            t_span: 1-D time grid
        Returns:
            h_final: (batch, hidden_dim) evolved state
            h0: (batch, hidden_dim) initial encoding (for coupling)
        """
        h0 = self.encoder(x)
        alpha = F.softmax(self.scale_attention, dim=0)

        h_combined = torch.zeros_like(h0)
        for s, (tau, blocks) in enumerate(
                zip(self.time_constants, self.scale_blocks)):
            # Rescale time grid by time constant
            t_scaled = t_span / tau
            h_s = h0
            for block in blocks:
                h_s = block(h_s, t_scaled)
            h_combined = h_combined + alpha[s] * h_s

        return h_combined, h0

    def forward_with_events(
            self, x_seq: torch.Tensor, t_events: torch.Tensor
    ) -> torch.Tensor:
        """Process event sequence with inter-event ODE integration.

        Args:
            x_seq: (seq_len, batch, input_dim) features at each event
            t_events: (seq_len,) timestamps
        Returns:
            h_states: (seq_len, batch, hidden_dim)
        """
        batch = x_seq.shape[1]
        h = self.encoder(x_seq[0])
        states = [h]

        for i in range(1, len(t_events)):
            # Integrate between events
            dt_grid = torch.linspace(0, 1, 5, device=h.device)
            dt_grid = dt_grid * (t_events[i] - t_events[i - 1])
            alpha = F.softmax(self.scale_attention, dim=0)
            h_new = torch.zeros_like(h)
            for s, (tau, blocks) in enumerate(
                    zip(self.time_constants, self.scale_blocks)):
                h_s = h
                for block in blocks:
                    h_s = block(h_s, dt_grid / tau)
                h_new = h_new + alpha[s] * h_s
            # Event-driven update: h(t_i) = h(t_i⁻) + u_ψ(x_i)
            h = h_new + self.event_update(x_seq[i])
            states.append(h)

        return torch.stack(states, dim=0)

    def regularisation_loss(self, t_span: torch.Tensor) -> torch.Tensor:
        """TA-BN regularisation across all scales and blocks."""
        loss = torch.tensor(0.0, device=t_span.device)
        for blocks in self.scale_blocks:
            for block in blocks:
                loss = loss + block.regularisation_loss(t_span)
        return loss

    @property
    def nfe_total(self) -> int:
        return sum(
            block.nfe
            for blocks in self.scale_blocks
            for block in blocks
        )
