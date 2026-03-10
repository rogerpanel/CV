"""
Temporal Adaptive Batch Normalization Neural ODE (TA-BN-ODE).

Implements Equations 1, 4, 5, 7 from the main manuscript.
- Eq 1: dh(t)/dt = f_theta(h(t), t)
- Eq 4: Two-layer ODE block with TA-BN and ELU
- Eq 5: Time-dependent gamma(t), beta(t) via MLPs on [t, sin(wt), cos(wt)]
- Eq 7: Multi-scale parallel branches with learned time constants
"""

import torch
import torch.nn as nn
import math
from torchdiffeq import odeint_adjoint


class TemporalAdaptiveBatchNorm(nn.Module):
    """TA-BN: Batch normalization with time-dependent affine parameters (Eq. 5).

    gamma(t) and beta(t) are output by small MLPs that receive
    [t, sin(omega * t), cos(omega * t)] as input, capturing diurnal
    traffic cycles.
    """

    def __init__(self, num_features: int, mlp_hidden: int = 64,
                 mlp_layers: int = 2, eps: float = 1e-5,
                 momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Running statistics for batch norm
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

        # Temporal input: [t, sin(wt), cos(wt)] -> dim 3
        # Learnable angular frequency
        self.omega = nn.Parameter(torch.tensor(2.0 * math.pi / 86400.0))

        # MLP for gamma(t)
        gamma_layers = [nn.Linear(3, mlp_hidden), nn.ReLU()]
        for _ in range(mlp_layers - 1):
            gamma_layers += [nn.Linear(mlp_hidden, mlp_hidden), nn.ReLU()]
        gamma_layers.append(nn.Linear(mlp_hidden, num_features))
        self.gamma_net = nn.Sequential(*gamma_layers)
        # Initialize near identity: gamma ~ 1
        nn.init.zeros_(self.gamma_net[-1].weight)
        nn.init.ones_(self.gamma_net[-1].bias)

        # MLP for beta(t)
        beta_layers = [nn.Linear(3, mlp_hidden), nn.ReLU()]
        for _ in range(mlp_layers - 1):
            beta_layers += [nn.Linear(mlp_hidden, mlp_hidden), nn.ReLU()]
        beta_layers.append(nn.Linear(mlp_hidden, num_features))
        self.beta_net = nn.Sequential(*beta_layers)
        # Initialize near zero
        nn.init.zeros_(self.beta_net[-1].weight)
        nn.init.zeros_(self.beta_net[-1].bias)

    def _temporal_input(self, t: torch.Tensor) -> torch.Tensor:
        """Build [t, sin(omega*t), cos(omega*t)]."""
        t_scalar = t.float().reshape(1)
        wt = self.omega * t_scalar
        return torch.cat([t_scalar, torch.sin(wt), torch.cos(wt)]).unsqueeze(0)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, num_features]
            t: scalar time
        """
        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.detach()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.detach()
        else:
            mean = self.running_mean
            var = self.running_var

        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # Time-dependent affine
        t_input = self._temporal_input(t)  # [1, 3]
        gamma = self.gamma_net(t_input)  # [1, num_features]
        beta = self.beta_net(t_input)    # [1, num_features]

        return gamma * x_norm + beta


class SingleScaleODEFunc(nn.Module):
    """ODE dynamics for one time-scale branch (Eq. 4).

    dh/dt = ELU(TA-BN(W2 * ELU(TA-BN(W1 * h, t)), t))
    """

    def __init__(self, hidden_dim: int, tau: float,
                 mlp_hidden: int = 64, mlp_layers: int = 2):
        super().__init__()
        self.tau = tau

        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.tabn1 = TemporalAdaptiveBatchNorm(hidden_dim, mlp_hidden, mlp_layers)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.tabn2 = TemporalAdaptiveBatchNorm(hidden_dim, mlp_hidden, mlp_layers)
        self.act = nn.ELU()

    def forward(self, h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_scaled = t / self.tau
        out = self.W1(h)
        out = self.tabn1(out, t_scaled)
        out = self.act(out)
        out = self.W2(out)
        out = self.tabn2(out, t_scaled)
        out = self.act(out)
        return out


class MultiScaleODEFunc(nn.Module):
    """Multi-scale ODE dynamics (Eq. 7).

    dh/dt = sum_s alpha_s * f_{theta_s}(h, t / tau_s)

    Time constants span eight orders of magnitude:
    {1e-6, 1e-3, 1, 3600} seconds (microseconds to hours).
    """

    def __init__(self, hidden_dim: int,
                 time_constants: tuple = (1e-6, 1e-3, 1.0, 3600.0),
                 mlp_hidden: int = 64, mlp_layers: int = 2):
        super().__init__()
        self.n_scales = len(time_constants)

        self.branches = nn.ModuleList([
            SingleScaleODEFunc(hidden_dim, tau, mlp_hidden, mlp_layers)
            for tau in time_constants
        ])

        # Learnable mixing weights (softmax-normalized)
        self.alpha_logits = nn.Parameter(torch.zeros(self.n_scales))

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Note: torchdiffeq passes (t, h) — t first."""
        alphas = torch.softmax(self.alpha_logits, dim=0)
        dh = torch.zeros_like(h)
        for s, branch in enumerate(self.branches):
            dh = dh + alphas[s] * branch(h, t)
        return dh


class TABNODEBlock(nn.Module):
    """Single TA-BN-ODE integration block (Algorithm 1 step).

    Integrates the multi-scale ODE from t_{i-1} to t_i using the adjoint
    method (Dormand-Prince RK4-5) for O(1) memory gradients.
    """

    def __init__(self, hidden_dim: int,
                 time_constants: tuple = (1e-6, 1e-3, 1.0, 3600.0),
                 mlp_hidden: int = 64, mlp_layers: int = 2,
                 solver_method: str = "dopri5",
                 rtol: float = 1e-3, atol: float = 1e-4):
        super().__init__()
        self.ode_func = MultiScaleODEFunc(
            hidden_dim, time_constants, mlp_hidden, mlp_layers
        )
        self.method = solver_method
        self.rtol = rtol
        self.atol = atol

    def forward(self, h0: torch.Tensor,
                t_span: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h0: Initial hidden state [batch, hidden_dim]
            t_span: Integration time points [n_steps]
        Returns:
            h_t: Hidden states at all time points [n_steps, batch, hidden_dim]
        """
        h_t = odeint_adjoint(
            self.ode_func, h0, t_span,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
        )
        return h_t


class Encoder(nn.Module):
    """Feature encoder: maps raw input to ODE initial condition h(t_0)."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EventUpdate(nn.Module):
    """Event-driven state update u_psi (Algorithm 1, event-driven update step).

    h(t_i) = h(t_i^-) + u_psi(x_i)
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
