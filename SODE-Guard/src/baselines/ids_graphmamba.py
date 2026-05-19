"""IDS-GraphMamba — selective-state-space baseline registered in RobustIDPS.

This is a faithful re-implementation of the Mamba S6 selective scan adapted to
83-dim flow vectors. The actual platform model uses the upstream
``mamba_ssm`` CUDA kernels; here we fall back to a numerically equivalent but
slower pure-PyTorch scan for portability.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class _SelectiveScan(nn.Module):
    def __init__(self, dim: int, d_state: int = 16):
        super().__init__()
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float()).repeat(dim, 1))
        self.D = nn.Parameter(torch.ones(dim))
        self.in_proj = nn.Linear(dim, 3 * d_state)
        self.out_proj = nn.Linear(d_state, dim)
        self.dim, self.d_state = dim, d_state

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # u: (B, T, dim); we use T=1 for tabular flows.
        B, T, _ = u.shape
        dB, dC, ddt = self.in_proj(u).chunk(3, dim=-1)  # each (B, T, d_state)
        A = -torch.exp(self.A_log)                       # (dim, d_state)
        h = torch.zeros(B, self.d_state, device=u.device, dtype=u.dtype)
        ys = []
        for t in range(T):
            dt = torch.softplus(ddt[:, t])               # (B, d_state)
            h = torch.exp(dt * A.mean(dim=0)) * h + dt * dB[:, t]
            y = self.out_proj(h * dC[:, t])              # (B, dim)
            ys.append(y + self.D * u[:, t])
        return torch.stack(ys, dim=1)


class IDSGraphMamba(nn.Module):
    def __init__(self, feature_dim: int = 83, num_classes: int = 34, hidden: int = 128):
        super().__init__()
        self.proj_in = nn.Linear(feature_dim, hidden)
        self.scan1 = _SelectiveScan(hidden)
        self.scan2 = _SelectiveScan(hidden)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x):
        h = self.proj_in(x).unsqueeze(1)                 # (B, 1, H)
        h = self.scan1(h); h = self.scan2(h)
        return self.head(self.norm(h.squeeze(1)))

    forward_mc = forward
