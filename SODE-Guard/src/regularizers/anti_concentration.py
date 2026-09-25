"""Anti-concentration regulariser L_AC (Eq. (7) of the paper).

A training-time surrogate that penalises the fraction of Brownian paths whose
top-1 minus top-2 logit margin falls inside a β-band around zero. This is the
near-boundary mass P[G(x) ≤ β] that appears as the first term of the
path-wise decision-flip bound (Proposition B, ``src/certify/path_flip.py``).
The (β / ||m||)^{1/d*} weighting follows the Carbery–Wright form; it is a
heuristic weighting only and no certificate is derived from it.

    L_AC = (1/|β-grid|) Σ_β log(1 + C·d*·(β/||m||_2)^{1/d*} · mean_i σ_κ(β − |m_i|))
"""
from __future__ import annotations
import torch
import torch.nn as nn

_CW_CONSTANT = 1.0


class AntiConcentrationLoss(nn.Module):
    def __init__(self, chaos_degree: int = 4,
                 beta_grid: tuple[float, ...] = (0.01, 0.025, 0.05, 0.10),
                 sharpness: float = 50.0):
        super().__init__()
        self.d_star = int(chaos_degree)
        self.beta_grid = tuple(float(b) for b in beta_grid)
        self.sharpness = float(sharpness)

    @staticmethod
    def _margin(logits: torch.Tensor) -> torch.Tensor:
        top2, _ = torch.topk(logits, k=2, dim=-1)
        return top2[..., 0] - top2[..., 1]

    def forward(self, logits_paths: torch.Tensor) -> torch.Tensor:
        """logits_paths: (B, N_paths, K). Returns a scalar loss."""
        m = self._margin(logits_paths.mean(dim=1)).abs()
        norm = m.detach().norm() + 1e-6
        loss = 0.0
        for beta in self.beta_grid:
            soft_indicator = torch.sigmoid(self.sharpness * (beta - m))
            ratio = (beta / norm).clamp_min(1e-12) ** (1.0 / self.d_star)
            loss = loss + torch.log1p(_CW_CONSTANT * self.d_star * ratio * soft_indicator.mean())
        return loss / len(self.beta_grid)
