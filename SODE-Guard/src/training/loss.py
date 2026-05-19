"""Composite loss L = L_CE + λ · L_AC used by SODE-Guard training."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..regularizers.anti_concentration import AntiConcentrationLoss


class CrossEntropyWithAC(nn.Module):
    def __init__(self, ac_weight: float = 0.10, chaos_degree: int = 4,
                 beta_grid: tuple[float, ...] = (0.01, 0.025, 0.05, 0.10),
                 label_smoothing: float = 0.0,
                 n_ac_paths: int = 4):
        super().__init__()
        self.lambda_ac = float(ac_weight)
        self.ac = AntiConcentrationLoss(chaos_degree=chaos_degree, beta_grid=beta_grid)
        self.label_smoothing = float(label_smoothing)
        self.n_ac_paths = int(n_ac_paths)

    def forward(self, model, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        logits_paths, _ = model.forward_with_paths(x, n_paths=self.n_ac_paths)  # (B, P, K)
        logits_mean = logits_paths.mean(dim=1)
        ce = F.cross_entropy(logits_mean, y, label_smoothing=self.label_smoothing)
        ac = self.ac(logits_paths) if self.lambda_ac > 0 else torch.tensor(0.0, device=x.device)
        total = ce + self.lambda_ac * ac
        return total, {"loss": float(total.item()), "ce": float(ce.item()),
                       "ac": float(ac.item() if isinstance(ac, torch.Tensor) else ac)}
