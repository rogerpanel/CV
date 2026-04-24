"""Training objective.

The manuscript's detection head uses cross-entropy with class-imbalance
handling (focal loss). The joint explain-and-detect loss optionally adds
the Problem 1 multi-objective term:

    L = L_detect + λ_f · (−Faith(a)) + λ_c · Cost(E) + λ_w · Width(C_α(a))

Cost(E) and Width(C_α(a)) are treated as regulariser proxies during
training; the full evaluation of Faith / Cost / Width is done offline.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    alpha: float | None = None,
) -> torch.Tensor:
    ce = F.cross_entropy(logits, target, reduction="none")
    p_t = (-ce).exp()
    focal = ((1 - p_t) ** gamma) * ce
    if alpha is not None:
        at = torch.full_like(target, alpha, dtype=torch.float32)
        at = torch.where(target == 0, 1 - at, at)
        focal = at * focal
    return focal.mean()


@dataclass
class CTExplainLoss(nn.Module):
    use_focal: bool = True
    focal_gamma: float = 2.0
    lambda_faith: float = 0.0
    lambda_cost: float = 0.0
    lambda_width: float = 0.0

    def __post_init__(self) -> None:
        super().__init__()

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        *,
        faith: torch.Tensor | None = None,
        cost: torch.Tensor | None = None,
        width: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        detect = (
            focal_loss(logits, labels, gamma=self.focal_gamma)
            if self.use_focal
            else F.cross_entropy(logits, labels)
        )
        total = detect
        parts = {"detect": detect}
        if self.lambda_faith and faith is not None:
            parts["faith"] = -faith.mean() * self.lambda_faith
            total = total + parts["faith"]
        if self.lambda_cost and cost is not None:
            parts["cost"] = cost.mean() * self.lambda_cost
            total = total + parts["cost"]
        if self.lambda_width and width is not None:
            parts["width"] = width.mean() * self.lambda_width
            total = total + parts["width"]
        parts["total"] = total
        return parts
