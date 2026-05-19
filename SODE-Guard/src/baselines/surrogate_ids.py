"""SurrogateIDS-7B: 7-branch ensemble used as the deployment teacher
in RobustIDPS v3 (97.8 % clean accuracy, 0.976 F1). Each branch is a small
MLP specialised on a feature group from ``feature_engineering.FEATURE_GROUPS``;
the gate mixes them with a softmax over per-branch confidences.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from ..data.feature_engineering import FEATURE_GROUPS, ALL_FEATURES


def _group_indices() -> dict[str, list[int]]:
    pos = {name: i for i, name in enumerate(ALL_FEATURES)}
    return {grp: [pos[f] for f in feats] for grp, feats in FEATURE_GROUPS.items()}


class _Branch(nn.Module):
    def __init__(self, in_dim: int, hidden: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class SurrogateIDS7B(nn.Module):
    def __init__(self, num_classes: int = 34, hidden: int = 256):
        super().__init__()
        self.group_idx = _group_indices()
        self.branches = nn.ModuleDict({
            grp: _Branch(len(idx), hidden, num_classes) for grp, idx in self.group_idx.items()
        })
        self.gate = nn.Linear(len(self.group_idx) * num_classes, len(self.group_idx))

    def forward(self, x):
        outs = []
        for grp, idx in self.group_idx.items():
            outs.append(self.branches[grp](x[:, idx]))
        stacked = torch.stack(outs, dim=1)               # (B, 7, K)
        gate_in = stacked.flatten(1)
        w = torch.softmax(self.gate(gate_in), dim=-1)    # (B, 7)
        return (stacked * w.unsqueeze(-1)).sum(dim=1)

    forward_mc = forward
