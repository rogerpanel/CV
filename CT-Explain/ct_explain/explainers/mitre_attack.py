"""MITRE ATT&CK kill-chain alignment.

The attention-flow explainer maps each solver step's attention field to a
kill-chain stage using a lightweight classifier. We implement it as a
thin logistic head over aggregated attention statistics — small, fast,
and deterministic, matching the paper's "lightweight phase classifier"
description. The tactic palette and colour map follow §3A.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

# 14 canonical ATT&CK tactics (MITRE v15+); colour map follows §3A.
KILL_CHAIN_STAGES: list[str] = [
    "reconnaissance",
    "resource-development",
    "initial-access",
    "execution",
    "persistence",
    "privilege-escalation",
    "defense-evasion",
    "credential-access",
    "discovery",
    "lateral-movement",
    "collection",
    "command-and-control",
    "exfiltration",
    "impact",
]

STAGE_COLORS: dict[str, str] = {
    "reconnaissance":         "#00B4D8",  # cyan
    "resource-development":   "#48CAE4",
    "initial-access":         "#0077B6",  # blue
    "execution":              "#023E8A",
    "persistence":            "#7209B7",
    "privilege-escalation":   "#B5179E",
    "defense-evasion":        "#F72585",
    "credential-access":      "#9D4EDD",
    "discovery":              "#06D6A0",
    "lateral-movement":       "#F4A261",  # orange
    "collection":             "#E9C46A",
    "command-and-control":    "#E76F51",
    "exfiltration":           "#D62828",  # red
    "impact":                 "#8B0000",
}


@dataclass
class KillChainAlignment:
    stage: str
    color: str
    confidence: float
    logits: torch.Tensor


class MITREAttackMapper(nn.Module):
    """Logistic regression head over attention-flow summary statistics.

    Input features (per time step):
        - attention entropy H[α]
        - attention mass concentrated on the top-k fraction
        - mean attention weight
        - max attention weight
        - edge count

    Output: distribution over 14 ATT&CK stages.
    """

    SUMMARY_DIM = 5

    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(self.SUMMARY_DIM, len(KILL_CHAIN_STAGES))

    # -------------------------------------------------------------- #
    def summarize(self, attention: torch.Tensor) -> torch.Tensor:
        """attention: (H, E) or (E,) → summary tensor (SUMMARY_DIM,)."""
        if attention.dim() == 2:
            alpha = attention.mean(0)                      # average over heads
        else:
            alpha = attention.reshape(-1)

        if alpha.numel() == 0:
            return torch.zeros(self.SUMMARY_DIM, device=alpha.device)

        p = alpha / (alpha.sum() + 1e-12)
        entropy = -(p * (p + 1e-12).log()).sum()
        top_k = max(1, int(0.05 * p.numel()))
        top_mass = p.topk(top_k).values.sum()
        return torch.stack([
            entropy,
            top_mass,
            alpha.mean(),
            alpha.max(),
            torch.tensor(float(alpha.numel()), device=alpha.device),
        ])

    def forward(self, attention: torch.Tensor) -> KillChainAlignment:
        summary = self.summarize(attention)
        logits = self.head(summary)
        probs = logits.softmax(-1)
        idx = int(probs.argmax().item())
        stage = KILL_CHAIN_STAGES[idx]
        return KillChainAlignment(
            stage=stage,
            color=STAGE_COLORS[stage],
            confidence=float(probs[idx].item()),
            logits=logits.detach(),
        )

    # -------------------------------------------------------------- #
    def annotate_trajectory(
        self, attention_trajectory: Sequence[torch.Tensor]
    ) -> list[KillChainAlignment]:
        return [self.forward(a) for a in attention_trajectory]
