"""SDE-TGNN: stochastic-differential-equation temporal graph network.

This is the *strongest internal baseline* the manuscript compares against
(96.0 % clean F1, 87.2 % under PGD-40 ε=0.03). It uses the same E-GraphSAGE
encoder as SODE-Guard but evolves the embedding through an SDE **without**
the ellipticity floor and anti-concentration regulariser — by design, this
isolates the contribution of those two ingredients.
"""
from __future__ import annotations
import torch.nn as nn

from ..models.sode_guard import SODEGuard, SODEGuardConfig


class SDE_TGNN(SODEGuard):
    def __init__(self, num_classes: int = 34, feature_dim: int = 83):
        cfg = SODEGuardConfig(
            feature_dim=feature_dim, num_classes=num_classes,
            ellipticity_floor=0.0,                        # ← key ablation
            spectral_norm=False,                          # ← key ablation
            mc_paths_eval=8,
        )
        super().__init__(cfg)
