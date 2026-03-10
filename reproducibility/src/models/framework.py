"""
Integrated TA-BN-ODE + DSTPP Framework
=======================================
Complete end-to-end model combining:
  - Multi-scale TA-BN-ODE for continuous state dynamics
  - Deep Spatio-Temporal Point Processes for event intensity
  - Structured Bayesian inference for uncertainty quantification
  - Bidirectional coupling between continuous and discrete components

Corresponds to Figure 2 (complete pipeline) in the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

from .tabn_ode import MultiScaleTABNODE
from .point_process import DeepSpatioTemporalPointProcess
from .bayesian import BayesianWrapper, TemperatureScaling


class TABNODEPointProcessFramework(nn.Module):
    """Complete framework integrating TA-BN-ODE, DSTPP, and Bayesian inference.

    Architecture (Figure 2):
        Input → Encoder → TA-BN-ODE (continuous) ←→ DSTPP (discrete)
                                                ↓
                                    Bayesian Classifier → Prediction
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256,
                 n_classes: int = 2, n_ode_blocks: int = 2,
                 time_constants=None,
                 n_heads: int = 8, n_transformer_layers: int = 4,
                 d_model: int = 512,
                 solver: str = "dopri5",
                 rtol: float = 1e-3, atol: float = 1e-4,
                 mc_samples_train: int = 10, mc_samples_test: int = 50,
                 dropout: float = 0.1, mu_barrier: float = 0.01):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

        # 1. Multi-Scale TA-BN-ODE
        self.ode = MultiScaleTABNODE(
            input_dim=input_dim, hidden_dim=hidden_dim,
            n_ode_blocks=n_ode_blocks, time_constants=time_constants,
            solver=solver, rtol=rtol, atol=atol,
        )

        # 2. Deep Spatio-Temporal Point Process
        self.dstpp = DeepSpatioTemporalPointProcess(
            hidden_dim=hidden_dim, n_marks=n_classes,
            n_heads=n_heads, n_layers=n_transformer_layers,
            d_model=d_model, mu_barrier=mu_barrier,
        )

        # 3. Coupling network (bidirectional ODE ↔ PP)
        self.coupling = nn.Sequential(
            nn.Linear(hidden_dim + n_classes, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 4. Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, n_classes),
        )

        # 5. Temperature scaling (fitted post-training)
        self.temp_scaling = TemperatureScaling()

        # MC dropout for uncertainty
        self.mc_dropout = nn.Dropout(p=dropout)
        self.mc_samples_train = mc_samples_train
        self.mc_samples_test = mc_samples_test

        # Noise parameter for trajectory sampling
        self.log_noise = nn.Parameter(torch.zeros(1))

    @property
    def n_attack_types(self):
        return self.n_classes

    def forward(self, x: torch.Tensor,
                t_span: torch.Tensor,
                timestamps: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None,
                ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (batch, input_dim) event features
            t_span: 1-D ODE integration time grid
            timestamps: (batch, seq_len) for PP intensity (optional)
            mask: (batch, seq_len) for PP (optional)
        Returns:
            logits: (batch, n_classes)
            h_final: (batch, hidden_dim)
            intensities: (batch, seq_len, n_marks) or None
        """
        # ODE forward pass
        h_final, h0 = self.ode(x, t_span)

        # Point process intensity (if timestamps provided)
        intensities = None
        if timestamps is not None:
            h_seq = h_final.unsqueeze(1).expand(-1, timestamps.size(1), -1)
            intensities = self.dstpp(h_seq, timestamps, mask)

            # Coupling: fuse ODE state with PP intensity
            pp_summary = intensities.mean(dim=1)  # (batch, n_marks)
            coupled = self.coupling(
                torch.cat([h_final, pp_summary], dim=-1)
            )
            h_final = h_final + coupled

        # Classification
        logits = self.classifier(h_final)

        return logits, h_final, intensities

    def predict_with_uncertainty(
            self, x: torch.Tensor, t_span: torch.Tensor,
            n_samples: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """MC-dropout prediction with uncertainty quantification.

        Returns:
            mean_probs: (batch, n_classes) averaged softmax
            uncertainty: (batch, n_classes) predictive std
            logits_all: (n_samples, batch, n_classes)
        """
        if n_samples is None:
            n_samples = (self.mc_samples_train if self.training
                         else self.mc_samples_test)

        self.mc_dropout.train()  # keep dropout active
        all_logits = []

        with torch.no_grad():
            for _ in range(n_samples):
                h_final, _ = self.ode(x, t_span)
                # Add noise for trajectory diversity
                noise = torch.randn_like(h_final) * torch.exp(self.log_noise)
                h_noisy = self.mc_dropout(h_final + noise)
                logits = self.classifier(h_noisy)
                all_logits.append(logits)

        stacked = torch.stack(all_logits, dim=0)
        probs = F.softmax(stacked, dim=-1)

        mean_probs = probs.mean(dim=0)
        uncertainty = probs.std(dim=0)

        return mean_probs, uncertainty, stacked

    def compute_loss(
            self, x: torch.Tensor, y: torch.Tensor,
            t_span: torch.Tensor,
            timestamps: Optional[torch.Tensor] = None,
            marks: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None,
            T: float = 1.0,
            kl_weight: float = 1e-4,
            tpp_weight: float = 0.1,
            reg_weight: float = 1e-3,
    ) -> Dict[str, torch.Tensor]:
        """Compute total loss (Eq. 3): L = L_cls + L_TPP + L_ELBO + L_reg.

        Returns dict with individual loss components for logging.
        """
        logits, h_final, intensities = self.forward(
            x, t_span, timestamps, mask
        )

        # L_cls: classification cross-entropy
        loss_cls = F.cross_entropy(logits, y)

        # L_TPP: point process NLL with log-barrier
        loss_tpp = torch.tensor(0.0, device=x.device)
        if intensities is not None and marks is not None:
            loss_tpp = self.dstpp.compute_loss(
                intensities, timestamps, marks, T, mask
            )

        # L_ELBO (KL proxy): weight regularisation
        kl_proxy = sum(p.pow(2).sum() for p in self.parameters()) * 0.5
        loss_kl = kl_weight * kl_proxy

        # L_reg: TA-BN stability regularisation (Theorem 1)
        loss_reg = reg_weight * self.ode.regularisation_loss(t_span)

        total = loss_cls + tpp_weight * loss_tpp + loss_kl + loss_reg

        return {
            "total": total,
            "cls": loss_cls,
            "tpp": loss_tpp,
            "kl": loss_kl,
            "reg": loss_reg,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> str:
        n_params = self.count_parameters()
        return (
            f"TABNODEPointProcessFramework\n"
            f"  Parameters: {n_params:,} ({n_params / 1e6:.2f}M)\n"
            f"  Hidden dim: {self.hidden_dim}\n"
            f"  Classes: {self.n_classes}\n"
            f"  ODE solver: {self.ode.scale_blocks[0][0].solver}\n"
            f"  Time constants: {self.ode.time_constants.tolist()}\n"
        )
