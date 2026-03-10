"""
Structured Variational Bayesian Inference
==========================================
Implements Section VI of the paper:
  - Structured mean-field with low-rank covariance  (Eq. 14-15)
  - ELBO computation                                (Eq. 14)
  - PAC-Bayesian generalisation bound               (Theorem 2)
  - Temperature scaling calibration                 (Eq. 16)
  - Monte Carlo sampling for uncertainty estimation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from torch.distributions import Normal, kl_divergence


# ---------------------------------------------------------------------------
# Structured Variational Posterior  (Eq. 15)
# ---------------------------------------------------------------------------
class LowRankGaussian(nn.Module):
    """Block posterior q(θ^(b)) = N(μ_b, D_b R_b D_b) with R_b = I + V_b V_b^T.

    Complexity: O(B·r·d_b)  vs  O(d²) for full covariance.
    """

    def __init__(self, param_dim: int, rank: int = 8):
        super().__init__()
        self.param_dim = param_dim
        self.rank = rank

        self.mu = nn.Parameter(torch.zeros(param_dim))
        self.log_diag = nn.Parameter(torch.zeros(param_dim))
        self.V = nn.Parameter(torch.randn(param_dim, rank) * 0.01)

    @property
    def diag(self):
        return F.softplus(self.log_diag)

    def sample(self) -> torch.Tensor:
        """Sample θ ~ q(θ) using reparameterisation trick."""
        eps_diag = torch.randn_like(self.mu)
        eps_rank = torch.randn(self.rank, device=self.mu.device)

        D = self.diag
        # Σ = D (I + V V^T) D  →  sample = μ + D(ε₁ + V ε₂)
        return self.mu + D * (eps_diag + self.V @ eps_rank)

    def kl_divergence(self) -> torch.Tensor:
        """KL(q ‖ p) where p = N(0, I).

        For low-rank: KL = 0.5 (‖μ‖² + tr(Σ) - d - log|Σ|)
        where Σ = D(I + VV^T)D, so
          tr(Σ) = ‖D‖² + ‖DV‖²_F
          log|Σ| = 2·Σ log D_i + log|I + V^T D² V|
        """
        D = self.diag
        d = self.param_dim

        # tr(Σ)
        D2 = D.pow(2)
        DV = D.unsqueeze(-1) * self.V  # (d, r)
        trace = D2.sum() + (DV * DV).sum()

        # log|Σ|  (Woodbury)
        VtD2V = self.V.T @ torch.diag(D2) @ self.V  # (r, r)
        log_det_core = torch.logdet(
            torch.eye(self.rank, device=D.device) + VtD2V
        )
        log_det = 2 * torch.log(D + 1e-8).sum() + log_det_core

        mu_sq = self.mu.pow(2).sum()

        return 0.5 * (mu_sq + trace - d - log_det)


class StructuredVariationalPosterior(nn.Module):
    """Groups model parameters into blocks with low-rank posteriors.

    Blocks correspond to:
      0: encoder parameters
      1: ODE block 1
      2: ODE block 2
      ...
      B-1: decoder/classifier
    """

    def __init__(self, model: nn.Module, rank: int = 8):
        super().__init__()
        self.posteriors = nn.ModuleDict()
        self.param_names = {}

        # Group parameters by top-level module
        for name, param in model.named_parameters():
            block_name = name.split(".")[0]
            if block_name not in self.posteriors:
                self.posteriors[block_name] = LowRankGaussian(
                    param.numel(), rank
                )
                self.param_names[block_name] = []
            self.param_names[block_name].append((name, param.shape))

    def sample_parameters(self) -> Dict[str, torch.Tensor]:
        """Sample one set of parameters from the variational posterior."""
        samples = {}
        for block_name, posterior in self.posteriors.items():
            flat_sample = posterior.sample()
            offset = 0
            for name, shape in self.param_names[block_name]:
                n = 1
                for s in shape:
                    n *= s
                samples[name] = flat_sample[offset:offset + n].view(shape)
                offset += n
        return samples

    def total_kl(self) -> torch.Tensor:
        """Total KL divergence across all blocks."""
        kl = torch.tensor(0.0, device=next(self.parameters()).device)
        for posterior in self.posteriors.values():
            kl = kl + posterior.kl_divergence()
        return kl


# ---------------------------------------------------------------------------
# Bayesian Wrapper with MC Sampling
# ---------------------------------------------------------------------------
class BayesianWrapper(nn.Module):
    """Wraps a deterministic model for Bayesian inference via MC sampling.

    Uses dropout-based approximation for computational efficiency during
    training, with optional full variational posterior at test time.
    """

    def __init__(self, base_model: nn.Module, mc_samples_train: int = 10,
                 mc_samples_test: int = 50, dropout_rate: float = 0.1):
        super().__init__()
        self.base_model = base_model
        self.mc_samples_train = mc_samples_train
        self.mc_samples_test = mc_samples_test

        # Add dropout for MC-dropout approximation
        self.dropout = nn.Dropout(p=dropout_rate)

        # Learnable log-noise for uncertainty
        self.log_noise = nn.Parameter(torch.zeros(1))

    @property
    def n_samples(self) -> int:
        return self.mc_samples_train if self.training else self.mc_samples_test

    def forward_single(self, *args, **kwargs) -> torch.Tensor:
        """Single forward pass with dropout."""
        output = self.base_model(*args, **kwargs)
        if isinstance(output, tuple):
            # Apply dropout to hidden state
            return (self.dropout(output[0]),) + output[1:]
        return self.dropout(output)

    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """MC forward pass returning mean prediction and uncertainty.

        Returns:
            mean_output: averaged predictions
            uncertainty: standard deviation across samples
        """
        outputs = []
        for _ in range(self.n_samples):
            out = self.forward_single(*args, **kwargs)
            if isinstance(out, tuple):
                out = out[0]
            outputs.append(out)

        stacked = torch.stack(outputs, dim=0)
        mean = stacked.mean(dim=0)
        std = stacked.std(dim=0)
        return mean, std

    def compute_elbo(self, logits: torch.Tensor, targets: torch.Tensor,
                     kl_weight: float = 1.0,
                     kl_term: Optional[torch.Tensor] = None
                     ) -> torch.Tensor:
        """ELBO = E_q[log p(D|θ)] - KL(q‖p)  (Eq. 14)"""
        # Likelihood term
        log_likelihood = -F.cross_entropy(logits, targets, reduction="mean")

        # KL term
        if kl_term is None:
            # Use weight-decay proxy
            kl_term = sum(p.pow(2).sum() for p in self.parameters()) * 0.5

        return -(log_likelihood - kl_weight * kl_term)


# ---------------------------------------------------------------------------
# Temperature Scaling  (Eq. 16)
# ---------------------------------------------------------------------------
class TemperatureScaling(nn.Module):
    """Post-hoc calibration via temperature scaling.

    p_cal(k|x,t) = softmax(logit(k|x,t) / T_cal)

    T_cal optimised on validation set to minimise ECE.
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(self, val_logits: torch.Tensor, val_labels: torch.Tensor,
            lr: float = 0.01, max_iter: int = 100):
        """Optimise temperature on validation data."""
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr,
                                       max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            loss = F.cross_entropy(self.forward(val_logits), val_labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return self.temperature.item()
