"""
Structured Variational Bayesian Inference.

Implements the ELBO (Eq. 12) and structured mean-field posterior
from Section 4.3 of the main manuscript.

- Eq 12: L_ELBO = E_{q(theta)}[log p(D|theta)] - KL(q(theta) || p(theta))
- Structured posterior: q(theta) = prod_b q(theta^(b))
  with q(theta^(b)) = N(mu_b, Sigma_b), Sigma_b = D_b R_b D_b,
  R_b = I + V_b V_b^T  (low-rank, r << d_b)
- Complexity: O(B r d_b) vs O(d^2) for full covariance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple


class LowRankGaussian(nn.Module):
    """Low-rank Gaussian variational distribution for one parameter block.

    Sigma = diag(d)^2 (I + V V^T)
    where d is a diagonal scaling and V is [dim, rank].
    """

    def __init__(self, param_shape: torch.Size, rank: int = 10):
        super().__init__()
        dim = param_shape.numel()
        self.dim = dim
        self.param_shape = param_shape
        self.rank = rank

        # Variational parameters
        self.mu = nn.Parameter(torch.zeros(dim))
        self.log_d = nn.Parameter(torch.zeros(dim))  # log diagonal scaling
        self.V = nn.Parameter(torch.randn(dim, rank) * 0.01)

    @property
    def d(self) -> torch.Tensor:
        return torch.exp(self.log_d)

    def sample(self, n_samples: int = 1) -> torch.Tensor:
        """Sample theta ~ q(theta) using the reparameterization trick."""
        eps_diag = torch.randn(n_samples, self.dim, device=self.mu.device)
        eps_lr = torch.randn(n_samples, self.rank, device=self.mu.device)

        d = self.d
        # theta = mu + d * (eps_diag + V @ eps_lr)
        samples = self.mu.unsqueeze(0) + d.unsqueeze(0) * (
            eps_diag + eps_lr @ self.V.t()
        )
        return samples.view(n_samples, *self.param_shape)

    def kl_divergence(self) -> torch.Tensor:
        """KL(q || N(0, I)) with low-rank structure.

        KL = 0.5 * (tr(Sigma) + mu^T mu - dim - log det(Sigma))

        For Sigma = D (I + V V^T) D:
          tr(Sigma) = sum(d^2) + sum(d^2 * ||v_r||^2)   (approx)
          log det(Sigma) = 2 sum(log d) + log det(I + V^T D^2 V)
        """
        d = self.d
        d_sq = d ** 2

        # Trace term
        VtDV = (self.V * d_sq.unsqueeze(1)).t() @ self.V  # [rank, rank]
        trace_sigma = d_sq.sum() + torch.trace(VtDV)

        # Log determinant via matrix determinant lemma
        I_r = torch.eye(self.rank, device=self.mu.device)
        log_det = 2.0 * self.log_d.sum() + torch.logdet(I_r + VtDV)

        # mu^T mu
        mu_sq = (self.mu ** 2).sum()

        kl = 0.5 * (trace_sigma + mu_sq - self.dim - log_det)
        return kl


class StructuredVariationalPosterior(nn.Module):
    """Structured mean-field variational posterior over all model parameters.

    q(theta) = prod_b q(theta^(b))

    Each block b corresponds to a parameter group (e.g., one layer's
    weight matrix). The total KL is the sum of per-block KLs.
    """

    def __init__(self, base_model: nn.Module, rank: int = 10):
        super().__init__()
        self.base_model = base_model
        self.rank = rank

        # Create variational distributions for each parameter
        self.var_params = nn.ModuleDict()
        self.param_names = []

        for name, param in base_model.named_parameters():
            safe_name = name.replace(".", "_")
            self.var_params[safe_name] = LowRankGaussian(param.shape, rank)
            self.param_names.append((name, safe_name))
            # Initialize mean at current parameter value
            self.var_params[safe_name].mu.data.copy_(param.data.flatten())

    def sample_parameters(self) -> None:
        """Sample parameters from q and set them on the base model."""
        for name, safe_name in self.param_names:
            var = self.var_params[safe_name]
            sample = var.sample(1).squeeze(0)
            # Set sampled parameter on base model
            parts = name.split(".")
            module = self.base_model
            for part in parts[:-1]:
                module = getattr(module, part)
            getattr(module, parts[-1]).data.copy_(sample)

    def get_mean_parameters(self) -> None:
        """Set parameters to posterior mean (for evaluation)."""
        for name, safe_name in self.param_names:
            var = self.var_params[safe_name]
            mean = var.mu.data.view(var.param_shape)
            parts = name.split(".")
            module = self.base_model
            for part in parts[:-1]:
                module = getattr(module, part)
            getattr(module, parts[-1]).data.copy_(mean)

    def total_kl(self) -> torch.Tensor:
        """Sum of KL divergences across all blocks."""
        kl = torch.tensor(0.0, device=next(self.parameters()).device)
        for safe_name in self.var_params:
            kl = kl + self.var_params[safe_name].kl_divergence()
        return kl


class BayesianWrapper(nn.Module):
    """Wraps a deterministic model with structured variational inference.

    Provides ELBO computation and MC-sampled predictions with
    uncertainty quantification.
    """

    def __init__(self, model: nn.Module, rank: int = 10):
        super().__init__()
        self.posterior = StructuredVariationalPosterior(model, rank)
        self.model = model

    def compute_elbo(self, loss_fn, *args,
                     n_samples: int = 10, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute ELBO = E_q[log p(D|theta)] - KL(q||p).

        Args:
            loss_fn: Function that takes model output and returns NLL
            n_samples: Number of MC samples
        Returns:
            (negative_elbo, kl_divergence)
        """
        total_nll = 0.0
        for _ in range(n_samples):
            self.posterior.sample_parameters()
            nll = loss_fn(self.model, *args, **kwargs)
            total_nll = total_nll + nll

        avg_nll = total_nll / n_samples
        kl = self.posterior.total_kl()
        neg_elbo = avg_nll + kl

        return neg_elbo, kl

    def predict_with_uncertainty(self, forward_fn, *args,
                                 n_samples: int = 50, **kwargs):
        """MC-sampled predictions for uncertainty quantification.

        Returns:
            mean_output: Mean prediction across samples
            std_output:  Standard deviation (epistemic uncertainty)
            all_outputs: All MC samples
        """
        outputs = []
        self.model.eval()
        with torch.no_grad():
            for _ in range(n_samples):
                self.posterior.sample_parameters()
                out = forward_fn(self.model, *args, **kwargs)
                outputs.append(out)

        stacked = torch.stack(outputs)
        return stacked.mean(0), stacked.std(0), stacked
