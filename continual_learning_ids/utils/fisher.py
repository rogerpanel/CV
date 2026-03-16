"""
Fisher Information Matrix computation.

Implements the empirical Fisher diagonal (Equation 5 in the paper):

    F_k^(t) = (1/|D_t|) * Σ (∂log p_θ(y|x) / ∂θ_k)^2

And the exponential moving average accumulation:

    F_hat_k^(t) = α * F_hat_k^(t-1) + (1-α) * F_k^(t)

with α = 0.5 (fisher_decay in config).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


def compute_fisher_diagonal(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_samples: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute the diagonal of the empirical Fisher Information Matrix.

    Equation 5 from the paper:
    F_k^(t) = (1/|D_t|) * Σ_{(x,y) in D_t} (∂log p_θ(y|x) / ∂θ_k)^2

    Args:
        model: The neural network.
        dataloader: DataLoader over the current task data D_t.
        device: Computation device.
        num_samples: Max number of samples to use (None = all).

    Returns:
        Dict mapping parameter name -> diagonal FIM tensor.
    """
    model.eval()
    fisher_diag = {}

    # Initialise to zero
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher_diag[name] = torch.zeros_like(param.data)

    total_samples = 0

    for batch_idx, (features, labels) in enumerate(dataloader):
        features = features.to(device)
        labels = labels.to(device)
        batch_size = features.shape[0]

        if num_samples is not None and total_samples >= num_samples:
            break

        model.zero_grad()

        # Forward pass
        if hasattr(model, "forward") and "features" in str(
            model.forward.__code__.co_varnames
        ):
            logits, _ = model(features)
        else:
            logits = model(features)
            if isinstance(logits, tuple):
                logits = logits[0]

        # Compute log-likelihood for the correct class
        log_probs = F.log_softmax(logits, dim=-1)
        log_likelihood = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)

        # Compute per-sample gradients and accumulate squared gradients
        for i in range(batch_size):
            model.zero_grad()
            log_likelihood[i].backward(retain_graph=(i < batch_size - 1))

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_diag[name] += param.grad.data.pow(2)

            total_samples += 1
            if num_samples is not None and total_samples >= num_samples:
                break

    # Average over samples
    if total_samples > 0:
        for name in fisher_diag:
            fisher_diag[name] /= total_samples

    logger.info(f"Computed Fisher diagonal over {total_samples} samples")
    return fisher_diag


def compute_fisher_diagonal_efficient(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_samples: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Efficient batch-level Fisher computation using the sum of squared gradients.

    Uses batch-level gradient accumulation instead of per-sample computation,
    which is faster but provides an approximation.
    """
    model.eval()
    fisher_diag = {}

    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher_diag[name] = torch.zeros_like(param.data)

    total_samples = 0

    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.to(device)
        batch_size = features.shape[0]

        if num_samples is not None and total_samples >= num_samples:
            break

        model.zero_grad()
        output = model(features)
        logits = output[0] if isinstance(output, tuple) else output

        log_probs = F.log_softmax(logits, dim=-1)
        nll = F.nll_loss(log_probs, labels, reduction="sum")
        nll.backward()

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher_diag[name] += param.grad.data.pow(2) * batch_size

        total_samples += batch_size

    if total_samples > 0:
        for name in fisher_diag:
            fisher_diag[name] /= total_samples

    return fisher_diag


def update_running_fisher(
    running_fisher: Dict[str, torch.Tensor],
    new_fisher: Dict[str, torch.Tensor],
    decay: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """
    Exponential moving average update for the running FIM estimate.

    Equation: F_hat_k^(t) = α * F_hat_k^(t-1) + (1-α) * F_k^(t)

    Args:
        running_fisher: Previous running average F_hat^(t-1).
        new_fisher: New task Fisher F^(t).
        decay: EMA coefficient α (default: 0.5).

    Returns:
        Updated running Fisher F_hat^(t).
    """
    updated = {}
    for name in new_fisher:
        if name in running_fisher:
            updated[name] = (
                decay * running_fisher[name] + (1 - decay) * new_fisher[name]
            )
        else:
            updated[name] = new_fisher[name].clone()
    return updated
