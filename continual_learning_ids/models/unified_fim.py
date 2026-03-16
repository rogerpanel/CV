"""
Unified Fisher Information Framework.

Implements Section IV-C (Equation 12):

  F_hat_k^unified = β * F_hat_k^det + (1-β) * [F_π]_kk

Where:
  - F_hat_k^det: Detection-side FIM from EWC (parameter importance for
    previously learned attack distributions)
  - [F_π]_kk: Policy-side FIM (sensitivity of policy log-likelihood to
    parameter perturbations)
  - β = 0.7: Balances detection preservation vs policy plasticity

Both FIMs share initial layers of the SurrogateIDS feature extractor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class UnifiedFIM:
    """
    Unified Fisher Information Matrix computation.

    Combines detection-side and policy-side Fisher information
    for shared layers, enabling:
      1. EWC knowledge preservation (detection)
      2. Trust-region constraint satisfaction (CPO)
    through a single shared computation.

    Args:
        beta: Mixing coefficient (0.7 = prioritise detection). Eq. 12.
        approximation: FIM approximation type ('diagonal', 'block_diagonal').
    """

    def __init__(
        self,
        beta: float = 0.7,
        approximation: str = "diagonal",
    ):
        self.beta = beta
        self.approximation = approximation
        self.unified_fisher: Dict[str, torch.Tensor] = {}
        self.detection_fisher: Dict[str, torch.Tensor] = {}
        self.policy_fisher: Dict[str, torch.Tensor] = {}

    def compute_detection_fisher(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        shared_param_names: Optional[list] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute detection-side FIM F_hat^det from Equation 5.

        F_k^det = (1/|D|) Σ (∂log p_θ(y|x) / ∂θ_k)^2
        """
        model.eval()
        fisher = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                if shared_param_names is None or name in shared_param_names:
                    fisher[name] = torch.zeros_like(param.data)

        total_samples = 0
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)
            batch_size = features.shape[0]

            model.zero_grad()
            output = model(features)
            logits = output[0] if isinstance(output, tuple) else output
            log_probs = F.log_softmax(logits, dim=-1)
            nll = F.nll_loss(log_probs, labels, reduction="sum")
            nll.backward()

            for name, param in model.named_parameters():
                if name in fisher and param.grad is not None:
                    fisher[name] += param.grad.data.pow(2) * batch_size

            total_samples += batch_size

        for name in fisher:
            fisher[name] /= max(total_samples, 1)

        self.detection_fisher = fisher
        return fisher

    def compute_policy_fisher(
        self,
        policy: nn.Module,
        states: torch.Tensor,
        actions: torch.Tensor,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute policy-side FIM F_π from Equation 11.

        [F_π]_ij = E_{s~d_π, a~π} [
            (∂log π_θ(a|s) / ∂θ_i) * (∂log π_θ(a|s) / ∂θ_j)
        ]
        """
        policy.eval()
        fisher = {}

        for name, param in policy.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param.data)

        states = states.to(device)
        actions = actions.to(device)
        batch_size = min(1024, len(states))

        total_samples = 0
        for i in range(0, len(states), batch_size):
            batch_states = states[i : i + batch_size]
            batch_actions = actions[i : i + batch_size]

            policy.zero_grad()
            logits = policy(batch_states)
            log_probs = F.log_softmax(logits, dim=-1)
            selected_log_probs = log_probs.gather(
                1, batch_actions.unsqueeze(1)
            ).squeeze(1)
            loss = -selected_log_probs.sum()
            loss.backward()

            for name, param in policy.named_parameters():
                if name in fisher and param.grad is not None:
                    fisher[name] += param.grad.data.pow(2) * len(batch_states)

            total_samples += len(batch_states)

        for name in fisher:
            fisher[name] /= max(total_samples, 1)

        self.policy_fisher = fisher
        return fisher

    def compute_unified(
        self,
        detection_fisher: Optional[Dict[str, torch.Tensor]] = None,
        policy_fisher: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute unified FIM: F_hat_k^unified = β * F_det + (1-β) * F_π

        Equation 12 from the paper.

        For shared parameters, both detection and policy Fisher contribute.
        For detection-only parameters, only F_det is used.
        For policy-only parameters, only F_π is used.
        """
        det_fisher = detection_fisher or self.detection_fisher
        pol_fisher = policy_fisher or self.policy_fisher

        unified = {}

        # Shared parameters: weighted combination
        all_params = set(list(det_fisher.keys()) + list(pol_fisher.keys()))

        for name in all_params:
            if name in det_fisher and name in pol_fisher:
                # Shared layer: Equation 12
                unified[name] = (
                    self.beta * det_fisher[name]
                    + (1 - self.beta) * pol_fisher[name]
                )
            elif name in det_fisher:
                # Detection-only parameter
                unified[name] = det_fisher[name]
            else:
                # Policy-only parameter
                unified[name] = pol_fisher[name]

        self.unified_fisher = unified
        logger.info(
            f"Unified FIM computed: {len(unified)} parameters, "
            f"β={self.beta}"
        )
        return unified

    def get_trust_region_matrix(self) -> Dict[str, torch.Tensor]:
        """
        Get FIM for CPO trust region computation.

        Returns the unified FIM if available, otherwise policy-only FIM.
        """
        if self.unified_fisher:
            return self.unified_fisher
        return self.policy_fisher

    def get_ewc_importance(self) -> Dict[str, torch.Tensor]:
        """
        Get FIM for EWC knowledge preservation.

        Returns the unified FIM if available, otherwise detection-only FIM.
        """
        if self.unified_fisher:
            return self.unified_fisher
        return self.detection_fisher

    def compute_parameter_importance_summary(self) -> Dict:
        """Summarise parameter importance across layers."""
        summary = {}
        fisher = self.unified_fisher or self.detection_fisher

        for name, values in fisher.items():
            summary[name] = {
                "mean": values.mean().item(),
                "max": values.max().item(),
                "std": values.std().item(),
                "nonzero_pct": (values > 1e-8).float().mean().item() * 100,
            }

        return summary
