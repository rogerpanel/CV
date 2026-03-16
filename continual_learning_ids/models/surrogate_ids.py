"""
SurrogateIDS: Seven-Branch Surrogate Ensemble with MC Dropout.

The detection backbone from the RobustIDPS.ai platform. Each branch
specialises in different feature extraction patterns, and MC Dropout
provides epistemic uncertainty estimates for the RL response agent.

Architecture (Section IV-A):
  - 7 parallel branches, each: Linear -> BN -> ReLU -> Dropout -> Linear -> BN -> ReLU
  - Shared feature fusion layer
  - Classification head with MC Dropout (p=0.1, T=20 forward passes)
  - State vector output (dim=55) for the RL agent
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np


class BranchNetwork(nn.Module):
    """Single branch of the surrogate ensemble."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
            ])
            prev_dim = h_dim
        self.network = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class SurrogateIDS(nn.Module):
    """
    Seven-Branch Surrogate Ensemble IDS with MC Dropout.

    Implements the detection backbone described in Section IV:
    - 7 parallel branches for diverse feature extraction
    - Shared feature fusion via concatenation + projection
    - MC Dropout for epistemic uncertainty estimation
    - Produces both classification output and RL state vector

    Args:
        input_dim: Number of input features (max 79 across datasets).
        num_classes: Number of output classes (up to 83 unique attacks + benign).
        num_branches: Number of parallel branches (default: 7).
        branch_hidden_dims: Hidden layer sizes per branch.
        shared_dim: Dimension of shared feature representation.
        mc_dropout_rate: Dropout rate for MC Dropout (default: 0.1).
        mc_samples: Number of forward passes for uncertainty (default: 20).
    """

    def __init__(
        self,
        input_dim: int = 79,
        num_classes: int = 34,
        num_branches: int = 7,
        branch_hidden_dims: list = None,
        shared_dim: int = 256,
        mc_dropout_rate: float = 0.1,
        mc_samples: int = 20,
    ):
        super().__init__()
        if branch_hidden_dims is None:
            branch_hidden_dims = [512, 256, 128]

        self.num_branches = num_branches
        self.num_classes = num_classes
        self.mc_samples = mc_samples
        self.mc_dropout_rate = mc_dropout_rate

        # Seven parallel branches
        self.branches = nn.ModuleList([
            BranchNetwork(input_dim, branch_hidden_dims, mc_dropout_rate)
            for _ in range(num_branches)
        ])

        # Feature fusion: concatenate all branches -> shared representation
        branch_out_dim = branch_hidden_dims[-1]
        fusion_input_dim = branch_out_dim * num_branches
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=mc_dropout_rate),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(shared_dim, shared_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=mc_dropout_rate),
            nn.Linear(shared_dim // 2, num_classes),
        )

        # Shared feature layers (first 3 layers used by both detection & RL)
        # These are the layers whose FIM is shared with the policy network
        self.shared_layer_names = self._get_shared_layer_names()

    def _get_shared_layer_names(self) -> list:
        """Identify shared layers for unified FIM computation."""
        shared = []
        for name, _ in self.named_parameters():
            if "fusion" in name or any(
                f"branches.{i}" in name for i in range(3)
            ):
                shared.append(name)
        return shared

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass producing logits and shared features.

        Returns:
            logits: (batch, num_classes)
            features: (batch, shared_dim) - shared representation for RL
        """
        branch_outputs = [branch(x) for branch in self.branches]
        concatenated = torch.cat(branch_outputs, dim=1)
        features = self.fusion(concatenated)
        logits = self.classifier(features)
        return logits, features

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        num_samples: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        MC Dropout inference for uncertainty estimation.

        Performs T stochastic forward passes and computes:
        - Mean prediction probabilities
        - Epistemic uncertainty (mutual information)
        - Aleatoric uncertainty (expected entropy)

        Args:
            x: Input tensor (batch, features).
            num_samples: Override number of MC samples.

        Returns:
            Dict with keys: 'probabilities', 'predictions',
            'epistemic_uncertainty', 'aleatoric_uncertainty', 'features'
        """
        T = num_samples or self.mc_samples
        self.train()  # Enable dropout

        all_probs = []
        all_features = []
        with torch.no_grad():
            for _ in range(T):
                logits, features = self.forward(x)
                probs = F.softmax(logits, dim=-1)
                all_probs.append(probs)
                all_features.append(features)

        self.eval()

        # Stack: (T, batch, num_classes)
        stacked_probs = torch.stack(all_probs, dim=0)
        mean_probs = stacked_probs.mean(dim=0)  # (batch, num_classes)
        mean_features = torch.stack(all_features, dim=0).mean(dim=0)

        # Epistemic uncertainty: mutual information
        # I[y, θ | x] = H[E[p]] - E[H[p]]
        entropy_of_mean = -torch.sum(
            mean_probs * torch.log(mean_probs + 1e-10), dim=-1
        )
        mean_of_entropy = -torch.mean(
            torch.sum(
                stacked_probs * torch.log(stacked_probs + 1e-10), dim=-1
            ),
            dim=0,
        )
        epistemic = entropy_of_mean - mean_of_entropy

        # Aleatoric uncertainty: expected entropy
        aleatoric = mean_of_entropy

        predictions = mean_probs.argmax(dim=-1)

        return {
            "probabilities": mean_probs,
            "predictions": predictions,
            "epistemic_uncertainty": epistemic,
            "aleatoric_uncertainty": aleatoric,
            "features": mean_features,
        }

    def construct_rl_state(
        self,
        x: torch.Tensor,
        flow_metadata: Optional[torch.Tensor] = None,
        context_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Construct the 55-dimensional RL state vector.

        State composition (Section IV-B):
          - Detection probabilities p ∈ R^K (up to num_classes)
          - Epistemic + aleatoric uncertainty ∈ R^2
          - Flow metadata (IP encodings, port, protocol, etc.) ∈ R^13
          - Contextual features (connection rate, alert count, CPU, mem) ∈ R^4

        For simplicity, we pad/truncate to exactly 55 dimensions.
        """
        result = self.predict_with_uncertainty(x)

        components = [result["probabilities"]]  # (batch, K)

        # Uncertainty: (batch, 2)
        uncertainty = torch.stack(
            [result["epistemic_uncertainty"], result["aleatoric_uncertainty"]],
            dim=-1,
        )
        components.append(uncertainty)

        if flow_metadata is not None:
            components.append(flow_metadata)

        if context_features is not None:
            components.append(context_features)

        state = torch.cat(components, dim=-1)

        # Pad or truncate to 55
        target_dim = 55
        if state.shape[-1] < target_dim:
            pad = torch.zeros(
                state.shape[0], target_dim - state.shape[-1],
                device=state.device,
            )
            state = torch.cat([state, pad], dim=-1)
        elif state.shape[-1] > target_dim:
            state = state[:, :target_dim]

        return state

    def get_shared_parameters(self) -> list:
        """Get parameters from shared layers (for unified FIM)."""
        shared_params = []
        for name, param in self.named_parameters():
            if name in self.shared_layer_names:
                shared_params.append(param)
        return shared_params

    def get_num_classes(self) -> int:
        return self.num_classes

    def update_classifier_head(self, new_num_classes: int):
        """Expand classifier for new classes encountered in continual learning."""
        if new_num_classes <= self.num_classes:
            return

        old_weight = self.classifier[-1].weight.data
        old_bias = self.classifier[-1].bias.data
        in_features = self.classifier[-1].in_features

        new_layer = nn.Linear(in_features, new_num_classes)
        new_layer.weight.data[:self.num_classes] = old_weight
        new_layer.bias.data[:self.num_classes] = old_bias
        # Xavier init for new classes
        nn.init.xavier_uniform_(new_layer.weight.data[self.num_classes:])
        nn.init.zeros_(new_layer.bias.data[self.num_classes:])

        self.classifier[-1] = new_layer
        self.num_classes = new_num_classes
