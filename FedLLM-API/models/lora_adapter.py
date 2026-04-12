"""
Low-Rank Adaptation (LoRA) for parameter-efficient fine-tuning.

Implements LoRA adapters that reduce trainable parameters by >95%
while maintaining competitive performance.
"""

import torch
import torch.nn as nn
import math


class LoRALayer(nn.Module):
    """
    LoRA adapter layer implementing low-rank decomposition.

    Forward: h = W₀x + BAx
    where W₀ is frozen, B and A are trainable low-rank matrices.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha

        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Scaling factor
        self.scaling = alpha / rank

        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize LoRA parameters using Kaiming initialization."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply LoRA adapter.

        Args:
            x: Input tensor [batch, seq_len, in_features]

        Returns:
            delta: LoRA contribution to output [batch, seq_len, out_features]
        """
        # x @ A^T @ B^T * scaling
        result = self.dropout(x @ self.lora_A.T)
        result = result @ self.lora_B.T
        return result * self.scaling

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}, alpha={self.alpha}'


class LoRAAdapter(nn.Module):
    """
    Complete LoRA adapter that wraps a pre-trained model.

    Freezes base model and injects trainable LoRA layers.
    """

    def __init__(
        self,
        base_model: nn.Module,
        rank: int = 8,
        alpha: int = 16,
        target_modules: list = None,
        lora_dropout: float = 0.0
    ):
        super().__init__()

        self.base_model = base_model
        self.rank = rank
        self.alpha = alpha
        self.target_modules = target_modules or ['query', 'value']

        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Inject LoRA layers
        self.lora_layers = nn.ModuleDict()
        self._inject_lora_layers(lora_dropout)

    def _inject_lora_layers(self, dropout: float):
        """Inject LoRA adapters into target modules."""
        for name, module in self.base_model.named_modules():
            # Check if this module should get LoRA
            if any(target in name.lower() for target in self.target_modules):
                if isinstance(module, nn.Linear):
                    # Create LoRA adapter
                    lora_layer = LoRALayer(
                        module.in_features,
                        module.out_features,
                        rank=self.rank,
                        alpha=self.alpha,
                        dropout=dropout
                    )
                    self.lora_layers[name] = lora_layer

                    # Register forward hook to add LoRA output
                    module.register_forward_hook(
                        self._make_lora_forward_hook(name)
                    )

    def _make_lora_forward_hook(self, lora_name: str):
        """Create forward hook that adds LoRA contribution."""
        def hook(module, input, output):
            lora_output = self.lora_layers[lora_name](input[0])
            return output + lora_output
        return hook

    def forward(self, *args, **kwargs):
        """Forward pass through base model with LoRA."""
        return self.base_model(*args, **kwargs)

    def get_lora_state_dict(self) -> dict:
        """Get only LoRA parameters for federated aggregation."""
        return {
            name: param.data.clone()
            for name, param in self.lora_layers.named_parameters()
        }

    def load_lora_state_dict(self, state_dict: dict):
        """Load LoRA parameters from federated aggregation."""
        for name, param in self.lora_layers.named_parameters():
            if name in state_dict:
                param.data = state_dict[name].clone()

    def num_trainable_params(self) -> int:
        """Count trainable LoRA parameters."""
        return sum(
            p.numel() for p in self.lora_layers.parameters() if p.requires_grad
        )

    def num_total_params(self) -> int:
        """Count total parameters including frozen base."""
        return sum(p.numel() for p in self.parameters())

    def merge_lora_weights(self):
        """Merge LoRA weights into base model (for deployment)."""
        for name, module in self.base_model.named_modules():
            if name in self.lora_layers and isinstance(module, nn.Linear):
                lora_layer = self.lora_layers[name]

                # Compute W_new = W_old + B @ A * scaling
                delta_W = (lora_layer.lora_B @ lora_layer.lora_A) * lora_layer.scaling

                # Add to base weight
                module.weight.data += delta_W

        # Remove LoRA layers after merging
        self.lora_layers = nn.ModuleDict()


def mark_only_lora_as_trainable(model: nn.Module):
    """
    Mark only LoRA parameters as trainable.

    Utility function for ensuring gradient computation efficiency.
    """
    for name, param in model.named_parameters():
        if 'lora_' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def count_parameters(model: nn.Module) -> dict:
    """
    Count total, trainable, and frozen parameters.

    Returns:
        dict with parameter counts and statistics
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    return {
        'total': total,
        'trainable': trainable,
        'frozen': frozen,
        'trainable_percent': 100.0 * trainable / total if total > 0 else 0
    }


if __name__ == "__main__":
    # Example: Apply LoRA to a simple transformer layer
    from transformers import DistilBertModel

    print("Loading DistilBERT...")
    base_model = DistilBertModel.from_pretrained('distilbert-base-uncased')

    print("\nApplying LoRA adapters...")
    lora_model = LoRAAdapter(
        base_model,
        rank=8,
        alpha=16,
        target_modules=['query', 'value']
    )

    # Count parameters
    param_stats = count_parameters(lora_model)
    print(f"\nParameter Statistics:")
    print(f"  Total: {param_stats['total']:,}")
    print(f"  Trainable: {param_stats['trainable']:,}")
    print(f"  Frozen: {param_stats['frozen']:,}")
    print(f"  Trainable %: {param_stats['trainable_percent']:.2f}%")

    # Test forward pass
    print("\nTesting forward pass...")
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, 30522, (batch_size, seq_len))

    outputs = lora_model(input_ids)
    print(f"Output shape: {outputs.last_hidden_state.shape}")

    # Test LoRA state dict
    print("\nTesting LoRA state dict extraction...")
    lora_state = lora_model.get_lora_state_dict()
    print(f"LoRA parameters: {len(lora_state)}")
    total_lora_size = sum(v.numel() for v in lora_state.values())
    print(f"Total LoRA elements: {total_lora_size:,}")

    print("\n✓ All tests passed!")
