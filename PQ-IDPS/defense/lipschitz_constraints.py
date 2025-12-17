"""
Lipschitz-Constrained Networks via Spectral Normalization

Implements spectral normalization to bound the Lipschitz constant of neural
networks, providing certified adversarial robustness.

Key property: ||f(x) - f(x')|| ≤ L||x - x'||

By constraining L ≤ 1, we limit how much small input perturbations can
affect model outputs, making adversarial attacks more difficult.

Reference:
    Miyato et al., "Spectral Normalization for Generative Adversarial Networks",
    ICLR 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from typing import Optional


class SpectralNormalization(nn.Module):
    """
    Wrapper for applying spectral normalization to linear and convolutional layers.

    Spectral normalization constrains the spectral norm (largest singular value)
    of weight matrices to be at most 1:

        W_normalized = W / max(σ_max(W), 1.0)

    This ensures the Lipschitz constant of the layer is bounded by 1.

    Args:
        layer: nn.Linear or nn.Conv1d/Conv2d layer
        n_power_iterations: Number of power iterations for singular value estimation

    Usage:
        linear = SpectralNormalization(nn.Linear(128, 64))
        conv = SpectralNormalization(nn.Conv1d(47, 128, kernel_size=3))
    """

    def __init__(self, layer: nn.Module, n_power_iterations: int = 1):
        super().__init__()

        # Apply spectral normalization
        if isinstance(layer, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            self.layer = spectral_norm(layer, n_power_iterations=n_power_iterations)
        else:
            raise TypeError(f"Unsupported layer type: {type(layer)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through spectrally normalized layer."""
        return self.layer(x)


class LipschitzLinear(nn.Module):
    """
    Custom implementation of Lipschitz-constrained linear layer.

    Provides more control than PyTorch's built-in spectral_norm.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        lipschitz_constant: Maximum Lipschitz constant (default 1.0)
        bias: Include bias term
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        lipschitz_constant: float = 1.0,
        bias: bool = True
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.lipschitz_constant = lipschitz_constant

        # Initialize weight
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        nn.init.xavier_normal_(self.weight)

        # Initialize bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # Initialize singular value estimate
        self.register_buffer('u', F.normalize(torch.randn(out_features), dim=0))
        self.register_buffer('v', F.normalize(torch.randn(in_features), dim=0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with spectral normalization.

        Uses power iteration to estimate largest singular value.
        """
        if self.training:
            # Update singular value estimate (power iteration)
            with torch.no_grad():
                # v = W^T u / ||W^T u||
                self.v = F.normalize(torch.matmul(self.weight.t(), self.u), dim=0)

                # u = W v / ||W v||
                self.u = F.normalize(torch.matmul(self.weight, self.v), dim=0)

        # Compute spectral norm (largest singular value)
        sigma = torch.dot(self.u, torch.matmul(self.weight, self.v))

        # Normalize weight
        weight_normalized = self.weight * (self.lipschitz_constant / (sigma + 1e-8))

        # Apply linear transformation
        output = F.linear(x, weight_normalized, self.bias)

        return output

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, ' \
               f'lipschitz_constant={self.lipschitz_constant}'


class LipschitzConv1d(nn.Module):
    """
    Lipschitz-constrained 1D convolutional layer.

    Args:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Convolution kernel size
        lipschitz_constant: Maximum Lipschitz constant
        stride: Convolution stride
        padding: Input padding
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        lipschitz_constant: float = 1.0,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.lipschitz_constant = lipschitz_constant

        # Initialize weight
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        nn.init.xavier_normal_(self.weight)

        # Initialize bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        # Initialize singular vectors
        weight_flat = self.weight.view(out_channels, -1)
        self.register_buffer('u', F.normalize(torch.randn(out_channels), dim=0))
        self.register_buffer('v', F.normalize(torch.randn(in_channels * kernel_size), dim=0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with spectral normalization."""
        if self.training:
            # Flatten weight for singular value computation
            weight_flat = self.weight.view(self.out_channels, -1)

            with torch.no_grad():
                # Power iteration
                self.v = F.normalize(torch.matmul(weight_flat.t(), self.u), dim=0)
                self.u = F.normalize(torch.matmul(weight_flat, self.v), dim=0)

        # Compute spectral norm
        weight_flat = self.weight.view(self.out_channels, -1)
        sigma = torch.dot(self.u, torch.matmul(weight_flat, self.v))

        # Normalize weight
        weight_normalized = self.weight * (self.lipschitz_constant / (sigma + 1e-8))

        # Apply convolution
        output = F.conv1d(x, weight_normalized, self.bias, self.stride, self.padding)

        return output


def compute_lipschitz_constant(model: nn.Module) -> float:
    """
    Estimate Lipschitz constant of a neural network.

    For a feedforward network, the Lipschitz constant is bounded by:
        L ≤ ∏ᵢ ||Wᵢ||₂

    This function computes this upper bound.

    Args:
        model: PyTorch model

    Returns:
        lipschitz_constant: Estimated Lipschitz constant
    """
    lipschitz = 1.0

    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            weight = module.weight.data

            # Compute spectral norm (largest singular value)
            if isinstance(module, nn.Linear):
                _, s, _ = torch.svd(weight)
                spectral_norm_value = s.max().item()
            else:
                # For convolutional layers, flatten spatial dimensions
                weight_flat = weight.view(weight.size(0), -1)
                _, s, _ = torch.svd(weight_flat)
                spectral_norm_value = s.max().item()

            lipschitz *= spectral_norm_value

    return lipschitz


def apply_spectral_norm_to_model(model: nn.Module, n_power_iterations: int = 1) -> nn.Module:
    """
    Apply spectral normalization to all Linear and Conv layers in a model.

    Args:
        model: PyTorch model
        n_power_iterations: Number of power iterations

    Returns:
        model: Modified model with spectral normalization
    """
    for name, module in model.named_children():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            # Replace with spectrally normalized version
            setattr(model, name, spectral_norm(module, n_power_iterations=n_power_iterations))
        elif len(list(module.children())) > 0:
            # Recursively apply to submodules
            apply_spectral_norm_to_model(module, n_power_iterations)

    return model


if __name__ == "__main__":
    """Test Lipschitz constraints."""

    print("Testing Lipschitz-Constrained Layers...")
    print("=" * 60)

    # Test SpectralNormalization wrapper
    print("1. SpectralNormalization Wrapper:")
    layer = SpectralNormalization(nn.Linear(128, 64))

    x = torch.randn(16, 128)
    output = layer(x)

    print(f"   Input Shape: {x.shape}")
    print(f"   Output Shape: {output.shape}")

    # Compute spectral norm
    U, S, V = torch.svd(layer.layer.weight)
    spectral_norm_value = S.max().item()
    print(f"   Spectral Norm: {spectral_norm_value:.4f}")
    print(f"   ✓ Should be ≤ 1.0: {spectral_norm_value <= 1.0}")
    print()

    # Test LipschitzLinear
    print("2. LipschitzLinear:")
    lipschitz_layer = LipschitzLinear(128, 64, lipschitz_constant=1.0)

    output = lipschitz_layer(x)
    print(f"   Input Shape: {x.shape}")
    print(f"   Output Shape: {output.shape}")

    # Test backward pass
    loss = output.sum()
    loss.backward()
    print(f"   ✓ Backward pass successful")
    print()

    # Test LipschitzConv1d
    print("3. LipschitzConv1d:")
    conv_layer = LipschitzConv1d(47, 128, kernel_size=5, lipschitz_constant=1.0, padding=2)

    x_conv = torch.randn(16, 47, 100)
    output_conv = conv_layer(x_conv)

    print(f"   Input Shape: {x_conv.shape}")
    print(f"   Output Shape: {output_conv.shape}")
    print()

    # Test Lipschitz constant computation
    print("4. Lipschitz Constant Estimation:")

    test_model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64)
    )

    lipschitz_before = compute_lipschitz_constant(test_model)
    print(f"   Before Spectral Norm: L ≤ {lipschitz_before:.4f}")

    # Apply spectral normalization
    test_model = apply_spectral_norm_to_model(test_model)

    lipschitz_after = compute_lipschitz_constant(test_model)
    print(f"   After Spectral Norm: L ≤ {lipschitz_after:.4f}")
    print(f"   Reduction: {lipschitz_before / lipschitz_after:.2f}x")
    print()

    print("=" * 60)
    print("✓ Lipschitz constraints test completed!")
