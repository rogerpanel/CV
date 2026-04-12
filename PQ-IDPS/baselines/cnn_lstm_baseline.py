"""
Classical CNN-LSTM Baseline (No Quantum Components)

Pure classical baseline for comparison with PQ-IDPS.
Uses only the classical pathway without quantum enhancement.

Expected performance (from paper):
- CESNET-TLS-22: 93.1% accuracy (benign), 81.4% (under quantum attack)
- Lower than PQ-IDPS (95.3% benign, 91.7% attacked)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.classical_pathway import ClassicalPathway
from defense.lipschitz_constraints import SpectralNormalization


class CNNLSTMBaseline(nn.Module):
    """
    Classical CNN-LSTM baseline without quantum components.

    Args:
        classical_channels: CNN channel sizes
        lstm_hidden_size: LSTM hidden units
        adversarial_defense: Enable Lipschitz constraints
    """

    def __init__(
        self,
        classical_channels: list = [128, 256, 512],
        lstm_hidden_size: int = 256,
        adversarial_defense: bool = True
    ):
        super().__init__()

        # Classical pathway
        self.classical_pathway = ClassicalPathway(
            input_channels=47,
            cnn_channels=classical_channels,
            lstm_hidden_size=lstm_hidden_size,
            output_dim=512,
            use_spectral_norm=adversarial_defense
        )

        # Classification head
        self.classifier = nn.Sequential(
            SpectralNormalization(nn.Linear(512, 256)) if adversarial_defense
            else nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),

            SpectralNormalization(nn.Linear(256, 128)) if adversarial_defense
            else nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),

            nn.Linear(128, 2)  # Binary classification
        )

    def forward(self, packet_sequences: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through classical pathway only.

        Args:
            packet_sequences: (batch_size, max_packets, 47)

        Returns:
            logits: (batch_size, 2)
        """
        # Extract features with CNN-LSTM
        embedding = self.classical_pathway(packet_sequences)

        # Classify
        logits = self.classifier(embedding)

        return logits


if __name__ == "__main__":
    """Test CNN-LSTM baseline."""

    print("Testing CNN-LSTM Baseline...")
    print("=" * 60)

    # Create model
    model = CNNLSTMBaseline(
        classical_channels=[128, 256, 512],
        lstm_hidden_size=256,
        adversarial_defense=True
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}")
    print()

    # Test forward pass
    batch_size = 16
    packet_sequences = torch.randn(batch_size, 100, 47)

    logits = model(packet_sequences)

    print(f"Input Shape: {packet_sequences.shape}")
    print(f"Output Shape: {logits.shape}")
    print()

    # Test backward pass
    labels = torch.randint(0, 2, (batch_size,))
    loss = F.cross_entropy(logits, labels)
    loss.backward()

    print(f"Loss: {loss.item():.4f}")
    print("✓ Backward pass successful")
    print()

    print("=" * 60)
    print("✓ CNN-LSTM baseline test completed!")
