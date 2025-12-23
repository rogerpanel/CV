"""
PQ-IDPS: Hybrid Classical-Quantum Intrusion Detection System

Main model architecture combining:
1. Protocol Detector: Identifies TLS handshake type (classical/hybrid/pure PQC)
2. Classical Pathway: CNN-LSTM for traditional TLS features
3. Quantum Pathway: Variational Quantum Classifier for PQC patterns
4. Adaptive Fusion: Protocol-aware dynamic weighting

Reference:
    Anaedevha et al., "PQ-IDPS: Adversarially Robust Intrusion Detection for
    Post-Quantum Encrypted Traffic", IEEE TIFS 2025.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import numpy as np

from models.classical_pathway import ClassicalPathway
from models.quantum_pathway import QuantumPathway
from defense.lipschitz_constraints import SpectralNormalization


class ProtocolDetector(nn.Module):
    """
    Detects TLS 1.3 handshake type from Client Hello message.

    Classifies into:
    - Classical ECDHE (type 0)
    - Hybrid X25519+MLKEM-768 (type 1)
    - Pure MLKEM-768 (type 2)

    Protocol detection informs adaptive fusion weights.
    """

    def __init__(self, input_dim: int = 47, hidden_dim: int = 128):
        super().__init__()

        self.detector = nn.Sequential(
            SpectralNormalization(nn.Linear(input_dim, hidden_dim)),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),

            SpectralNormalization(nn.Linear(hidden_dim, 64)),
            nn.ReLU(),
            nn.BatchNorm1d(64),

            nn.Linear(64, 3)  # 3 protocol types
        )

    def forward(self, client_hello_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            client_hello_features: (batch_size, 47) - First packet features

        Returns:
            protocol_logits: (batch_size, 3) - Protocol type logits
            protocol_probs: (batch_size, 3) - Softmax probabilities
        """
        logits = self.detector(client_hello_features)
        probs = F.softmax(logits, dim=-1)

        return logits, probs


class AdaptiveFusion(nn.Module):
    """
    Protocol-aware adaptive fusion of classical and quantum pathways.

    Fusion weights depend on detected protocol type:
    - Classical ECDHE: 85% classical, 15% quantum
    - Hybrid X25519+MLKEM: 70% classical, 30% quantum
    - Pure MLKEM-768: 30% classical, 70% quantum

    Learnable gating allows fine-tuning beyond these base weights.
    """

    def __init__(self, classical_dim: int = 512, quantum_dim: int = 1,
                 hidden_dim: int = 128):
        super().__init__()

        # Base fusion weights per protocol type
        self.register_buffer('base_weights', torch.tensor([
            [0.85, 0.15],  # Classical ECDHE
            [0.70, 0.30],  # Hybrid X25519+MLKEM
            [0.30, 0.70]   # Pure MLKEM-768
        ]))

        # Learnable gating network
        self.gate = nn.Sequential(
            SpectralNormalization(nn.Linear(classical_dim + 3, hidden_dim)),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),

            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1)
        )

        # Final classification head
        self.classifier = nn.Sequential(
            SpectralNormalization(nn.Linear(classical_dim + 1, 256)),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),

            SpectralNormalization(nn.Linear(256, 128)),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.Linear(128, 2)  # Binary classification
        )

    def forward(
        self,
        classical_embedding: torch.Tensor,
        quantum_prediction: torch.Tensor,
        protocol_probs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            classical_embedding: (batch_size, 512) - CNN-LSTM output
            quantum_prediction: (batch_size, 1) - VQC output
            protocol_probs: (batch_size, 3) - Protocol type probabilities

        Returns:
            logits: (batch_size, 2) - Final classification logits
            fusion_weights: (batch_size, 2) - Applied fusion weights
        """
        batch_size = classical_embedding.size(0)

        # Compute base weights from protocol detection
        base_weights = torch.matmul(protocol_probs, self.base_weights)  # (batch_size, 2)

        # Compute learned gating adjustments
        gate_input = torch.cat([classical_embedding, protocol_probs], dim=-1)
        learned_weights = self.gate(gate_input)  # (batch_size, 2)

        # Combine base and learned weights (50-50 interpolation)
        fusion_weights = 0.5 * base_weights + 0.5 * learned_weights
        fusion_weights = F.normalize(fusion_weights, p=1, dim=-1)  # Ensure sum to 1

        # Apply fusion
        # Classical pathway contributes embedding features
        # Quantum pathway contributes scalar prediction
        fused_features = torch.cat([
            classical_embedding * fusion_weights[:, 0:1],  # Weight classical
            quantum_prediction * fusion_weights[:, 1:2]     # Weight quantum
        ], dim=-1)

        # Final classification
        logits = self.classifier(fused_features)

        return logits, fusion_weights


class PQIDPS(nn.Module):
    """
    PQ-IDPS: Complete hybrid classical-quantum intrusion detection system.

    Architecture:
    1. Protocol Detector identifies handshake type
    2. Classical Pathway (CNN-LSTM) processes packet sequences
    3. Quantum Pathway (VQC) processes statistical features
    4. Adaptive Fusion combines pathways with protocol-aware weighting

    Args:
        classical_channels: CNN channel sizes, default [128, 256, 512]
        lstm_hidden_size: LSTM hidden units, default 256
        num_qubits: VQC qubits, default 12
        num_vqc_layers: VQC circuit depth, default 6
        fusion_method: 'adaptive' (protocol-aware) or 'fixed' (equal weighting)
        adversarial_defense: Enable Lipschitz constraints, default True

    Input:
        packet_sequences: (batch_size, max_packets=100, features=47)
        statistical_features: (batch_size, 12) - For quantum pathway

    Output:
        logits: (batch_size, 2) - Benign/malicious classification
        auxiliary: Dict with pathway outputs, fusion weights, protocol detection
    """

    def __init__(
        self,
        classical_channels: List[int] = [128, 256, 512],
        lstm_hidden_size: int = 256,
        num_qubits: int = 12,
        num_vqc_layers: int = 6,
        fusion_method: str = 'adaptive',
        adversarial_defense: bool = True
    ):
        super().__init__()

        self.fusion_method = fusion_method
        self.adversarial_defense = adversarial_defense

        # Protocol detection from first packet
        self.protocol_detector = ProtocolDetector(input_dim=47, hidden_dim=128)

        # Classical pathway: CNN + LSTM
        self.classical_pathway = ClassicalPathway(
            input_channels=47,
            cnn_channels=classical_channels,
            lstm_hidden_size=lstm_hidden_size,
            output_dim=512,
            use_spectral_norm=adversarial_defense
        )

        # Quantum pathway: VQC
        self.quantum_pathway = QuantumPathway(
            num_qubits=num_qubits,
            num_layers=num_vqc_layers,
            input_dim=12
        )

        # Adaptive fusion
        self.fusion = AdaptiveFusion(
            classical_dim=512,
            quantum_dim=1,
            hidden_dim=128
        )

        # Fixed fusion baseline (for ablation)
        if fusion_method == 'fixed':
            self.fixed_classifier = nn.Sequential(
                SpectralNormalization(nn.Linear(513, 256)),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(0.3),
                nn.Linear(256, 2)
            )

    def forward(
        self,
        packet_sequences: torch.Tensor,
        statistical_features: torch.Tensor,
        return_auxiliary: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass through hybrid architecture.

        Args:
            packet_sequences: (batch_size, max_packets, 47) - Packet-level features
            statistical_features: (batch_size, 12) - Connection-level statistics
            return_auxiliary: Return intermediate outputs for analysis

        Returns:
            logits: (batch_size, 2) - Classification logits
            auxiliary: Optional dict with pathway outputs, weights, etc.
        """
        batch_size = packet_sequences.size(0)

        # 1. Protocol Detection (from first packet)
        client_hello_features = packet_sequences[:, 0, :]  # (batch_size, 47)
        protocol_logits, protocol_probs = self.protocol_detector(client_hello_features)

        # 2. Classical Pathway
        classical_embedding = self.classical_pathway(packet_sequences)  # (batch_size, 512)

        # 3. Quantum Pathway
        quantum_prediction = self.quantum_pathway(statistical_features)  # (batch_size, 1)

        # 4. Fusion
        if self.fusion_method == 'adaptive':
            logits, fusion_weights = self.fusion(
                classical_embedding,
                quantum_prediction,
                protocol_probs
            )
        else:
            # Fixed fusion baseline (equal weighting)
            fused = torch.cat([classical_embedding, quantum_prediction], dim=-1)
            logits = self.fixed_classifier(fused)
            fusion_weights = torch.full((batch_size, 2), 0.5, device=logits.device)

        if not return_auxiliary:
            return logits, None

        # Return auxiliary information for analysis
        auxiliary = {
            'protocol_logits': protocol_logits,
            'protocol_probs': protocol_probs,
            'protocol_type': torch.argmax(protocol_probs, dim=-1),
            'classical_embedding': classical_embedding,
            'quantum_prediction': quantum_prediction,
            'fusion_weights': fusion_weights,
            'classical_weight': fusion_weights[:, 0].mean().item(),
            'quantum_weight': fusion_weights[:, 1].mean().item()
        }

        return logits, auxiliary

    def compute_pathway_accuracies(
        self,
        packet_sequences: torch.Tensor,
        statistical_features: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute individual pathway accuracies (for monitoring).

        Useful for tracking if one pathway is underperforming.

        Args:
            packet_sequences: (batch_size, max_packets, 47)
            statistical_features: (batch_size, 12)
            labels: (batch_size,) - Ground truth labels

        Returns:
            Dictionary with classical_acc, quantum_acc, fusion_acc
        """
        with torch.no_grad():
            # Get embeddings
            client_hello = packet_sequences[:, 0, :]
            _, protocol_probs = self.protocol_detector(client_hello)

            classical_embedding = self.classical_pathway(packet_sequences)
            quantum_prediction = self.quantum_pathway(statistical_features)

            # Classical pathway standalone prediction
            classical_logits = self.fusion.classifier(
                torch.cat([classical_embedding, torch.zeros_like(quantum_prediction)], dim=-1)
            )
            classical_preds = torch.argmax(classical_logits, dim=-1)
            classical_acc = (classical_preds == labels).float().mean().item()

            # Quantum pathway standalone prediction (threshold at 0.5)
            quantum_preds = (quantum_prediction.squeeze() > 0.5).long()
            quantum_acc = (quantum_preds == labels).float().mean().item()

            # Fused prediction
            logits, _ = self.forward(packet_sequences, statistical_features, return_auxiliary=False)
            fusion_preds = torch.argmax(logits, dim=-1)
            fusion_acc = (fusion_preds == labels).float().mean().item()

        return {
            'classical_acc': classical_acc,
            'quantum_acc': quantum_acc,
            'fusion_acc': fusion_acc
        }

    def enable_adversarial_defense(self):
        """Enable Lipschitz constraints (spectral normalization)."""
        self.adversarial_defense = True
        # Already applied via SpectralNormalization wrappers

    def disable_adversarial_defense(self):
        """Disable adversarial defense (for ablation studies)."""
        self.adversarial_defense = False
        # Would need to replace SpectralNormalization with regular Linear layers


def create_pq_idps(config: Dict) -> PQIDPS:
    """
    Factory function to create PQ-IDPS model from config.

    Args:
        config: Configuration dictionary with model hyperparameters

    Returns:
        Initialized PQ-IDPS model
    """
    model_config = config.get('model', {})

    return PQIDPS(
        classical_channels=model_config.get('classical_channels', [128, 256, 512]),
        lstm_hidden_size=model_config.get('lstm_hidden_size', 256),
        num_qubits=model_config.get('num_qubits', 12),
        num_vqc_layers=model_config.get('num_vqc_layers', 6),
        fusion_method=model_config.get('fusion_method', 'adaptive'),
        adversarial_defense=model_config.get('adversarial_defense', True)
    )


if __name__ == "__main__":
    """Test PQ-IDPS model architecture."""

    print("Testing PQ-IDPS Model...")
    print("=" * 60)

    # Create model
    model = PQIDPS(
        classical_channels=[128, 256, 512],
        lstm_hidden_size=256,
        num_qubits=12,
        num_vqc_layers=6,
        fusion_method='adaptive',
        adversarial_defense=True
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print()

    # Test forward pass
    batch_size = 8
    max_packets = 100
    packet_features = 47
    statistical_features_dim = 12

    # Create dummy inputs
    packet_sequences = torch.randn(batch_size, max_packets, packet_features)
    statistical_features = torch.randn(batch_size, statistical_features_dim)
    labels = torch.randint(0, 2, (batch_size,))

    # Forward pass
    logits, auxiliary = model(
        packet_sequences,
        statistical_features,
        return_auxiliary=True
    )

    print(f"Input Shapes:")
    print(f"  Packet Sequences: {packet_sequences.shape}")
    print(f"  Statistical Features: {statistical_features.shape}")
    print()

    print(f"Output Shapes:")
    print(f"  Logits: {logits.shape}")
    print(f"  Protocol Probs: {auxiliary['protocol_probs'].shape}")
    print(f"  Classical Embedding: {auxiliary['classical_embedding'].shape}")
    print(f"  Quantum Prediction: {auxiliary['quantum_prediction'].shape}")
    print(f"  Fusion Weights: {auxiliary['fusion_weights'].shape}")
    print()

    # Protocol distribution
    protocol_types = auxiliary['protocol_type']
    print(f"Detected Protocol Types:")
    print(f"  Classical ECDHE (0): {(protocol_types == 0).sum().item()}")
    print(f"  Hybrid X25519+MLKEM (1): {(protocol_types == 1).sum().item()}")
    print(f"  Pure MLKEM-768 (2): {(protocol_types == 2).sum().item()}")
    print()

    # Average fusion weights
    print(f"Average Fusion Weights:")
    print(f"  Classical: {auxiliary['classical_weight']:.3f}")
    print(f"  Quantum: {auxiliary['quantum_weight']:.3f}")
    print()

    # Compute pathway accuracies
    accuracies = model.compute_pathway_accuracies(
        packet_sequences,
        statistical_features,
        labels
    )

    print(f"Pathway Accuracies (random data):")
    print(f"  Classical: {accuracies['classical_acc']:.3f}")
    print(f"  Quantum: {accuracies['quantum_acc']:.3f}")
    print(f"  Fusion: {accuracies['fusion_acc']:.3f}")
    print()

    # Test backward pass
    loss = F.cross_entropy(logits, labels)
    loss.backward()

    print(f"Loss: {loss.item():.4f}")
    print(f"✓ Backward pass successful")
    print()

    print("=" * 60)
    print("✓ PQ-IDPS model test completed successfully!")
