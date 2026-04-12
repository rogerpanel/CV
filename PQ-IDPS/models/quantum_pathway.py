"""
Quantum Pathway: Variational Quantum Classifier (VQC)

Processes statistical features using parameterized quantum circuits.
Designed to capture quantum-resistant patterns in PQC traffic that
classical models struggle to learn.

Circuit Architecture:
    - Angle encoding: |ψ(x)⟩ = ⊗ᵢ Ry(xᵢ)|0⟩
    - Variational layers: Ry(θ), Rz(φ) + CNOT entanglement
    - Measurement: Pauli-Z expectation on all qubits
    - Post-processing: Sigmoid for binary classification

Supports both quantum simulation and hardware execution.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List
import pennylane as qml


class QuantumPathway(nn.Module):
    """
    Quantum pathway using Variational Quantum Circuit.

    Args:
        num_qubits: Number of qubits (default 12)
        num_layers: Circuit depth (default 6)
        input_dim: Input feature dimension (default 12)
        device_type: 'default.qubit' (CPU) or 'lightning.gpu' (GPU)
        noise_channels: Apply quantum noise for adversarial robustness

    Input:
        statistical_features: (batch_size, input_dim)

    Output:
        prediction: (batch_size, 1) - Sigmoid output in [0, 1]
    """

    def __init__(
        self,
        num_qubits: int = 12,
        num_layers: int = 6,
        input_dim: int = 12,
        device_type: str = 'default.qubit',
        noise_channels: bool = False
    ):
        super().__init__()

        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.noise_channels = noise_channels

        # Validate dimensions
        if input_dim != num_qubits:
            raise ValueError(f"input_dim ({input_dim}) must match num_qubits ({num_qubits})")

        # Create quantum device
        if device_type == 'lightning.gpu' and qml.device('lightning.gpu', wires=1):
            try:
                self.dev = qml.device('lightning.gpu', wires=num_qubits)
            except:
                print("Warning: lightning.gpu not available, falling back to default.qubit")
                self.dev = qml.device('default.qubit', wires=num_qubits)
        else:
            self.dev = qml.device('default.qubit', wires=num_qubits)

        # Create quantum circuit
        self.qnode = qml.QNode(self._circuit, self.dev, interface='torch', diff_method='backprop')

        # Variational parameters
        # Each layer has 2 rotation gates per qubit
        num_params = num_qubits * 2 * num_layers
        self.params = nn.Parameter(torch.randn(num_params) * 0.1)

        # Post-processing for binary classification
        self.post_process = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid()
        )

    def _circuit(self, inputs: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Quantum circuit implementation.

        Args:
            inputs: (batch_size, num_qubits) - Input features
            params: (num_params,) - Variational parameters

        Returns:
            measurement: (batch_size,) - Pauli-Z expectation
        """
        batch_size = inputs.shape[0] if len(inputs.shape) > 1 else 1

        # Angle encoding (data embedding)
        for i in range(self.num_qubits):
            qml.RY(inputs[i] * np.pi, wires=i)

        # Variational layers
        param_idx = 0
        for layer in range(self.num_layers):
            # Rotation gates
            for i in range(self.num_qubits):
                qml.RY(params[param_idx], wires=i)
                param_idx += 1

            for i in range(self.num_qubits):
                qml.RZ(params[param_idx], wires=i)
                param_idx += 1

            # Entanglement layer (CNOT ladder)
            for i in range(self.num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

            # Wrap-around entanglement
            if self.num_qubits > 2:
                qml.CNOT(wires=[self.num_qubits - 1, 0])

            # Apply noise channels if enabled (for adversarial defense)
            if self.noise_channels and self.training:
                for i in range(self.num_qubits):
                    # Depolarizing noise (p=0.01)
                    qml.DepolarizingChannel(0.01, wires=i)

                    # Amplitude damping (γ=0.05)
                    qml.AmplitudeDamping(0.05, wires=i)

                    # Phase damping (λ=0.03)
                    qml.PhaseDamping(0.03, wires=i)

        # Measurement: Pauli-Z on all qubits
        observables = [qml.PauliZ(i) for i in range(self.num_qubits)]
        measurement = qml.expval(qml.sum(*observables))

        return measurement

    def forward(self, statistical_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through quantum circuit.

        Args:
            statistical_features: (batch_size, input_dim)

        Returns:
            prediction: (batch_size, 1)
        """
        batch_size = statistical_features.size(0)

        # Normalize inputs to [-1, 1] for angle encoding
        normalized = torch.tanh(statistical_features)

        # Process each sample through quantum circuit
        measurements = []
        for i in range(batch_size):
            sample = normalized[i]
            measurement = self.qnode(sample, self.params)
            measurements.append(measurement)

        measurements = torch.stack(measurements).unsqueeze(-1)  # (batch_size, 1)

        # Post-process: normalize measurement and apply sigmoid
        # Measurement ranges from -num_qubits to +num_qubits
        normalized_measurement = measurements / self.num_qubits

        prediction = self.post_process(normalized_measurement)

        return prediction

    def enable_noise_channels(self):
        """Enable quantum noise channels for adversarial defense."""
        self.noise_channels = True

    def disable_noise_channels(self):
        """Disable quantum noise channels."""
        self.noise_channels = False

    def get_circuit_string(self) -> str:
        """
        Get string representation of quantum circuit.

        Useful for visualization and debugging.
        """
        sample_input = torch.zeros(self.num_qubits)
        drawer = qml.draw(self.qnode, show_all_wires=True)
        return drawer(sample_input, self.params)

    def extract_statistical_features(self, packet_sequences: torch.Tensor) -> torch.Tensor:
        """
        Extract 12-dimensional statistical features from packet sequences.

        Features designed to capture distribution characteristics that
        differ between classical and PQC traffic.

        Args:
            packet_sequences: (batch_size, max_packets, 47)

        Returns:
            statistical_features: (batch_size, 12)
        """
        batch_size = packet_sequences.size(0)
        features_list = []

        for i in range(batch_size):
            sequence = packet_sequences[i]  # (max_packets, 47)

            # Remove padding (zero packets)
            non_zero_mask = sequence.abs().sum(dim=1) > 0
            valid_sequence = sequence[non_zero_mask]

            if valid_sequence.size(0) == 0:
                # Handle empty sequence
                features_list.append(torch.zeros(12))
                continue

            feat = []

            # 1. Mean packet size
            sizes = valid_sequence[:, 0]  # First feature is normalized size
            feat.append(sizes.mean())

            # 2. Std packet size
            feat.append(sizes.std() if sizes.size(0) > 1 else torch.tensor(0.0))

            # 3. Max packet size
            feat.append(sizes.max())

            # 4. Mean inter-arrival time
            iats = valid_sequence[:, 2]  # Third feature is IAT
            feat.append(iats.mean())

            # 5. Std inter-arrival time
            feat.append(iats.std() if iats.size(0) > 1 else torch.tensor(0.0))

            # 6. Ratio of large packets (>MTU indicator for PQC)
            large_packets = (sizes > 0.8).float().mean()  # Normalized by 1500
            feat.append(large_packets)

            # 7. Handshake complexity (count of unique handshake types)
            handshake_features = valid_sequence[:, 13:21]  # One-hot handshake types
            handshake_types = handshake_features.sum(dim=0)
            handshake_complexity = (handshake_types > 0).float().sum() / 8.0
            feat.append(handshake_complexity)

            # 8. Key exchange size indicator
            kex_sizes = valid_sequence[:, 39]  # Key exchange size feature
            feat.append(kex_sizes.max())  # Max indicates PQC presence

            # 9. Signature size indicator
            sig_sizes = valid_sequence[:, 40]  # Signature size feature
            feat.append(sig_sizes.max())  # Max indicates Dilithium

            # 10. Entropy of packet sizes
            size_hist, _ = torch.histogram(sizes, bins=10, range=(0.0, 1.0))
            size_probs = size_hist.float() / (size_hist.sum() + 1e-8)
            entropy = -(size_probs * torch.log(size_probs + 1e-8)).sum()
            feat.append(entropy / np.log(10))  # Normalize by max entropy

            # 11. Burst ratio (consecutive packets with small IAT)
            burst_mask = iats < 0.01  # Less than 10ms
            burst_ratio = burst_mask.float().mean()
            feat.append(burst_ratio)

            # 12. Connection duration estimate
            total_time = iats.sum()
            feat.append(torch.clamp(total_time, 0.0, 1.0))

            features_list.append(torch.stack(feat))

        statistical_features = torch.stack(features_list)

        return statistical_features


def create_vqc_circuit_diagram(num_qubits: int = 12, num_layers: int = 6) -> str:
    """
    Create ASCII diagram of VQC circuit for documentation.

    Args:
        num_qubits: Number of qubits
        num_layers: Number of variational layers

    Returns:
        circuit_diagram: ASCII representation
    """
    dev = qml.device('default.qubit', wires=num_qubits)

    @qml.qnode(dev)
    def circuit(inputs, params):
        # Encoding
        for i in range(num_qubits):
            qml.RY(inputs[i], wires=i)

        # Single variational layer for visualization
        param_idx = 0
        for i in range(num_qubits):
            qml.RY(params[param_idx], wires=i)
            param_idx += 1

        for i in range(num_qubits):
            qml.RZ(params[param_idx], wires=i)
            param_idx += 1

        for i in range(num_qubits - 1):
            qml.CNOT(wires=[i, i + 1])

        return qml.expval(qml.PauliZ(0))

    # Create sample inputs
    inputs = np.zeros(num_qubits)
    params = np.zeros(num_qubits * 2)

    drawer = qml.draw(circuit, show_all_wires=True)
    return drawer(inputs, params)


if __name__ == "__main__":
    """Test Quantum Pathway."""

    print("Testing Quantum Pathway...")
    print("=" * 60)

    # Create model
    model = QuantumPathway(
        num_qubits=12,
        num_layers=6,
        input_dim=12,
        device_type='default.qubit',
        noise_channels=False
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}")
    print(f"Quantum Circuit Parameters: {model.params.numel():,}")
    print(f"Post-processing Parameters: {total_params - model.params.numel():,}")
    print()

    # Test forward pass
    batch_size = 8
    input_dim = 12

    statistical_features = torch.randn(batch_size, input_dim)

    print(f"Input Shape: {statistical_features.shape}")

    prediction = model(statistical_features)

    print(f"Output Shape: {prediction.shape}")
    print(f"Output Range: [{prediction.min().item():.4f}, {prediction.max().item():.4f}]")
    print(f"Output Mean: {prediction.mean().item():.4f}")
    print()

    # Test backward pass
    loss = prediction.sum()
    loss.backward()
    print("✓ Backward pass successful")
    print()

    # Test feature extraction
    packet_sequences = torch.randn(4, 100, 47)
    extracted_features = model.extract_statistical_features(packet_sequences)
    print(f"Extracted Statistical Features Shape: {extracted_features.shape}")
    print(f"Feature Means: {extracted_features.mean(dim=0)[:4]}")
    print()

    # Print circuit diagram
    print("Circuit Diagram (1 layer shown):")
    print(create_vqc_circuit_diagram(num_qubits=4, num_layers=1))
    print()

    # Test with noise channels
    model.enable_noise_channels()
    model.train()
    prediction_noisy = model(statistical_features)
    print(f"Prediction with Noise: Mean={prediction_noisy.mean().item():.4f}, "
          f"Std={prediction_noisy.std().item():.4f}")
    print()

    print("=" * 60)
    print("✓ Quantum Pathway test completed!")
