"""
Classical Pathway: CNN-LSTM for TLS Traffic Analysis

Processes packet-level features from TLS 1.3 connections, including:
- Packet sizes and inter-arrival times
- TCP flags and window sizes
- TLS record types and content types
- Handshake message types
- Certificate chain lengths
- Cipher suite negotiations

Architecture:
    Conv1D(47 -> 128) -> Conv1D(128 -> 256) -> Conv1D(256 -> 512)
    -> MaxPool -> Bi-LSTM(256) -> Embedding(512)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math

from defense.lipschitz_constraints import SpectralNormalization


class Conv1DBlock(nn.Module):
    """
    1D Convolutional block with optional spectral normalization.

    Includes convolution, activation, batch normalization, and pooling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_spectral_norm: bool = True,
        pool: bool = False
    ):
        super().__init__()

        if use_spectral_norm:
            self.conv = SpectralNormalization(
                nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
            )
        else:
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)

        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2) if pool else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, in_channels, sequence_length)

        Returns:
            out: (batch_size, out_channels, sequence_length')
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        if self.pool is not None:
            x = self.pool(x)

        return x


class ClassicalPathway(nn.Module):
    """
    Classical pathway for PQ-IDPS using CNN + Bi-LSTM.

    Processes packet sequences to extract temporal and spatial patterns
    characteristic of malicious traffic in PQC-encrypted connections.

    Args:
        input_channels: Number of features per packet (default 47)
        cnn_channels: List of CNN output channels (default [128, 256, 512])
        lstm_hidden_size: LSTM hidden units (default 256)
        output_dim: Final embedding dimension (default 512)
        use_spectral_norm: Apply Lipschitz constraints (default True)

    Input:
        packet_sequences: (batch_size, max_packets, input_channels)

    Output:
        embedding: (batch_size, output_dim)
    """

    def __init__(
        self,
        input_channels: int = 47,
        cnn_channels: list = [128, 256, 512],
        lstm_hidden_size: int = 256,
        output_dim: int = 512,
        use_spectral_norm: bool = True
    ):
        super().__init__()

        self.input_channels = input_channels
        self.output_dim = output_dim

        # CNN layers for local pattern extraction
        self.cnn = nn.ModuleList()

        # First conv layer
        self.cnn.append(Conv1DBlock(
            in_channels=input_channels,
            out_channels=cnn_channels[0],
            kernel_size=5,
            padding=2,
            use_spectral_norm=use_spectral_norm,
            pool=False
        ))

        # Subsequent conv layers
        for i in range(len(cnn_channels) - 1):
            self.cnn.append(Conv1DBlock(
                in_channels=cnn_channels[i],
                out_channels=cnn_channels[i + 1],
                kernel_size=3,
                padding=1,
                use_spectral_norm=use_spectral_norm,
                pool=(i == len(cnn_channels) - 2)  # Pool after last conv
            ))

        # Bi-LSTM for temporal dependencies
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # Project LSTM output to embedding
        lstm_output_dim = lstm_hidden_size * 2  # Bidirectional
        if use_spectral_norm:
            self.projection = SpectralNormalization(
                nn.Linear(lstm_output_dim, output_dim)
            )
        else:
            self.projection = nn.Linear(lstm_output_dim, output_dim)

        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, packet_sequences: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN-LSTM.

        Args:
            packet_sequences: (batch_size, max_packets, input_channels)

        Returns:
            embedding: (batch_size, output_dim)
        """
        batch_size, max_packets, features = packet_sequences.shape

        # Transpose for Conv1D: (batch, channels, sequence)
        x = packet_sequences.transpose(1, 2)  # (batch_size, input_channels, max_packets)

        # CNN feature extraction
        for conv_block in self.cnn:
            x = conv_block(x)

        # Transpose back for LSTM: (batch, sequence, features)
        x = x.transpose(1, 2)  # (batch_size, sequence_length, cnn_channels[-1])

        # LSTM for temporal modeling
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use final hidden state from both directions
        # h_n shape: (num_layers * num_directions, batch, hidden_size)
        forward_hidden = h_n[-2, :, :]  # Last layer, forward direction
        backward_hidden = h_n[-1, :, :]  # Last layer, backward direction

        combined = torch.cat([forward_hidden, backward_hidden], dim=-1)

        # Project to output dimension
        embedding = self.projection(combined)
        embedding = self.layer_norm(embedding)
        embedding = self.dropout(embedding)

        return embedding

    def extract_packet_features(self, raw_packets: list) -> torch.Tensor:
        """
        Extract 47-dimensional features from raw packet data.

        Features include:
        - Packet sizes (payload, total)
        - Inter-arrival times
        - TCP flags (SYN, ACK, FIN, RST, PSH, URG)
        - TCP window size
        - TLS record type and version
        - TLS content type
        - Handshake message type
        - Certificate chain length
        - Cipher suite
        - Extensions length
        - Key exchange size (for PQC detection)
        - Signature size (for Dilithium detection)

        Args:
            raw_packets: List of packet dictionaries

        Returns:
            features: (num_packets, 47)
        """
        features_list = []

        prev_timestamp = None

        for pkt in raw_packets:
            feat = []

            # Size features (2)
            feat.append(pkt.get('payload_size', 0) / 1500.0)  # Normalize by MTU
            feat.append(pkt.get('total_size', 0) / 1500.0)

            # Timing features (1)
            timestamp = pkt.get('timestamp', 0)
            if prev_timestamp is not None:
                iat = timestamp - prev_timestamp
                feat.append(min(iat, 1.0))  # Cap at 1 second
            else:
                feat.append(0.0)
            prev_timestamp = timestamp

            # TCP flags (6)
            tcp_flags = pkt.get('tcp_flags', {})
            feat.append(float(tcp_flags.get('syn', False)))
            feat.append(float(tcp_flags.get('ack', False)))
            feat.append(float(tcp_flags.get('fin', False)))
            feat.append(float(tcp_flags.get('rst', False)))
            feat.append(float(tcp_flags.get('psh', False)))
            feat.append(float(tcp_flags.get('urg', False)))

            # TCP window (1)
            feat.append(pkt.get('tcp_window', 0) / 65535.0)  # Normalize

            # TLS features (32)
            tls = pkt.get('tls', {})

            # Record type and version (3)
            record_type = tls.get('record_type', 0)
            feat.extend(self._one_hot(record_type, num_classes=3))

            # Content type (4)
            content_type = tls.get('content_type', 0)
            feat.extend(self._one_hot(content_type, num_classes=4))

            # Handshake message type (8)
            handshake_type = tls.get('handshake_type', 0)
            feat.extend(self._one_hot(handshake_type, num_classes=8))

            # Certificate chain length (1)
            feat.append(min(tls.get('cert_chain_len', 0) / 10.0, 1.0))

            # Cipher suite (1 - just indicator)
            feat.append(float(tls.get('cipher_suite', 0) > 0))

            # Extensions length (1)
            feat.append(min(tls.get('extensions_len', 0) / 1000.0, 1.0))

            # Key exchange size (1) - Large values indicate PQC (Kyber: ~1184 bytes)
            kex_size = tls.get('key_exchange_size', 0)
            feat.append(min(kex_size / 2000.0, 1.0))

            # Signature size (1) - Large values indicate Dilithium (~3293 bytes)
            sig_size = tls.get('signature_size', 0)
            feat.append(min(sig_size / 4000.0, 1.0))

            # Server name indication length (1)
            feat.append(min(tls.get('sni_len', 0) / 100.0, 1.0))

            # ALPN length (1)
            feat.append(min(tls.get('alpn_len', 0) / 50.0, 1.0))

            # Supported groups count (1) - Indicates hybrid support
            feat.append(min(tls.get('supported_groups_count', 0) / 10.0, 1.0))

            # Signature algorithms count (1)
            feat.append(min(tls.get('sig_algs_count', 0) / 20.0, 1.0))

            # PQC indicators (2)
            feat.append(float(tls.get('has_kyber', False)))
            feat.append(float(tls.get('has_dilithium', False)))

            # Direction (1)
            feat.append(float(pkt.get('direction', 'outbound') == 'inbound'))

            # Statistical features (4)
            feat.append(min(pkt.get('entropy', 0) / 8.0, 1.0))
            feat.append(pkt.get('mean_byte_value', 127.5) / 255.0)
            feat.append(pkt.get('std_byte_value', 0) / 128.0)
            feat.append(min(pkt.get('compression_ratio', 1.0), 1.0))

            features_list.append(feat)

        features = torch.tensor(features_list, dtype=torch.float32)

        # Pad or truncate to ensure 47 features
        if features.size(1) < 47:
            padding = torch.zeros(features.size(0), 47 - features.size(1))
            features = torch.cat([features, padding], dim=1)
        elif features.size(1) > 47:
            features = features[:, :47]

        return features

    @staticmethod
    def _one_hot(value: int, num_classes: int) -> list:
        """Helper to create one-hot encoding."""
        one_hot = [0.0] * num_classes
        if 0 <= value < num_classes:
            one_hot[value] = 1.0
        return one_hot


if __name__ == "__main__":
    """Test Classical Pathway."""

    print("Testing Classical Pathway...")
    print("=" * 60)

    # Create model
    model = ClassicalPathway(
        input_channels=47,
        cnn_channels=[128, 256, 512],
        lstm_hidden_size=256,
        output_dim=512,
        use_spectral_norm=True
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}")
    print()

    # Test forward pass
    batch_size = 16
    max_packets = 100
    input_channels = 47

    packet_sequences = torch.randn(batch_size, max_packets, input_channels)

    print(f"Input Shape: {packet_sequences.shape}")

    embedding = model(packet_sequences)

    print(f"Output Shape: {embedding.shape}")
    print(f"Output Mean: {embedding.mean().item():.4f}")
    print(f"Output Std: {embedding.std().item():.4f}")
    print()

    # Test backward pass
    loss = embedding.sum()
    loss.backward()
    print("✓ Backward pass successful")
    print()

    print("=" * 60)
    print("✓ Classical Pathway test completed!")
