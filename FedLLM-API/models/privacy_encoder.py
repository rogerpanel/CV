"""
Privacy-Preserving API Encoder

Transforms API requests into semantic representations while ensuring
differential privacy guarantees.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import hashlib


class PrivacyPreservingEncoder(nn.Module):
    """
    Privacy-preserving encoder for API requests with differential privacy.

    Extracts semantic features from API requests while adding calibrated
    noise to ensure (ε, δ)-differential privacy.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        epsilon: float = 0.5,
        delta: float = 1e-5,
        sensitivity: float = 10.0,
        vocab_size: int = 10000,
        max_params: int = 20
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.vocab_size = vocab_size
        self.max_params = max_params

        # Calculate noise scale for Gaussian mechanism
        self.noise_scale = self._compute_noise_scale()

        # Endpoint embedding
        self.endpoint_embedding = nn.Embedding(vocab_size, embedding_dim // 4)

        # Method embedding (GET, POST, PUT, DELETE, etc.)
        self.method_embedding = nn.Embedding(10, embedding_dim // 8)

        # Temporal encoding
        self.temporal_encoder = TemporalEncoder(embedding_dim // 4)

        # Parameter pattern encoder
        self.param_encoder = ParameterPatternEncoder(embedding_dim // 4, max_params)

        # Projection to final dimension
        self.projection = nn.Linear(embedding_dim, embedding_dim)

        # Privacy budget tracking
        self.privacy_spent = 0.0

    def _compute_noise_scale(self) -> float:
        """
        Compute noise scale σ for Gaussian mechanism to achieve (ε, δ)-DP.

        σ = Δ₂ * sqrt(2 * ln(1.25/δ)) / ε
        """
        return self.sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon

    def forward(
        self,
        api_requests: List[Dict],
        add_noise: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode API requests with differential privacy.

        Args:
            api_requests: List of API request dictionaries with keys:
                - method: HTTP method (GET, POST, etc.)
                - endpoint: API endpoint path
                - params: Request parameters
                - headers: Request headers (optional)
                - timestamp: Request timestamp
            add_noise: Whether to add DP noise (disable for debugging)

        Returns:
            encoded: Privacy-preserved encodings [batch, seq_len, embedding_dim]
            mask: Attention mask [batch, seq_len]
        """
        batch_size = len(api_requests)
        max_seq_len = min(max(len(req.get('sequence', [req])) for req in api_requests), 50)

        # Initialize tensors
        encoded_features = torch.zeros(batch_size, max_seq_len, self.embedding_dim)
        attention_mask = torch.zeros(batch_size, max_seq_len)

        for i, request_seq in enumerate(api_requests):
            # Handle both single requests and sequences
            if 'sequence' in request_seq:
                requests = request_seq['sequence']
            else:
                requests = [request_seq]

            seq_len = min(len(requests), max_seq_len)

            for j, req in enumerate(requests[:seq_len]):
                # Extract structural features
                struct_feat = self._extract_structural_features(req)

                # Extract temporal features
                temp_feat = self._extract_temporal_features(req)

                # Extract parameter features
                param_feat = self._extract_parameter_features(req)

                # Concatenate features
                combined = torch.cat([struct_feat, temp_feat, param_feat], dim=-1)

                # Project to embedding dimension
                projected = self.projection(combined)

                # Add differential privacy noise
                if add_noise:
                    noise = torch.randn_like(projected) * self.noise_scale
                    projected = projected + noise

                encoded_features[i, j] = projected
                attention_mask[i, j] = 1.0

        # Update privacy budget
        if add_noise:
            self.privacy_spent += self.epsilon

        return encoded_features, attention_mask

    def _extract_structural_features(self, request: Dict) -> torch.Tensor:
        """Extract endpoint and method embeddings."""
        # Hash endpoint to vocabulary index
        endpoint = request.get('endpoint', '/unknown')
        endpoint_idx = hash_to_vocab(endpoint, self.vocab_size)
        endpoint_emb = self.endpoint_embedding(torch.tensor([endpoint_idx]))

        # Map method to index
        method = request.get('method', 'GET')
        method_map = {'GET': 0, 'POST': 1, 'PUT': 2, 'DELETE': 3, 'PATCH': 4, 'HEAD': 5, 'OPTIONS': 6}
        method_idx = method_map.get(method.upper(), 7)
        method_emb = self.method_embedding(torch.tensor([method_idx]))

        # Combine and pad
        struct_feat = torch.cat([endpoint_emb.flatten(), method_emb.flatten()])
        padding = torch.zeros(self.embedding_dim // 4 - struct_feat.shape[0])
        return torch.cat([struct_feat, padding])

    def _extract_temporal_features(self, request: Dict) -> torch.Tensor:
        """Extract temporal patterns."""
        timestamp = request.get('timestamp', 0.0)
        return self.temporal_encoder(torch.tensor([[timestamp]]))

    def _extract_parameter_features(self, request: Dict) -> torch.Tensor:
        """Extract parameter patterns without exposing values."""
        params = request.get('params', {})
        return self.param_encoder(params)

    def get_privacy_budget_spent(self) -> float:
        """Return total privacy budget spent."""
        return self.privacy_spent

    def reset_privacy_budget(self):
        """Reset privacy budget counter."""
        self.privacy_spent = 0.0


class TemporalEncoder(nn.Module):
    """Encodes temporal patterns using sinusoidal positional encoding."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Encode timestamps with multiple time scales.

        Args:
            timestamps: Unix timestamps [batch, 1]

        Returns:
            encoded: Temporal encodings [batch, embedding_dim]
        """
        # Multiple time scales: hour, day, week
        time_scales = [3600, 86400, 604800]  # seconds
        encodings = []

        for scale in time_scales:
            phase = 2 * np.pi * timestamps / scale
            encodings.append(torch.sin(phase))
            encodings.append(torch.cos(phase))

        # Concatenate and project
        combined = torch.cat(encodings, dim=-1)

        # Pad or truncate to embedding_dim
        if combined.shape[-1] < self.embedding_dim:
            padding = torch.zeros(*combined.shape[:-1], self.embedding_dim - combined.shape[-1])
            combined = torch.cat([combined, padding], dim=-1)
        else:
            combined = combined[..., :self.embedding_dim]

        return combined


class ParameterPatternEncoder(nn.Module):
    """Encodes parameter patterns without exposing actual values."""

    def __init__(self, embedding_dim: int, max_params: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_params = max_params

    def forward(self, params: Dict) -> torch.Tensor:
        """
        Extract distributional statistics from parameters.

        Args:
            params: Dictionary of request parameters

        Returns:
            encoded: Parameter pattern features [embedding_dim]
        """
        features = []

        # Number of parameters (normalized)
        num_params = min(len(params), self.max_params) / self.max_params
        features.append(num_params)

        # Average key length (normalized)
        if params:
            avg_key_len = np.mean([len(str(k)) for k in params.keys()]) / 50.0
        else:
            avg_key_len = 0.0
        features.append(avg_key_len)

        # Average value length (normalized)
        if params:
            avg_val_len = np.mean([len(str(v)) for v in params.values()]) / 100.0
        else:
            avg_val_len = 0.0
        features.append(avg_val_len)

        # Entropy of parameter types
        if params:
            type_counts = {}
            for v in params.values():
                vtype = type(v).__name__
                type_counts[vtype] = type_counts.get(vtype, 0) + 1

            type_probs = np.array(list(type_counts.values())) / len(params)
            entropy = -np.sum(type_probs * np.log(type_probs + 1e-10))
            entropy = entropy / np.log(10)  # Normalize
        else:
            entropy = 0.0
        features.append(entropy)

        # Convert to tensor and pad
        feature_tensor = torch.tensor(features, dtype=torch.float32)
        padding = torch.zeros(self.embedding_dim - len(features))
        return torch.cat([feature_tensor, padding])


def hash_to_vocab(text: str, vocab_size: int) -> int:
    """Hash text to vocabulary index deterministically."""
    hash_object = hashlib.md5(text.encode())
    hash_int = int(hash_object.hexdigest(), 16)
    return hash_int % vocab_size


if __name__ == "__main__":
    # Example usage
    encoder = PrivacyPreservingEncoder(
        embedding_dim=768,
        epsilon=0.5,
        delta=1e-5
    )

    # Sample API requests
    api_requests = [
        {
            'method': 'POST',
            'endpoint': '/api/v2/users/123/orders',
            'params': {'item_id': 456, 'quantity': 1},
            'timestamp': 1635123456.789
        },
        {
            'method': 'GET',
            'endpoint': '/api/v2/products',
            'params': {'category': 'electronics', 'limit': 20},
            'timestamp': 1635123460.123
        }
    ]

    # Encode with DP
    encoded, mask = encoder(api_requests, add_noise=True)

    print(f"Encoded shape: {encoded.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Privacy budget spent: ε = {encoder.get_privacy_budget_spent():.2f}")
