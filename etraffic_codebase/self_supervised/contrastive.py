"""
InfoNCE Contrastive Learning for Encrypted Traffic

Implements self-supervised pretraining that learns meaningful
representations from unlabeled encrypted traffic data.

Key result: Improves few-shot zero-day detection by 7.3 percentage
points over random initialization (Paper Section 3.6).

Architecture:
    1. Backbone encoder (CNN-LSTM or Transformer)
    2. Projection head (MLP -> low-dim embedding)
    3. InfoNCE loss for contrastive learning
    4. Traffic-specific augmentations (jitter, mask, permute)

References:
    Paper Section 3.6 - Self-Supervised Contrastive Pretraining
    Oord et al. (2018) - Contrastive Predictive Coding (InfoNCE)
    Chen et al. (2020) - SimCLR framework
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
from torch.utils.data import DataLoader


class TrafficAugmentation:
    """
    Data augmentation strategies for encrypted traffic flows.

    Augmentations preserve the semantic content while creating
    diverse views for contrastive learning:
    - Gaussian jitter on continuous features
    - Feature masking (random dropout)
    - Temporal permutation of packet subsequences
    - Feature scaling perturbation
    """

    def __init__(self, jitter_std: float = 0.1,
                 mask_ratio: float = 0.15,
                 permute_max_segments: int = 5,
                 scale_range: Tuple[float, float] = (0.8, 1.2)):
        self.jitter_std = jitter_std
        self.mask_ratio = mask_ratio
        self.permute_max_segments = permute_max_segments
        self.scale_range = scale_range

    def jitter(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to continuous features."""
        noise = torch.randn_like(x) * self.jitter_std
        return x + noise

    def mask(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly mask features."""
        mask = torch.bernoulli(
            torch.ones_like(x) * (1 - self.mask_ratio)
        )
        return x * mask

    def permute(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly permute temporal segments."""
        if x.dim() < 2:
            return x

        seq_len = x.size(0) if x.dim() == 2 else x.size(1)
        n_segments = min(self.permute_max_segments, seq_len)

        if n_segments < 2:
            return x

        segment_size = seq_len // n_segments
        segments = list(range(n_segments))
        np.random.shuffle(segments)

        if x.dim() == 2:
            permuted = torch.cat([
                x[s * segment_size:(s + 1) * segment_size]
                for s in segments
            ], dim=0)
            # Handle remainder
            if n_segments * segment_size < seq_len:
                permuted = torch.cat([
                    permuted, x[n_segments * segment_size:]
                ], dim=0)
        else:
            permuted = torch.cat([
                x[:, s * segment_size:(s + 1) * segment_size]
                for s in segments
            ], dim=1)
            if n_segments * segment_size < seq_len:
                permuted = torch.cat([
                    permuted, x[:, n_segments * segment_size:]
                ], dim=1)

        return permuted

    def scale(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random scaling."""
        factor = (
            torch.rand(1, device=x.device)
            * (self.scale_range[1] - self.scale_range[0])
            + self.scale_range[0]
        )
        return x * factor

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random augmentation."""
        aug_idx = np.random.randint(4)
        if aug_idx == 0:
            return self.jitter(x)
        elif aug_idx == 1:
            return self.mask(x)
        elif aug_idx == 2:
            return self.permute(x)
        else:
            return self.scale(x)


class InfoNCELoss(nn.Module):
    """
    InfoNCE (Noise Contrastive Estimation) loss.

    Maximizes agreement between positive pairs (augmented views
    of the same flow) while minimizing agreement with negative
    pairs (different flows).

    L = -log(exp(sim(z_i, z_j)/tau) / sum_k exp(sim(z_i, z_k)/tau))

    Reference: Oord et al. (2018) - CPC
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Compute InfoNCE loss.

        Args:
            z_i: Embeddings of view 1 (batch_size, embed_dim)
            z_j: Embeddings of view 2 (batch_size, embed_dim)

        Returns:
            Scalar loss
        """
        batch_size = z_i.size(0)

        # Normalize embeddings
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Concatenate
        representations = torch.cat([z_i, z_j], dim=0)  # (2N, D)

        # Similarity matrix
        similarity = torch.mm(
            representations, representations.t()
        ) / self.temperature  # (2N, 2N)

        # Create labels: positive pairs are (i, i+N) and (i+N, i)
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(0, batch_size)
        ]).to(z_i.device)

        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)
        similarity.masked_fill_(mask, -1e9)

        loss = F.cross_entropy(similarity, labels)
        return loss


class ContrastiveEncoder(nn.Module):
    """
    Encoder with projection head for contrastive pretraining.

    Architecture:
    - Backbone (shared with downstream task)
    - Projection head (discarded after pretraining)
    """

    def __init__(self, backbone: nn.Module, feature_dim: int = 256,
                 projection_dim: int = 128):
        """
        Args:
            backbone: Encoder backbone (CNN-LSTM or Transformer)
            feature_dim: Output dimension of backbone
            projection_dim: Output dimension of projection head
        """
        super().__init__()
        self.backbone = backbone

        # Projection head (MLP)
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, projection_dim)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (backbone_features, projected_embeddings)
        """
        # Get features from backbone
        if hasattr(self.backbone, 'forward') and 'return_features' in \
                self.backbone.forward.__code__.co_varnames:
            _, features = self.backbone(x, return_features=True)
        else:
            features = self.backbone(x)
            if features.dim() == 3:
                features = features.mean(dim=1)

        # Project
        projected = self.projection_head(features)

        return features, projected


class ContrastivePretrainer:
    """
    Self-supervised contrastive pretraining for encrypted traffic.

    Trains backbone encoder using InfoNCE loss on augmented views
    of unlabeled encrypted traffic data.

    After pretraining, the backbone can be fine-tuned on labeled
    data for downstream classification tasks.
    """

    def __init__(self, encoder: ContrastiveEncoder,
                 augmentation: TrafficAugmentation = None,
                 temperature: float = 0.07,
                 learning_rate: float = 0.001,
                 device: torch.device = None):
        self.encoder = encoder
        self.augmentation = augmentation or TrafficAugmentation()
        self.criterion = InfoNCELoss(temperature=temperature)
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.encoder.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.encoder.parameters(), lr=learning_rate, weight_decay=1e-4
        )

    def pretrain(self, dataloader: DataLoader, num_epochs: int = 100,
                 verbose: bool = True) -> list:
        """
        Run contrastive pretraining.

        Args:
            dataloader: DataLoader with unlabeled traffic data
            num_epochs: Number of pretraining epochs
            verbose: Print progress

        Returns:
            List of losses per epoch
        """
        self.encoder.train()
        losses = []

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch in dataloader:
                # Handle different batch formats
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch

                x = x.to(self.device)

                # Create two augmented views
                view1 = self._augment_batch(x)
                view2 = self._augment_batch(x)

                # Forward pass
                _, z1 = self.encoder(view1)
                _, z2 = self.encoder(view2)

                # Compute loss
                loss = self.criterion(z1, z2)

                # Backprop
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            losses.append(avg_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] "
                      f"Loss: {avg_loss:.4f}")

        return losses

    def _augment_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Apply augmentation to a batch."""
        augmented = []
        for i in range(x.size(0)):
            aug = self.augmentation(x[i])
            augmented.append(aug)
        return torch.stack(augmented)

    def get_backbone(self) -> nn.Module:
        """Return the pretrained backbone (without projection head)."""
        return self.encoder.backbone
