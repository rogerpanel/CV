"""
Prototypical Networks for Few-Shot Encrypted Traffic Classification

Learns an embedding space where traffic classes cluster around prototypes,
enabling classification through distance metrics with minimal labeled samples.

Achieves 93-98.5% accuracy on 5-way 5-shot encrypted traffic classification.

Reference:
    Paper Section 3.7 - Few-Shot Learning
    Snell et al. (2017) - Prototypical Networks for Few-Shot Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List


class PrototypicalEmbedding(nn.Module):
    """
    Embedding network for Prototypical Networks.

    Maps encrypted traffic features to a low-dimensional embedding
    space with L2 normalization for distance-based classification.
    """

    def __init__(self, input_dim: int = 88, hidden_dim: int = 256,
                 embedding_dim: int = 128, num_layers: int = 3,
                 dropout: float = 0.2):
        super().__init__()

        layers = []
        in_dim = input_dim
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, embedding_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.mean(dim=1)  # Average over sequence
        embeddings = self.network(x)
        return F.normalize(embeddings, p=2, dim=-1)


class PrototypicalNetwork(nn.Module):
    """
    Prototypical Network for few-shot encrypted traffic classification.

    Computes class prototypes as mean embeddings of support set samples,
    then classifies queries by distance to nearest prototype.
    """

    def __init__(self, embedding: PrototypicalEmbedding,
                 distance: str = 'euclidean'):
        super().__init__()
        self.embedding = embedding
        self.distance = distance

    def compute_prototypes(self, support: torch.Tensor,
                           support_labels: torch.Tensor) -> torch.Tensor:
        """Compute class prototypes from support set."""
        classes = torch.unique(support_labels)
        prototypes = []
        for c in classes:
            mask = (support_labels == c)
            prototype = support[mask].mean(dim=0)
            prototypes.append(prototype)
        return torch.stack(prototypes)

    def forward(self, support: torch.Tensor, support_labels: torch.Tensor,
                query: torch.Tensor) -> torch.Tensor:
        """
        Args:
            support: Support set (n_way * k_shot, input_dim)
            support_labels: Support labels (n_way * k_shot,)
            query: Query set (n_query, input_dim)

        Returns:
            Log-probabilities (n_query, n_way)
        """
        support_emb = self.embedding(support)
        query_emb = self.embedding(query)

        prototypes = self.compute_prototypes(support_emb, support_labels)

        # Compute distances
        if self.distance == 'euclidean':
            dists = torch.cdist(query_emb, prototypes, p=2)
            return -dists  # Negative distance as logits
        elif self.distance == 'cosine':
            similarity = torch.mm(
                query_emb, prototypes.t()
            )
            return similarity
        else:
            raise ValueError(f"Unknown distance: {self.distance}")


class PrototypicalTrainer:
    """
    Episodic trainer for Prototypical Networks.

    Implements N-way K-shot training with episodes sampled from the dataset.
    """

    def __init__(self, model: PrototypicalNetwork,
                 n_way: int = 5, k_shot: int = 5,
                 n_query: int = 15,
                 learning_rate: float = 0.001,
                 device: torch.device = None):
        self.model = model
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate
        )

    def sample_episode(self, features: np.ndarray,
                       labels: np.ndarray
                       ) -> Tuple[torch.Tensor, torch.Tensor,
                                  torch.Tensor, torch.Tensor]:
        """Sample an N-way K-shot episode."""
        unique_classes = np.unique(labels)
        selected = np.random.choice(
            unique_classes, self.n_way, replace=False
        )

        support_x, support_y = [], []
        query_x, query_y = [], []

        for i, c in enumerate(selected):
            mask = labels == c
            indices = np.where(mask)[0]
            chosen = np.random.choice(
                indices, self.k_shot + self.n_query, replace=False
            )

            support_x.append(features[chosen[:self.k_shot]])
            support_y.extend([i] * self.k_shot)
            query_x.append(features[chosen[self.k_shot:]])
            query_y.extend([i] * self.n_query)

        return (
            torch.FloatTensor(np.vstack(support_x)).to(self.device),
            torch.LongTensor(support_y).to(self.device),
            torch.FloatTensor(np.vstack(query_x)).to(self.device),
            torch.LongTensor(query_y).to(self.device),
        )

    def train(self, features: np.ndarray, labels: np.ndarray,
              num_episodes: int = 1000, val_features: np.ndarray = None,
              val_labels: np.ndarray = None,
              verbose: bool = True) -> List[float]:
        """Train with episodic learning."""
        self.model.train()
        losses = []

        for ep in range(num_episodes):
            support_x, support_y, query_x, query_y = self.sample_episode(
                features, labels
            )

            logits = self.model(support_x, support_y, query_x)
            loss = F.cross_entropy(logits, query_y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())

            if verbose and (ep + 1) % 100 == 0:
                acc = (logits.argmax(dim=1) == query_y).float().mean()
                print(f"Episode [{ep+1}/{num_episodes}] "
                      f"Loss: {loss.item():.4f} Acc: {acc.item():.4f}")

        return losses
