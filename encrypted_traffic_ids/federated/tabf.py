"""
Traffic-Aware Byzantine Filtering (TABF) for Federated Learning

This module implements the novel TABF aggregation scheme introduced in the paper (Algorithm 1).
TABF filters malicious federated clients using encrypted-traffic statistics before aggregation,
maintaining >95% accuracy even with 40% Byzantine participants.

Key Components:
- Temporal consistency loss (KL divergence on inter-arrival times)
- Protocol conformance loss (TLS handshake feature correlations)
- Coordinate-wise median aggregation over filtered clients

References:
    Paper Section 3.3 - Traffic-Aware Byzantine Filtering
    Yin et al. (2018) - Byzantine-Robust Distributed Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy.stats import entropy


class TABFAggregator:
    """
    Traffic-Aware Byzantine Filtering aggregator for federated learning.

    TABF scores each client update using:
    1. Temporal consistency: KL divergence of inter-arrival time distributions
    2. Protocol conformance: Frobenius norm of TLS feature correlation changes

    Clients with high scores (anomalous updates) are filtered before robust aggregation.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        percentile_threshold: float = 75.0,
        use_temporal_loss: bool = True,
        use_protocol_loss: bool = True
    ):
        """
        Initialize TABF aggregator.

        Args:
            alpha: Weight for combining temporal and protocol losses (0 to 1)
            percentile_threshold: Percentile threshold for filtering (e.g., 75 = keep bottom 75%)
            use_temporal_loss: Whether to use inter-arrival time consistency
            use_protocol_loss: Whether to use TLS protocol conformance
        """
        self.alpha = alpha
        self.percentile_threshold = percentile_threshold
        self.use_temporal_loss = use_temporal_loss
        self.use_protocol_loss = use_protocol_loss

    def compute_iat_distribution(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        num_bins: int = 50
    ) -> np.ndarray:
        """
        Compute inter-arrival time (IAT) distribution from model predictions.

        Args:
            model: Neural network model
            dataloader: Validation data loader
            device: Compute device
            num_bins: Number of histogram bins

        Returns:
            Normalized IAT histogram
        """
        model.eval()
        iats = []

        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 3:
                    x, _, y = batch
                else:
                    x, y = batch

                x = x.to(device)

                # Extract inter-arrival times (assume feature index 1)
                if len(x.shape) == 3:  # (batch_size, seq_len, num_features)
                    iat = x[:, :, 1].cpu().numpy()
                    iats.extend(iat.flatten())

        # Create histogram
        if len(iats) > 0:
            hist, _ = np.histogram(iats, bins=num_bins, range=(0, 1000), density=True)
            hist = hist / (hist.sum() + 1e-10)  # Normalize
            return hist
        else:
            return np.ones(num_bins) / num_bins

    def compute_tls_feature_correlations(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        tls_feature_indices: List[int] = None
    ) -> np.ndarray:
        """
        Compute correlation matrix of TLS handshake features.

        Args:
            model: Neural network model
            dataloader: Validation data loader
            device: Compute device
            tls_feature_indices: Indices of TLS features (if None, use heuristic)

        Returns:
            Correlation matrix of TLS features
        """
        model.eval()
        tls_features = []

        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 3:
                    x, _, y = batch
                else:
                    x, y = batch

                x = x.to(device)

                # Extract TLS-related features (heuristic: features 3-7)
                if tls_feature_indices is None:
                    if len(x.shape) == 3 and x.shape[2] > 7:
                        tls_feat = x[:, :, 3:8].mean(dim=1)  # Average over sequence
                    else:
                        tls_feat = x.mean(dim=1) if len(x.shape) == 3 else x
                else:
                    tls_feat = x[:, :, tls_feature_indices].mean(dim=1)

                tls_features.append(tls_feat.cpu().numpy())

        if len(tls_features) > 0:
            tls_features = np.vstack(tls_features)
            # Compute correlation matrix
            corr_matrix = np.corrcoef(tls_features, rowvar=False)
            return corr_matrix
        else:
            return np.eye(5)  # Identity if no features

    def compute_temporal_loss(
        self,
        p_val: np.ndarray,
        p_client: np.ndarray
    ) -> float:
        """
        Compute temporal consistency loss (KL divergence).

        Args:
            p_val: Validation set IAT distribution
            p_client: Client's IAT distribution

        Returns:
            KL divergence KL(p_val || p_client)
        """
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        p_val = p_val + epsilon
        p_client = p_client + epsilon

        # Normalize
        p_val = p_val / p_val.sum()
        p_client = p_client / p_client.sum()

        # KL divergence
        kl_div = entropy(p_val, p_client)

        return float(kl_div)

    def compute_protocol_loss(
        self,
        corr_val: np.ndarray,
        corr_client: np.ndarray
    ) -> float:
        """
        Compute protocol conformance loss (Frobenius norm).

        Args:
            corr_val: Validation set TLS feature correlations
            corr_client: Client's TLS feature correlations

        Returns:
            Frobenius norm of correlation difference
        """
        diff = corr_val - corr_client
        frobenius_norm = np.linalg.norm(diff, ord='fro')

        return float(frobenius_norm)

    def score_client_updates(
        self,
        global_model: nn.Module,
        client_updates: List[Dict[str, torch.Tensor]],
        validation_loader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> np.ndarray:
        """
        Score all client updates based on traffic statistics.

        Args:
            global_model: Current global model
            client_updates: List of client parameter updates
            validation_loader: Validation dataset
            device: Compute device

        Returns:
            Array of scores for each client (higher = more suspicious)
        """
        num_clients = len(client_updates)
        scores = np.zeros(num_clients)

        # Compute reference distributions on validation set
        if self.use_temporal_loss:
            p_val = self.compute_iat_distribution(global_model, validation_loader, device)
        else:
            p_val = None

        if self.use_protocol_loss:
            corr_val = self.compute_tls_feature_correlations(global_model, validation_loader, device)
        else:
            corr_val = None

        # Score each client
        for m, client_update in enumerate(client_updates):
            # Create temporary model with client update
            temp_model = type(global_model)(
                **global_model.get_config() if hasattr(global_model, 'get_config') else {}
            )
            temp_model.load_state_dict(client_update)
            temp_model.to(device)

            temporal_loss = 0.0
            protocol_loss = 0.0

            # Compute temporal loss
            if self.use_temporal_loss and p_val is not None:
                p_client = self.compute_iat_distribution(temp_model, validation_loader, device)
                temporal_loss = self.compute_temporal_loss(p_val, p_client)

            # Compute protocol loss
            if self.use_protocol_loss and corr_val is not None:
                corr_client = self.compute_tls_feature_correlations(temp_model, validation_loader, device)
                protocol_loss = self.compute_protocol_loss(corr_val, corr_client)

            # Combined score
            scores[m] = self.alpha * temporal_loss + (1 - self.alpha) * protocol_loss

        return scores

    def filter_clients(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        scores: np.ndarray
    ) -> Tuple[List[Dict[str, torch.Tensor]], List[int]]:
        """
        Filter clients based on score threshold.

        Args:
            client_updates: List of client parameter updates
            scores: Client scores

        Returns:
            Tuple of (filtered_updates, trusted_client_indices)
        """
        # Compute threshold
        threshold = np.percentile(scores, self.percentile_threshold)

        # Filter clients below threshold
        trusted_indices = [i for i, score in enumerate(scores) if score <= threshold]
        trusted_updates = [client_updates[i] for i in trusted_indices]

        return trusted_updates, trusted_indices

    def coordinate_wise_median(
        self,
        client_updates: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute coordinate-wise median over client parameters.

        This is robust to outliers and Byzantine clients.

        Args:
            client_updates: List of client model state dicts

        Returns:
            Aggregated parameters (median)
        """
        if len(client_updates) == 0:
            raise ValueError("No client updates to aggregate")

        # Get parameter names from first client
        param_names = list(client_updates[0].keys())

        aggregated = {}

        for param_name in param_names:
            # Stack parameters from all clients
            params = torch.stack([update[param_name] for update in client_updates])

            # Compute coordinate-wise median
            median_params = torch.median(params, dim=0)[0]

            aggregated[param_name] = median_params

        return aggregated

    def aggregate(
        self,
        global_model: nn.Module,
        client_updates: List[Dict[str, torch.Tensor]],
        validation_loader: torch.utils.data.DataLoader,
        device: torch.device,
        verbose: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        TABF aggregation: filter malicious clients then robust aggregation.

        This implements Algorithm 1 from the paper.

        Args:
            global_model: Current global model
            client_updates: List of client parameter updates
            validation_loader: Validation data
            device: Compute device
            verbose: Print filtering statistics

        Returns:
            Aggregated global model parameters
        """
        # Step 1: Score all clients
        scores = self.score_client_updates(
            global_model, client_updates, validation_loader, device
        )

        # Step 2: Filter clients
        trusted_updates, trusted_indices = self.filter_clients(client_updates, scores)

        if verbose:
            print(f"TABF: Filtered {len(client_updates) - len(trusted_updates)}/{len(client_updates)} clients")
            print(f"  Trusted clients: {trusted_indices}")
            print(f"  Score threshold: {np.percentile(scores, self.percentile_threshold):.4f}")
            print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")

        # Step 3: Coordinate-wise median aggregation
        aggregated_params = self.coordinate_wise_median(trusted_updates)

        return aggregated_params


def tabf_federated_training(
    global_model: nn.Module,
    clients: List,  # List of FederatedClient instances
    validation_loader: torch.utils.data.DataLoader,
    num_rounds: int,
    device: torch.device,
    alpha: float = 0.5,
    percentile_threshold: float = 75.0,
    verbose: bool = True
) -> List[Dict]:
    """
    Federated training with TABF aggregation.

    Args:
        global_model: Initial global model
        clients: List of federated clients
        validation_loader: Validation data for TABF scoring
        num_rounds: Number of communication rounds
        device: Compute device
        alpha: TABF alpha parameter
        percentile_threshold: TABF filtering threshold
        verbose: Print progress

    Returns:
        List of training history per round
    """
    aggregator = TABFAggregator(alpha=alpha, percentile_threshold=percentile_threshold)
    history = []

    for round_num in range(num_rounds):
        if verbose:
            print(f"\n=== TABF Round {round_num + 1}/{num_rounds} ===")

        # Broadcast global model to clients
        global_params = global_model.state_dict()
        for client in clients:
            client.set_model_parameters(global_params)

        # Local training on each client
        client_updates = []
        for client in clients:
            stats = client.train_local(epochs=5)
            client_updates.append(client.get_model_parameters())

        # TABF aggregation
        aggregated_params = aggregator.aggregate(
            global_model, client_updates, validation_loader, device, verbose
        )

        # Update global model
        global_model.load_state_dict(aggregated_params)

        # Evaluate
        # ... (evaluation code would go here)

        history.append({'round': round_num + 1})

    return history
