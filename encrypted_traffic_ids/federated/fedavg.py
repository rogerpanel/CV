"""
Federated Averaging (FedAvg) implementation

Implements the FedAvg algorithm for privacy-preserving collaborative training
across multiple clients without centralizing data.

Reference:
    McMahan et al. (2017) - Communication-Efficient Learning of Deep Networks
    from Decentralized Data
"""

import torch
import torch.nn as nn
import copy
from typing import List, Dict, Optional, Tuple
import numpy as np


class FederatedClient:
    """
    Federated learning client.

    Each client trains locally on private encrypted traffic data and sends
    only model updates to the central server, preserving data privacy.
    """

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        device: torch.device,
        learning_rate: float = 0.001
    ):
        """
        Initialize federated client.

        Args:
            client_id: Unique client identifier
            model: Model to train
            train_loader: Training data loader (local data)
            device: Device for computation
            learning_rate: Learning rate for local training
        """
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.learning_rate = learning_rate

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def train_local(self, epochs: int = 1) -> Dict[str, float]:
        """
        Perform local training for specified number of epochs.

        Args:
            epochs: Number of local training epochs

        Returns:
            Dictionary with training statistics
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for epoch in range(epochs):
            epoch_loss = 0.0

            for batch_idx, batch in enumerate(self.train_loader):
                # Unpack batch
                if len(batch) == 3:
                    x, _, y = batch
                else:
                    x, y = batch

                x, y = x.to(self.device), y.to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(x)

                loss = self.criterion(outputs, y)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                # Statistics
                epoch_loss += loss.item()
                _, predicted = outputs.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()

            total_loss += epoch_loss

        avg_loss = total_loss / (epochs * len(self.train_loader))
        accuracy = 100.0 * correct / total

        return {
            'client_id': self.client_id,
            'loss': avg_loss,
            'accuracy': accuracy,
            'num_samples': total
        }

    def get_model_parameters(self) -> Dict[str, torch.Tensor]:
        """Get current model parameters."""
        return {name: param.data.clone() for name, param in self.model.named_parameters()}

    def set_model_parameters(self, parameters: Dict[str, torch.Tensor]) -> None:
        """Set model parameters from server."""
        for name, param in self.model.named_parameters():
            param.data = parameters[name].clone()


class FederatedServer:
    """
    Federated learning server.

    Coordinates training across clients by:
    1. Broadcasting global model to clients
    2. Collecting client updates
    3. Aggregating updates (FedAvg)
    4. Updating global model
    """

    def __init__(
        self,
        global_model: nn.Module,
        device: torch.device,
        aggregation_strategy: str = 'fedavg'
    ):
        """
        Initialize federated server.

        Args:
            global_model: Global model to train
            device: Device for computation
            aggregation_strategy: Aggregation method ('fedavg', 'fedprox', 'gradient_similarity')
        """
        self.global_model = global_model.to(device)
        self.device = device
        self.aggregation_strategy = aggregation_strategy
        self.round = 0

    def get_global_model_parameters(self) -> Dict[str, torch.Tensor]:
        """Get global model parameters to broadcast to clients."""
        return {name: param.data.clone() for name, param in self.global_model.named_parameters()}

    def aggregate_fedavg(
        self,
        client_parameters: List[Dict[str, torch.Tensor]],
        client_weights: List[float]
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate client models using FedAvg (weighted averaging).

        Args:
            client_parameters: List of client model parameters
            client_weights: List of client weights (typically proportional to data size)

        Returns:
            Aggregated model parameters
        """
        # Normalize weights to sum to 1
        total_weight = sum(client_weights)
        normalized_weights = [w / total_weight for w in client_weights]

        # Initialize aggregated parameters
        aggregated = {}

        # Get parameter names from first client
        param_names = list(client_parameters[0].keys())

        # Weighted average for each parameter
        for param_name in param_names:
            aggregated[param_name] = sum(
                weight * client_params[param_name]
                for weight, client_params in zip(normalized_weights, client_parameters)
            )

        return aggregated

    def update_global_model(
        self,
        client_parameters: List[Dict[str, torch.Tensor]],
        client_weights: List[float]
    ) -> None:
        """
        Update global model with aggregated client parameters.

        Args:
            client_parameters: List of client model parameters
            client_weights: List of client weights
        """
        if self.aggregation_strategy == 'fedavg':
            aggregated = self.aggregate_fedavg(client_parameters, client_weights)
        else:
            raise ValueError(f"Unknown aggregation strategy: {self.aggregation_strategy}")

        # Update global model
        for name, param in self.global_model.named_parameters():
            param.data = aggregated[name].clone()

        self.round += 1

    def evaluate(
        self,
        test_loader: torch.utils.data.DataLoader,
        criterion: nn.Module = None
    ) -> Dict[str, float]:
        """
        Evaluate global model on test data.

        Args:
            test_loader: Test data loader
            criterion: Loss criterion

        Returns:
            Dictionary with evaluation metrics
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        self.global_model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in test_loader:
                # Unpack batch
                if len(batch) == 3:
                    x, _, y = batch
                else:
                    x, y = batch

                x, y = x.to(self.device), y.to(self.device)

                # Forward pass
                outputs = self.global_model(x)
                loss = criterion(outputs, y)

                # Statistics
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()

        avg_loss = test_loss / len(test_loader)
        accuracy = 100.0 * correct / total

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'total_samples': total
        }


def federated_training(
    server: FederatedServer,
    clients: List[FederatedClient],
    num_rounds: int,
    local_epochs: int,
    test_loader: Optional[torch.utils.data.DataLoader] = None,
    verbose: bool = True
) -> List[Dict[str, float]]:
    """
    Run federated training.

    Args:
        server: Federated server
        clients: List of federated clients
        num_rounds: Number of communication rounds
        local_epochs: Number of local training epochs per round
        test_loader: Test data loader for evaluation
        verbose: Whether to print progress

    Returns:
        List of metrics for each round
    """
    history = []

    for round_num in range(num_rounds):
        if verbose:
            print(f"\n=== Round {round_num + 1}/{num_rounds} ===")

        # Broadcast global model to clients
        global_params = server.get_global_model_parameters()
        for client in clients:
            client.set_model_parameters(global_params)

        # Local training on each client
        client_params = []
        client_weights = []
        client_stats = []

        for client in clients:
            # Train locally
            stats = client.train_local(epochs=local_epochs)
            client_stats.append(stats)

            # Collect model parameters and weights
            client_params.append(client.get_model_parameters())
            client_weights.append(stats['num_samples'])  # Weight by dataset size

            if verbose:
                print(f"  Client {client.client_id}: Loss={stats['loss']:.4f}, "
                      f"Acc={stats['accuracy']:.2f}%")

        # Aggregate client updates
        server.update_global_model(client_params, client_weights)

        # Evaluate global model
        round_metrics = {'round': round_num + 1}

        if test_loader is not None:
            eval_metrics = server.evaluate(test_loader)
            round_metrics.update(eval_metrics)

            if verbose:
                print(f"  Global Model: Loss={eval_metrics['loss']:.4f}, "
                      f"Acc={eval_metrics['accuracy']:.2f}%")

        history.append(round_metrics)

    return history
