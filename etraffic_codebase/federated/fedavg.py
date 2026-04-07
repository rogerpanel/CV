"""
Federated Averaging (FedAvg) for encrypted traffic IDS

Implements the standard FedAvg algorithm for privacy-preserving
collaborative training across distributed network monitoring nodes.

Each client trains on local encrypted traffic data without sharing
raw network data, preserving organizational privacy.

Reference:
    Paper Section 3.5 - Federated Learning Framework
    McMahan et al. (2017) - Communication-Efficient Learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader
import copy


class FederatedClient:
    """
    Federated learning client.

    Represents an individual network monitoring node that trains
    locally on private encrypted traffic data.
    """

    def __init__(self, client_id: int, model: nn.Module,
                 train_loader: DataLoader, device: torch.device,
                 learning_rate: float = 0.001):
        self.client_id = client_id
        self.model = copy.deepcopy(model)
        self.train_loader = train_loader
        self.device = device
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate
        )
        self.criterion = nn.CrossEntropyLoss()

    def set_model_parameters(self, params: Dict[str, torch.Tensor]):
        """Update local model with global parameters."""
        self.model.load_state_dict(params)

    def get_model_parameters(self) -> Dict[str, torch.Tensor]:
        """Return current model parameters."""
        return copy.deepcopy(self.model.state_dict())

    def train_local(self, epochs: int = 5) -> Dict[str, float]:
        """
        Train model locally for specified epochs.

        Args:
            epochs: Number of local training epochs

        Returns:
            Training statistics
        """
        self.model.to(self.device)
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for _ in range(epochs):
            for batch in self.train_loader:
                if len(batch) == 3:
                    x, _, y = batch
                else:
                    x, y = batch

                x, y = x.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.criterion(output, y)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                preds = output.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        return {
            'loss': total_loss / max(total, 1),
            'accuracy': correct / max(total, 1),
            'num_samples': total
        }

    @property
    def num_samples(self) -> int:
        return len(self.train_loader.dataset)


class FederatedServer:
    """
    Federated learning server.

    Coordinates the federated training process by broadcasting
    the global model, aggregating client updates, and evaluating
    performance on held-out test data.
    """

    def __init__(self, global_model: nn.Module, device: torch.device):
        self.global_model = global_model.to(device)
        self.device = device

    def broadcast(self, clients: List[FederatedClient]):
        """Broadcast global model to all clients."""
        global_params = self.global_model.state_dict()
        for client in clients:
            client.set_model_parameters(global_params)

    def aggregate(self, client_updates: List[Tuple[Dict, int]]):
        """
        Aggregate client updates via weighted averaging.

        Args:
            client_updates: List of (state_dict, num_samples) tuples
        """
        total_samples = sum(n for _, n in client_updates)

        aggregated = {}
        for name in client_updates[0][0]:
            aggregated[name] = sum(
                params[name] * (n / total_samples)
                for params, n in client_updates
            )

        self.global_model.load_state_dict(aggregated)

    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate global model on test data."""
        self.global_model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in test_loader:
                if len(batch) == 3:
                    x, _, y = batch
                else:
                    x, y = batch
                x, y = x.to(self.device), y.to(self.device)

                output = self.global_model(x)
                preds = output.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        return {'accuracy': correct / max(total, 1)}


def federated_training(
    global_model: nn.Module,
    clients: List[FederatedClient],
    test_loader: DataLoader,
    num_rounds: int = 20,
    local_epochs: int = 5,
    device: torch.device = None,
    verbose: bool = True
) -> List[Dict]:
    """
    Full federated training loop.

    Args:
        global_model: Initial global model
        clients: List of federated clients
        test_loader: Test data for evaluation
        num_rounds: Number of communication rounds
        local_epochs: Local epochs per round
        device: Compute device
        verbose: Print progress

    Returns:
        Training history per round
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    server = FederatedServer(global_model, device)
    history = []

    for round_num in range(num_rounds):
        if verbose:
            print(f"\n=== Round {round_num + 1}/{num_rounds} ===")

        # Broadcast global model
        server.broadcast(clients)

        # Local training
        client_updates = []
        for client in clients:
            stats = client.train_local(epochs=local_epochs)
            params = client.get_model_parameters()
            client_updates.append((params, client.num_samples))

            if verbose:
                print(f"  Client {client.client_id}: "
                      f"loss={stats['loss']:.4f}, acc={stats['accuracy']:.4f}")

        # Aggregate
        server.aggregate(client_updates)

        # Evaluate
        eval_metrics = server.evaluate(test_loader)
        history.append({
            'round': round_num + 1,
            **eval_metrics
        })

        if verbose:
            print(f"  Global accuracy: {eval_metrics['accuracy']:.4f}")

    return history
