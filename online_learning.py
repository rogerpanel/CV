"""
Online Learning with Drift Detection and Elastic Weight Consolidation
Implements Algorithm 2 from the paper

Components:
- Population Stability Index (PSI) for drift detection (threshold: 0.2)
- Elastic Weight Consolidation (EWC) to prevent catastrophic forgetting
- Differential Privacy with DP-SGD
- Maintains 98.8% accuracy over 50-day deployments

Authors: Roger Nick Anaedevha, Alexander Gennadevich Trofimov, Yuri Vladimirovich Borodachev
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import deque
import copy


class PopulationStabilityIndex:
    """
    Population Stability Index (PSI) for drift detection

    PSI measures distribution shift between baseline and current data:
    PSI = Σ (p_current - p_baseline) * log(p_current / p_baseline)

    Triggers adaptation when PSI > 0.2

    Args:
        n_bins: Number of bins for histogram
        threshold: PSI threshold for drift detection
    """

    def __init__(self, n_bins: int = 10, threshold: float = 0.2):
        self.n_bins = n_bins
        self.threshold = threshold
        self.baseline_dist = None
        self.feature_ranges = None

    def fit_baseline(self, X: np.ndarray):
        """
        Fit baseline distribution

        Args:
            X: Baseline data [n_samples, n_features]
        """
        n_features = X.shape[1]

        # Compute feature ranges
        self.feature_ranges = []
        self.baseline_dist = []

        for f in range(n_features):
            feature_data = X[:, f]

            # Compute range
            min_val = feature_data.min()
            max_val = feature_data.max()
            self.feature_ranges.append((min_val, max_val))

            # Compute histogram
            hist, _ = np.histogram(
                feature_data,
                bins=self.n_bins,
                range=(min_val, max_val)
            )

            # Normalize to probabilities
            hist = hist / hist.sum()

            # Add small epsilon to avoid log(0)
            hist = hist + 1e-10
            hist = hist / hist.sum()

            self.baseline_dist.append(hist)

    def compute_psi(self, X: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute PSI for current data

        Args:
            X: Current data [n_samples, n_features]

        Returns:
            psi_total: Total PSI across all features
            psi_per_feature: PSI for each feature
        """
        if self.baseline_dist is None:
            raise ValueError("Must fit baseline distribution first")

        n_features = X.shape[1]
        psi_per_feature = np.zeros(n_features)

        for f in range(n_features):
            feature_data = X[:, f]
            min_val, max_val = self.feature_ranges[f]

            # Compute current histogram
            hist_current, _ = np.histogram(
                feature_data,
                bins=self.n_bins,
                range=(min_val, max_val)
            )

            # Normalize
            hist_current = hist_current / hist_current.sum()
            hist_current = hist_current + 1e-10
            hist_current = hist_current / hist_current.sum()

            # Compute PSI for this feature
            hist_baseline = self.baseline_dist[f]

            psi = np.sum(
                (hist_current - hist_baseline) * np.log(hist_current / hist_baseline)
            )

            psi_per_feature[f] = psi

        psi_total = psi_per_feature.mean()

        return psi_total, psi_per_feature

    def detect_drift(self, X: np.ndarray) -> Tuple[bool, float]:
        """
        Detect distribution drift

        Args:
            X: Current data

        Returns:
            is_drift: Whether drift is detected
            psi: PSI value
        """
        psi_total, _ = self.compute_psi(X)
        is_drift = psi_total > self.threshold

        return is_drift, psi_total


class ElasticWeightConsolidation:
    """
    Elastic Weight Consolidation (EWC) for preventing catastrophic forgetting

    Penalizes changes to important parameters:
    L_EWC = L_task + (λ/2) Σ_i F_i (θ_i - θ*_i)²

    where F_i is the Fisher information matrix diagonal

    Args:
        model: Neural network model
        importance_weight: λ parameter (default: 5e-3 from paper)
    """

    def __init__(self, model: nn.Module, importance_weight: float = 5e-3):
        self.model = model
        self.importance_weight = importance_weight

        # Store important parameters
        self.anchor_params = {}
        self.fisher_information = {}

    def compute_fisher_information(
        self,
        dataloader: torch.utils.data.DataLoader,
        device: str = "cuda"
    ):
        """
        Compute Fisher Information Matrix (diagonal approximation)

        F_i = E[(∂log p(y|x,θ)/∂θ_i)²]

        Args:
            dataloader: Data loader for computing Fisher information
            device: Computation device
        """
        self.model.eval()

        # Initialize Fisher information
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.fisher_information[name] = torch.zeros_like(param.data)

        n_samples = 0

        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            # Forward pass
            self.model.zero_grad()
            output, _ = self.model(x)

            # Compute log likelihood
            log_likelihood = torch.nn.functional.log_softmax(output, dim=1)[
                range(len(y)), y
            ].sum()

            # Compute gradients
            log_likelihood.backward()

            # Accumulate Fisher information (gradient squared)
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.fisher_information[name] += param.grad.data ** 2

            n_samples += len(x)

        # Normalize
        for name in self.fisher_information:
            self.fisher_information[name] /= n_samples

    def store_anchor_params(self):
        """Store current parameters as anchor"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.anchor_params[name] = param.data.clone()

    def compute_ewc_loss(self) -> Tensor:
        """
        Compute EWC regularization loss

        Returns:
            EWC loss
        """
        loss = torch.tensor(0.0, device=next(self.model.parameters()).device)

        for name, param in self.model.named_parameters():
            if name in self.fisher_information:
                fisher = self.fisher_information[name]
                anchor = self.anchor_params[name]

                # Penalize deviation from anchor weighted by Fisher information
                loss += (fisher * (param - anchor) ** 2).sum()

        return self.importance_weight * loss / 2


class DifferentialPrivacySGD:
    """
    Differential Privacy Stochastic Gradient Descent (DP-SGD)

    Adds calibrated noise to gradients for privacy preservation:
    - Gradient clipping to bound sensitivity
    - Gaussian noise addition

    Args:
        epsilon: Privacy budget (default: 1.0)
        delta: Failure probability (default: 1e-5)
        max_grad_norm: Maximum gradient norm for clipping
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        max_grad_norm: float = 1.0
    ):
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm

        # Compute noise multiplier from privacy budget
        self.noise_multiplier = self._compute_noise_multiplier()

    def _compute_noise_multiplier(self) -> float:
        """
        Compute noise multiplier from (ε, δ)-DP parameters

        Uses moments accountant approximation
        """
        # Simplified calculation (for production, use opacus or similar)
        noise_multiplier = np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        return noise_multiplier

    def clip_and_add_noise(self, model: nn.Module, batch_size: int):
        """
        Clip gradients and add Gaussian noise

        Args:
            model: Neural network model
            batch_size: Current batch size
        """
        # Clip gradients
        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            self.max_grad_norm
        )

        # Add noise to each parameter
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                noise = torch.randn_like(param.grad) * self.noise_multiplier * self.max_grad_norm / batch_size
                param.grad += noise

        return total_norm


class OnlineAdaptiveSystem:
    """
    Online adaptive learning system with drift detection and EWC
    Implements Algorithm 2 from the paper

    Args:
        model: TA-BN-ODE model
        device: Computation device
        ema_rate: Exponential moving average rate (ρ = 0.98)
        ewc_weight: EWC importance weight (η = 5e-3)
        buffer_size: Size of replay buffer
        psi_threshold: PSI threshold for drift detection (0.2)
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        ema_rate: float = 0.98,
        ewc_weight: float = 5e-3,
        buffer_size: int = 1000,
        psi_threshold: float = 0.2,
        use_dp: bool = False,
        dp_epsilon: float = 1.0
    ):
        self.model = model.to(device)
        self.device = device
        self.ema_rate = ema_rate
        self.buffer_size = buffer_size

        # Drift detection
        self.psi_monitor = PopulationStabilityIndex(threshold=psi_threshold)

        # EWC for preventing forgetting
        self.ewc = ElasticWeightConsolidation(model, importance_weight=ewc_weight)

        # Differential privacy (optional)
        self.use_dp = use_dp
        if use_dp:
            self.dp_sgd = DifferentialPrivacySGD(epsilon=dp_epsilon)

        # Experience replay buffer
        self.buffer_x = deque(maxlen=buffer_size)
        self.buffer_y = deque(maxlen=buffer_size)

        # EMA of model parameters
        self.ema_model = copy.deepcopy(model)

        # Statistics tracking
        self.n_updates = 0
        self.drift_count = 0
        self.psi_history = []
        self.accuracy_history = []

        # Optimizer
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    def initialize_baseline(self, X_baseline: np.ndarray, dataloader):
        """
        Initialize baseline distribution and Fisher information

        Args:
            X_baseline: Baseline feature matrix
            dataloader: Baseline data loader for Fisher computation
        """
        # Fit PSI baseline
        self.psi_monitor.fit_baseline(X_baseline)

        # Compute initial Fisher information
        self.ewc.compute_fisher_information(dataloader, self.device)

        # Store anchor parameters
        self.ewc.store_anchor_params()

        print("Baseline initialized:")
        print(f"  Features: {X_baseline.shape[1]}")
        print(f"  Samples: {X_baseline.shape[0]}")

    def update_online(
        self,
        x: Tensor,
        y: Tensor,
        X_current_batch: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Online update with single sample or batch (Algorithm 2)

        Args:
            x: Input features [batch_size, n_features]
            y: Labels [batch_size]
            X_current_batch: Current batch for PSI computation

        Returns:
            Metrics dictionary
        """
        self.model.train()

        # Check for drift
        drift_detected = False
        psi_value = 0.0

        if X_current_batch is not None:
            drift_detected, psi_value = self.psi_monitor.detect_drift(X_current_batch)
            self.psi_history.append(psi_value)

            if drift_detected:
                self.drift_count += 1
                print(f"Drift detected! PSI: {psi_value:.4f}")

        # Add to buffer
        for i in range(len(x)):
            self.buffer_x.append(x[i])
            self.buffer_y.append(y[i])

        # Sample from buffer for training
        if len(self.buffer_x) >= 32:
            # Mini-batch from buffer
            indices = np.random.choice(len(self.buffer_x), size=32, replace=False)
            x_batch = torch.stack([self.buffer_x[i] for i in indices]).to(self.device)
            y_batch = torch.stack([self.buffer_y[i] for i in indices]).to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            output, _ = self.model(x_batch)

            # Task loss
            loss_task = nn.functional.cross_entropy(output, y_batch)

            # EWC regularization loss
            loss_ewc = self.ewc.compute_ewc_loss()

            # Total loss
            loss = loss_task + loss_ewc

            # Backward pass
            loss.backward()

            # Apply differential privacy if enabled
            if self.use_dp:
                self.dp_sgd.clip_and_add_noise(self.model, len(x_batch))

            # Optimizer step
            self.optimizer.step()

            # Update EMA model
            self._update_ema()

            # Compute accuracy
            with torch.no_grad():
                preds = output.argmax(dim=1)
                acc = (preds == y_batch).float().mean().item()
                self.accuracy_history.append(acc)

            self.n_updates += 1

            # If drift detected, update anchor
            if drift_detected:
                self.ewc.compute_fisher_information(
                    self._create_buffer_dataloader(),
                    self.device
                )
                self.ewc.store_anchor_params()

            metrics = {
                'loss_task': loss_task.item(),
                'loss_ewc': loss_ewc.item(),
                'loss_total': loss.item(),
                'accuracy': acc,
                'psi': psi_value,
                'drift_detected': drift_detected,
                'n_updates': self.n_updates,
                'drift_count': self.drift_count
            }

            return metrics
        else:
            return {'status': 'buffering'}

    def _update_ema(self):
        """Update Exponential Moving Average of model parameters"""
        for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data.mul_(self.ema_rate).add_(param.data, alpha=1 - self.ema_rate)

    def _create_buffer_dataloader(self):
        """Create dataloader from buffer"""
        from torch.utils.data import TensorDataset, DataLoader

        X = torch.stack(list(self.buffer_x))
        y = torch.stack(list(self.buffer_y))

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        return loader

    def predict(self, x: Tensor, use_ema: bool = True) -> Tensor:
        """
        Predict with online model

        Args:
            x: Input features
            use_ema: Whether to use EMA model (more stable)

        Returns:
            Predictions
        """
        model = self.ema_model if use_ema else self.model
        model.eval()

        with torch.no_grad():
            output, _ = model(x)
            preds = output.argmax(dim=1)

        return preds

    def get_statistics(self) -> Dict[str, any]:
        """Get system statistics"""
        stats = {
            'n_updates': self.n_updates,
            'drift_count': self.drift_count,
            'buffer_size': len(self.buffer_x),
            'mean_psi': np.mean(self.psi_history) if self.psi_history else 0.0,
            'mean_accuracy': np.mean(self.accuracy_history[-100:]) if self.accuracy_history else 0.0,
            'std_accuracy': np.std(self.accuracy_history[-100:]) if self.accuracy_history else 0.0
        }

        return stats


if __name__ == "__main__":
    # Test online learning system
    print("="*80)
    print("Testing Online Adaptive Learning System")
    print("="*80)

    # Mock model
    class MockModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)
            self.output_dim = output_dim

        def forward(self, x):
            h = torch.relu(self.fc1(x))
            return self.fc2(h), h

    input_dim = 64
    hidden_dim = 128
    num_classes = 15

    model = MockModel(input_dim, hidden_dim, num_classes)

    # Create online system
    online_system = OnlineAdaptiveSystem(
        model=model,
        device="cpu",
        ema_rate=0.98,
        ewc_weight=5e-3,
        buffer_size=1000,
        psi_threshold=0.2,
        use_dp=False
    )

    print(f"\nOnline System Configuration:")
    print(f"  EMA rate (ρ): {online_system.ema_rate}")
    print(f"  EWC weight (η): {online_system.ewc.importance_weight}")
    print(f"  Buffer size: {online_system.buffer_size}")
    print(f"  PSI threshold: {online_system.psi_monitor.threshold}")

    # Create baseline data
    X_baseline = np.random.randn(1000, input_dim)

    from torch.utils.data import TensorDataset, DataLoader
    baseline_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(X_baseline),
            torch.randint(0, num_classes, (1000,))
        ),
        batch_size=32
    )

    # Initialize baseline
    online_system.initialize_baseline(X_baseline, baseline_loader)

    # Simulate online stream
    print(f"\nSimulating online data stream:")

    for t in range(100):
        # Generate sample (with gradual drift)
        drift_factor = t / 100
        x = torch.randn(8, input_dim) + drift_factor * 0.5
        y = torch.randint(0, num_classes, (8,))

        X_batch = x.numpy()

        # Online update
        metrics = online_system.update_online(x, y, X_batch)

        if metrics.get('status') != 'buffering' and t % 10 == 0:
            print(f"  Step {t}:")
            print(f"    Loss: {metrics['loss_total']:.4f}")
            print(f"    Accuracy: {metrics['accuracy']:.2%}")
            print(f"    PSI: {metrics['psi']:.4f}")
            print(f"    Drift detected: {metrics['drift_detected']}")

    # Final statistics
    print(f"\nFinal Statistics:")
    stats = online_system.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Test prediction
    x_test = torch.randn(16, input_dim)
    preds = online_system.predict(x_test, use_ema=True)

    print(f"\nPrediction test:")
    print(f"  Input shape: {x_test.shape}")
    print(f"  Predictions shape: {preds.shape}")
    print(f"  Prediction distribution: {torch.bincount(preds, minlength=num_classes)}")

    print("\n" + "="*80)
    print("Online Learning Test Complete")
    print("="*80)
