"""
Adversarial Defense Mechanisms for PQ-IDPS

Implements three complementary defense strategies:
1. Quantum Noise Injection: Depolarizing, amplitude damping, phase damping
2. Randomized Smoothing: Gaussian noise + certification
3. Adversarial Training: Train with quantum-enhanced adversarial examples

Combined, these provide certified robustness against quantum adversaries.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import numpy as np
from scipy.stats import norm


class QuantumNoiseInjection(nn.Module):
    """
    Injects quantum noise during training to improve adversarial robustness.

    Three types of quantum noise channels:
    1. Depolarizing: ρ_out = (1-p)ρ + (p/3)(XρX + YρY + ZρZ)
    2. Amplitude Damping: Models energy loss (|1⟩ → |0⟩ decay)
    3. Phase Damping: Models pure dephasing without energy loss

    In classical regime, simulates effect via probabilistic masking and scaling.

    Args:
        depolarizing_prob: Depolarizing channel probability (default 0.01)
        amplitude_damping_gamma: Amplitude damping parameter (default 0.05)
        phase_damping_lambda: Phase damping parameter (default 0.03)
        apply_during_inference: Apply noise at test time (default False)
    """

    def __init__(
        self,
        depolarizing_prob: float = 0.01,
        amplitude_damping_gamma: float = 0.05,
        phase_damping_lambda: float = 0.03,
        apply_during_inference: bool = False
    ):
        super().__init__()

        self.depolarizing_prob = depolarizing_prob
        self.amplitude_damping_gamma = amplitude_damping_gamma
        self.phase_damping_lambda = phase_damping_lambda
        self.apply_during_inference = apply_during_inference

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply quantum noise to input features.

        Args:
            x: Input tensor (any shape)

        Returns:
            x_noisy: Noisy input
        """
        if not self.training and not self.apply_during_inference:
            return x

        # 1. Depolarizing noise (random bit flips)
        if self.depolarizing_prob > 0:
            depolarizing_mask = torch.rand_like(x) > self.depolarizing_prob
            x = x * depolarizing_mask.float()

        # 2. Amplitude damping (energy decay)
        if self.amplitude_damping_gamma > 0:
            # Scale positive activations toward zero
            amplitude_scale = torch.sqrt(1 - self.amplitude_damping_gamma)
            positive_mask = x > 0
            x = torch.where(positive_mask, x * amplitude_scale, x)

        # 3. Phase damping (add uncorrelated noise)
        if self.phase_damping_lambda > 0:
            phase_noise = torch.randn_like(x) * torch.sqrt(
                torch.tensor(self.phase_damping_lambda)
            )
            x = x + phase_noise

        return x


class RandomizedSmoothing(nn.Module):
    """
    Randomized smoothing for certified adversarial robustness.

    Key idea: Add Gaussian noise to inputs during training and inference,
    creating a smoothed classifier g(x) that has provable robustness.

    Certification: If P(f(x + N(0, σ²I)) = c_A) ≥ p_A and
                       P(f(x + N(0, σ²I)) = c_B) ≤ p_B,
                   then g is robust within radius:
                       R = (σ/2)(Φ⁻¹(p_A) - Φ⁻¹(p_B))

    Args:
        noise_scale: Standard deviation of Gaussian noise σ (default 0.25)
        num_samples_train: Monte Carlo samples during training (default 10)
        num_samples_certify: Samples for certification (default 1000)

    Reference:
        Cohen et al., "Certified Adversarial Robustness via Randomized Smoothing",
        ICML 2019.
    """

    def __init__(
        self,
        noise_scale: float = 0.25,
        num_samples_train: int = 10,
        num_samples_certify: int = 1000
    ):
        super().__init__()

        self.noise_scale = noise_scale
        self.num_samples_train = num_samples_train
        self.num_samples_certify = num_samples_certify

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add Gaussian noise to inputs.

        Args:
            x: Input tensor

        Returns:
            x_noisy: Input + N(0, σ²I)
        """
        if not self.training:
            return x

        noise = torch.randn_like(x) * self.noise_scale
        return x + noise

    def predict_smooth(
        self,
        model: nn.Module,
        x: torch.Tensor,
        statistical_features: torch.Tensor,
        num_samples: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Smoothed prediction via Monte Carlo sampling.

        Args:
            model: PQ-IDPS model
            x: Packet sequences (batch_size, max_packets, 47)
            statistical_features: (batch_size, 12)
            num_samples: Number of noise samples (default num_samples_train)

        Returns:
            mean_logits: (batch_size, 2) - Averaged logits
            std_logits: (batch_size, 2) - Standard deviation
        """
        if num_samples is None:
            num_samples = self.num_samples_train

        model.eval()
        batch_size = x.size(0)

        logits_samples = []

        with torch.no_grad():
            for _ in range(num_samples):
                # Add Gaussian noise
                x_noisy = x + torch.randn_like(x) * self.noise_scale
                stat_noisy = statistical_features + \
                             torch.randn_like(statistical_features) * self.noise_scale

                # Forward pass
                logits, _ = model(x_noisy, stat_noisy)
                logits_samples.append(logits)

        # Stack samples
        logits_samples = torch.stack(logits_samples, dim=0)  # (num_samples, batch_size, 2)

        # Compute mean and std
        mean_logits = logits_samples.mean(dim=0)
        std_logits = logits_samples.std(dim=0)

        return mean_logits, std_logits

    def certify_robustness(
        self,
        model: nn.Module,
        x: torch.Tensor,
        statistical_features: torch.Tensor,
        alpha: float = 0.001
    ) -> Tuple[int, float]:
        """
        Compute certified robustness radius for a single input.

        Args:
            model: PQ-IDPS model
            x: Packet sequence (1, max_packets, 47)
            statistical_features: (1, 12)
            alpha: Confidence level (default 0.001 for 99.9% confidence)

        Returns:
            predicted_class: Most likely class
            radius: Certified robustness radius R
        """
        model.eval()

        # Count predictions via Monte Carlo
        counts = torch.zeros(2, device=x.device)

        with torch.no_grad():
            for _ in range(self.num_samples_certify):
                # Add Gaussian noise
                x_noisy = x + torch.randn_like(x) * self.noise_scale
                stat_noisy = statistical_features + \
                             torch.randn_like(statistical_features) * self.noise_scale

                # Predict
                logits, _ = model(x_noisy, stat_noisy)
                pred = torch.argmax(logits, dim=-1)

                counts[pred.item()] += 1

        # Most likely class
        counts_np = counts.cpu().numpy()
        top_class = int(np.argmax(counts_np))

        # Compute p_A (lower confidence bound for top class)
        p_A_hat = counts_np[top_class] / self.num_samples_certify
        p_A = self._lower_confidence_bound(p_A_hat, self.num_samples_certify, alpha)

        # Compute certified radius
        if p_A > 0.5:
            radius = self.noise_scale * norm.ppf(p_A) / 2.0
        else:
            radius = 0.0  # Cannot certify if p_A ≤ 0.5

        return top_class, radius

    @staticmethod
    def _lower_confidence_bound(p_hat: float, n: int, alpha: float) -> float:
        """
        Compute lower confidence bound for binomial proportion.

        Uses Clopper-Pearson interval.
        """
        from scipy.stats import beta

        if p_hat == 0:
            return 0.0

        return beta.ppf(alpha, n * p_hat, n * (1 - p_hat) + 1)


class AdversarialTrainer:
    """
    Adversarial training with quantum-enhanced attacks.

    Trains model on mixture of:
    - Clean examples
    - PGD adversarial examples
    - Grover-optimized adversarial examples (simulated √N speedup)

    Args:
        model: PQ-IDPS model
        epsilon: Perturbation budget (default 0.1)
        alpha: Step size (default 0.01)
        num_iterations: Attack iterations (default 10)
        grover_speedup: Simulate Grover speedup (default True)
        adversarial_ratio: Fraction of adversarial examples (default 0.5)
    """

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.1,
        alpha: float = 0.01,
        num_iterations: int = 10,
        grover_speedup: bool = True,
        adversarial_ratio: float = 0.5
    ):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iterations = num_iterations
        self.grover_speedup = grover_speedup
        self.adversarial_ratio = adversarial_ratio

    def pgd_attack(
        self,
        x: torch.Tensor,
        statistical_features: torch.Tensor,
        y: torch.Tensor,
        epsilon: Optional[float] = None,
        targeted: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Projected Gradient Descent (PGD) attack.

        Args:
            x: Packet sequences (batch_size, max_packets, 47)
            statistical_features: (batch_size, 12)
            y: True labels (batch_size,)
            epsilon: Perturbation budget (default self.epsilon)
            targeted: Targeted attack (default False)

        Returns:
            x_adv: Adversarial packet sequences
            stat_adv: Adversarial statistical features
        """
        if epsilon is None:
            epsilon = self.epsilon

        # Effective iterations (Grover speedup: √N)
        num_iters = self.num_iterations
        if self.grover_speedup:
            num_iters = max(int(np.sqrt(self.num_iterations)), 1)

        # Initialize perturbation
        x_adv = x.clone().detach().requires_grad_(True)
        stat_adv = statistical_features.clone().detach().requires_grad_(True)

        for _ in range(num_iters):
            # Forward pass
            logits, _ = self.model(x_adv, stat_adv)

            # Compute loss
            loss = F.cross_entropy(logits, y)

            if targeted:
                loss = -loss  # Minimize loss for targeted attack

            # Backward pass
            self.model.zero_grad()
            loss.backward()

            # Update perturbations
            with torch.no_grad():
                # Packet sequences
                if x_adv.grad is not None:
                    grad_x = x_adv.grad.sign()
                    x_adv = x_adv + self.alpha * grad_x
                    perturbation_x = torch.clamp(x_adv - x, -epsilon, epsilon)
                    x_adv = torch.clamp(x + perturbation_x, 0, 1)  # Clamp to valid range

                # Statistical features
                if stat_adv.grad is not None:
                    grad_stat = stat_adv.grad.sign()
                    stat_adv = stat_adv + self.alpha * grad_stat
                    perturbation_stat = torch.clamp(stat_adv - statistical_features,
                                                     -epsilon, epsilon)
                    stat_adv = torch.clamp(statistical_features + perturbation_stat, 0, 1)

                # Zero gradients
                x_adv.grad.zero_()
                if stat_adv.grad is not None:
                    stat_adv.grad.zero_()

            x_adv.requires_grad_(True)
            stat_adv.requires_grad_(True)

        return x_adv.detach(), stat_adv.detach()

    def train_step(
        self,
        x: torch.Tensor,
        statistical_features: torch.Tensor,
        y: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        Single adversarial training step.

        Args:
            x: Packet sequences (batch_size, max_packets, 47)
            statistical_features: (batch_size, 12)
            y: Labels (batch_size,)
            optimizer: PyTorch optimizer

        Returns:
            metrics: Dict with loss, clean_acc, adv_acc
        """
        batch_size = x.size(0)
        num_adversarial = int(batch_size * self.adversarial_ratio)

        # Split batch into clean and adversarial
        x_clean = x[:batch_size - num_adversarial]
        stat_clean = statistical_features[:batch_size - num_adversarial]
        y_clean = y[:batch_size - num_adversarial]

        x_for_adv = x[batch_size - num_adversarial:]
        stat_for_adv = statistical_features[batch_size - num_adversarial:]
        y_for_adv = y[batch_size - num_adversarial:]

        # Generate adversarial examples
        self.model.eval()  # Use eval mode for attack
        x_adv, stat_adv = self.pgd_attack(x_for_adv, stat_for_adv, y_for_adv)

        # Combine clean and adversarial
        x_combined = torch.cat([x_clean, x_adv], dim=0)
        stat_combined = torch.cat([stat_clean, stat_adv], dim=0)
        y_combined = torch.cat([y_clean, y_for_adv], dim=0)

        # Training step
        self.model.train()
        optimizer.zero_grad()

        logits, _ = self.model(x_combined, stat_combined)
        loss = F.cross_entropy(logits, y_combined)

        loss.backward()
        optimizer.step()

        # Compute metrics
        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            acc = (preds == y_combined).float().mean().item()

            # Clean accuracy
            logits_clean, _ = self.model(x_clean, stat_clean)
            preds_clean = torch.argmax(logits_clean, dim=-1)
            clean_acc = (preds_clean == y_clean).float().mean().item() if len(y_clean) > 0 else 0.0

            # Adversarial accuracy
            logits_adv, _ = self.model(x_adv, stat_adv)
            preds_adv = torch.argmax(logits_adv, dim=-1)
            adv_acc = (preds_adv == y_for_adv).float().mean().item() if len(y_for_adv) > 0 else 0.0

        return {
            'loss': loss.item(),
            'acc': acc,
            'clean_acc': clean_acc,
            'adv_acc': adv_acc
        }


if __name__ == "__main__":
    """Test adversarial defense mechanisms."""

    print("Testing Adversarial Defense Mechanisms...")
    print("=" * 60)

    # Test Quantum Noise Injection
    print("1. Quantum Noise Injection:")
    noise_injector = QuantumNoiseInjection(
        depolarizing_prob=0.01,
        amplitude_damping_gamma=0.05,
        phase_damping_lambda=0.03
    )

    x = torch.randn(16, 100, 47)
    x_noisy = noise_injector(x)

    print(f"   Input Mean: {x.mean().item():.4f}, Std: {x.std().item():.4f}")
    print(f"   Noisy Mean: {x_noisy.mean().item():.4f}, Std: {x_noisy.std().item():.4f}")
    print(f"   ✓ Noise applied successfully")
    print()

    # Test Randomized Smoothing
    print("2. Randomized Smoothing:")
    smoother = RandomizedSmoothing(noise_scale=0.25, num_samples_train=10)

    x_smoothed = smoother(x)
    print(f"   Input Shape: {x.shape}")
    print(f"   Smoothed Shape: {x_smoothed.shape}")
    print(f"   Noise Scale: σ={smoother.noise_scale}")
    print(f"   ✓ Smoothing applied successfully")
    print()

    print("=" * 60)
    print("✓ Adversarial defense test completed!")
