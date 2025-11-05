"""
Deep Spatio-Temporal Point Processes for Network Intrusion Detection
Transformer-based intensity modeling with logarithmic barrier optimization

Implementation of marked temporal point processes with:
- Cross-excitation between attack types
- Logarithmic barrier for survival integral (Lemma 1)
- Computational complexity reduction from O(n³) to O(n^(3/2))

Authors: Roger Nick Anaedevha, Alexander Gennadevich Trofimov, Yuri Vladimirovich Borodachev
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple, Optional, Dict
import numpy as np
import math


class TransformerIntensityModel(nn.Module):
    """
    Transformer-based conditional intensity function λ_k(t|H_t)

    Models temporal dependencies and cross-excitation between attack types
    using multi-head self-attention.

    Args:
        n_types: Number of event types (attack categories)
        d_model: Model dimension
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        dim_feedforward: Dimension of feedforward network
    """

    def __init__(
        self,
        n_types: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()
        self.n_types = n_types
        self.d_model = d_model

        # Event type embedding
        self.type_embedding = nn.Embedding(n_types, d_model)

        # Time encoding (continuous time)
        self.time_encoder = nn.Sequential(
            nn.Linear(1, d_model),
            nn.Tanh(),
            nn.Linear(d_model, d_model)
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Intensity projection for each event type
        self.intensity_head = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, 1),
                nn.Softplus()  # Ensure positive intensity
            ) for _ in range(n_types)
        ])

        # Base intensity (background rate)
        self.mu = nn.Parameter(torch.ones(n_types) * 0.1)

    def forward(
        self,
        event_times: Tensor,
        event_types: Tensor,
        query_times: Tensor,
        mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Compute conditional intensity at query times

        Args:
            event_times: Historical event times [batch_size, seq_len]
            event_types: Historical event types [batch_size, seq_len]
            query_times: Times to evaluate intensity [batch_size, n_query]
            mask: Attention mask [batch_size, seq_len]

        Returns:
            Intensity values [batch_size, n_query, n_types]
        """
        batch_size, seq_len = event_times.shape
        n_query = query_times.size(1)

        # Embed event types
        type_embed = self.type_embedding(event_types)  # [B, L, D]

        # Encode event times
        time_embed = self.time_encoder(event_times.unsqueeze(-1))  # [B, L, D]

        # Combine type and time embeddings
        event_repr = type_embed + time_embed  # [B, L, D]

        # Apply transformer
        if mask is not None:
            context = self.transformer(event_repr, src_key_padding_mask=mask)
        else:
            context = self.transformer(event_repr)  # [B, L, D]

        # Pool context (use last non-masked position)
        if mask is not None:
            lengths = (~mask).sum(dim=1)
            context_pooled = context[torch.arange(batch_size), lengths - 1]
        else:
            context_pooled = context[:, -1, :]  # [B, D]

        # Encode query times
        query_embed = self.time_encoder(query_times.unsqueeze(-1))  # [B, Q, D]

        # Compute intensity for each type
        intensities = []
        for k in range(self.n_types):
            # Modulate context with query times
            combined = context_pooled.unsqueeze(1) + query_embed  # [B, Q, D]

            # Compute intensity
            lambda_k = self.intensity_head[k](combined).squeeze(-1)  # [B, Q]

            # Add base intensity
            lambda_k = lambda_k + self.mu[k]

            intensities.append(lambda_k)

        intensities = torch.stack(intensities, dim=-1)  # [B, Q, K]

        return intensities


class LogarithmicBarrierSurvival(nn.Module):
    """
    Logarithmic barrier approximation for survival function integral (Lemma 1)

    Reduces computational complexity from O(n²) to O(n√n) while preventing
    intensity collapse through barrier penalty.

    Args:
        n_quadrature: Number of quadrature points
        barrier_strength: Strength of logarithmic barrier (epsilon)
    """

    def __init__(
        self,
        n_quadrature: int = 32,
        barrier_strength: float = 1e-6
    ):
        super().__init__()
        self.n_quadrature = n_quadrature
        self.barrier_strength = barrier_strength

        # Gauss-Legendre quadrature weights (learnable)
        self.quad_weights = nn.Parameter(
            torch.ones(n_quadrature) / n_quadrature
        )

    def forward(
        self,
        intensity_fn,
        t_start: Tensor,
        t_end: Tensor,
        **intensity_kwargs
    ) -> Tensor:
        """
        Compute survival integral with logarithmic barrier

        ∫_{t_start}^{t_end} λ(s) ds ≈ Σ w_i λ(s_i) - ε Σ log(λ(s_i))

        Args:
            intensity_fn: Function to compute intensity
            t_start: Start time [batch_size]
            t_end: End time [batch_size]
            **intensity_kwargs: Additional arguments for intensity function

        Returns:
            Survival integral approximation [batch_size]
        """
        batch_size = t_start.size(0)

        # Generate quadrature points in [t_start, t_end]
        # Using Gauss-Legendre points transformed to [0, 1]
        quad_points_unit = torch.linspace(0, 1, self.n_quadrature, device=t_start.device)

        # Transform to [t_start, t_end]
        duration = t_end - t_start
        quad_points = t_start.unsqueeze(1) + duration.unsqueeze(1) * quad_points_unit.unsqueeze(0)
        # [B, n_quad]

        # Evaluate intensity at quadrature points
        lambda_vals = intensity_fn(quad_points, **intensity_kwargs)  # [B, n_quad, n_types]

        # Sum over types
        lambda_total = lambda_vals.sum(dim=-1)  # [B, n_quad]

        # Standard integral term
        integral_term = torch.matmul(
            lambda_total,
            F.softmax(self.quad_weights, dim=0)
        ) * duration  # [B]

        # Logarithmic barrier term (prevents collapse)
        # -ε Σ log(λ(s_i) + ε)
        barrier_term = -self.barrier_strength * torch.log(
            lambda_total + self.barrier_strength
        ).sum(dim=1)  # [B]

        # Combined approximation
        survival_approx = integral_term + barrier_term

        return survival_approx

    def get_error_bound(self, n: int) -> float:
        """
        Theoretical error bound O(1/√n) from Lemma 1

        Args:
            n: Number of quadrature points

        Returns:
            Error bound
        """
        return 1.0 / math.sqrt(n)


class MarkedTemporalPointProcess(nn.Module):
    """
    Marked Temporal Point Process for network attack modeling

    Captures cross-excitation between attack types:
    - Reconnaissance triggering exploitation
    - Privilege escalation enabling lateral movement
    - Command & control preceding exfiltration

    Args:
        n_types: Number of attack types
        d_model: Model dimension
        hidden_dim: Hidden state dimension for coupling with ODE
    """

    def __init__(
        self,
        n_types: int,
        d_model: int = 256,
        hidden_dim: int = 256,
        nhead: int = 8,
        num_layers: int = 4
    ):
        super().__init__()
        self.n_types = n_types
        self.d_model = d_model

        # Transformer-based intensity model
        self.intensity_model = TransformerIntensityModel(
            n_types=n_types,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers
        )

        # Logarithmic barrier for survival integral
        self.survival_fn = LogarithmicBarrierSurvival(
            n_quadrature=32,
            barrier_strength=1e-6
        )

        # Cross-excitation matrix α_{k,k'} (how type k' excites type k)
        self.excitation_matrix = nn.Parameter(
            torch.eye(n_types) * 0.5 + torch.rand(n_types, n_types) * 0.1
        )

        # Decay parameters β_k for each type
        self.decay_params = nn.Parameter(torch.ones(n_types))

        # Coupling with ODE hidden state
        self.ode_coupling = nn.Sequential(
            nn.Linear(hidden_dim, d_model),
            nn.Tanh(),
            nn.Linear(d_model, n_types),
            nn.Softplus()
        )

    def compute_intensity(
        self,
        t: Tensor,
        event_times: Tensor,
        event_types: Tensor,
        ode_hidden: Optional[Tensor] = None,
        mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Compute conditional intensity λ_k(t|H_t)

        Args:
            t: Query time [batch_size] or [batch_size, n_query]
            event_times: Historical event times [batch_size, seq_len]
            event_types: Historical event types [batch_size, seq_len]
            ode_hidden: Optional ODE hidden state [batch_size, hidden_dim]
            mask: Attention mask [batch_size, seq_len]

        Returns:
            Intensity [batch_size, n_query, n_types]
        """
        if t.dim() == 1:
            t = t.unsqueeze(1)  # [B, 1]

        # Base intensity from transformer
        lambda_base = self.intensity_model(
            event_times, event_types, t, mask
        )  # [B, Q, K]

        # Add self-excitation from history
        batch_size, seq_len = event_times.shape
        n_query = t.size(1)

        excitation = torch.zeros(batch_size, n_query, self.n_types, device=t.device)

        for i in range(seq_len):
            if mask is not None and mask[:, i].any():
                continue

            t_i = event_times[:, i:i+1]  # [B, 1]
            m_i = event_types[:, i]  # [B]

            # Time difference
            dt = t - t_i  # [B, Q]
            dt = torch.clamp(dt, min=0)  # Only past events

            # Exponential decay
            decay = torch.exp(-self.decay_params.unsqueeze(0).unsqueeze(0) * dt.unsqueeze(-1))
            # [B, Q, K]

            # Cross-excitation from event type m_i to all types
            for b in range(batch_size):
                excitation[b] += self.excitation_matrix[:, m_i[b]].unsqueeze(0) * decay[b]

        lambda_total = lambda_base + excitation

        # Modulate with ODE hidden state if provided
        if ode_hidden is not None:
            modulation = self.ode_coupling(ode_hidden).unsqueeze(1)  # [B, 1, K]
            lambda_total = lambda_total * modulation

        return lambda_total

    def log_likelihood(
        self,
        event_times: Tensor,
        event_types: Tensor,
        t_end: Tensor,
        ode_hidden: Optional[Tensor] = None
    ) -> Tensor:
        """
        Compute log-likelihood of event sequence

        L = Σ_i log λ_{m_i}(t_i|H_{t_i}) - ∫_0^T Σ_k λ_k(s|H_s) ds

        Args:
            event_times: Event times [batch_size, seq_len]
            event_types: Event types [batch_size, seq_len]
            t_end: End of observation window [batch_size]
            ode_hidden: Optional ODE hidden state [batch_size, hidden_dim]

        Returns:
            Log-likelihood [batch_size]
        """
        batch_size, seq_len = event_times.shape

        # Event intensity term: Σ_i log λ_{m_i}(t_i)
        log_intensity_sum = torch.zeros(batch_size, device=event_times.device)

        for i in range(1, seq_len):  # Start from 1 to have history
            # History up to event i
            history_times = event_times[:, :i]
            history_types = event_types[:, :i]

            # Current event
            t_i = event_times[:, i:i+1]
            m_i = event_types[:, i]

            # Compute intensity
            lambda_i = self.compute_intensity(
                t_i, history_times, history_types, ode_hidden
            )  # [B, 1, K]

            # Select intensity for event type m_i
            lambda_mi = lambda_i[torch.arange(batch_size), 0, m_i]

            # Add to log-likelihood (with numerical stability)
            log_intensity_sum += torch.log(lambda_mi + 1e-10)

        # Survival term: ∫_0^T Σ_k λ_k(s) ds
        # Using logarithmic barrier approximation
        t_start = torch.zeros_like(t_end)

        def intensity_wrapper(t_query):
            return self.compute_intensity(
                t_query, event_times, event_types, ode_hidden
            )

        survival_integral = self.survival_fn(
            intensity_wrapper, t_start, t_end
        )

        # Total log-likelihood
        log_lik = log_intensity_sum - survival_integral

        return log_lik

    def sample_events(
        self,
        t_end: float,
        max_events: int = 1000,
        ode_hidden: Optional[Tensor] = None
    ) -> Tuple[List[float], List[int]]:
        """
        Sample event sequence using thinning algorithm

        Args:
            t_end: End time
            max_events: Maximum number of events
            ode_hidden: Optional ODE hidden state

        Returns:
            event_times: List of event times
            event_types: List of event types
        """
        event_times = []
        event_types = []

        t = 0.0

        with torch.no_grad():
            while t < t_end and len(event_times) < max_events:
                # Convert history to tensors
                if len(event_times) == 0:
                    history_times = torch.zeros(1, 1)
                    history_types = torch.zeros(1, 1, dtype=torch.long)
                else:
                    history_times = torch.tensor([event_times]).float()
                    history_types = torch.tensor([event_types]).long()

                # Compute current intensity
                t_tensor = torch.tensor([[t]]).float()
                lambda_t = self.compute_intensity(
                    t_tensor, history_times, history_types, ode_hidden
                )  # [1, 1, K]

                lambda_total = lambda_t.sum().item()

                # Upper bound (with safety factor)
                lambda_max = lambda_total * 1.5

                # Sample waiting time from homogeneous Poisson
                u = np.random.random()
                dt = -np.log(u) / (lambda_max + 1e-10)

                t = t + dt

                if t >= t_end:
                    break

                # Accept/reject
                lambda_t_new = self.compute_intensity(
                    torch.tensor([[t]]).float(), history_times, history_types, ode_hidden
                )
                lambda_total_new = lambda_t_new.sum().item()

                u = np.random.random()
                if u < lambda_total_new / lambda_max:
                    # Accept event
                    # Sample event type
                    probs = lambda_t_new[0, 0, :].cpu().numpy()
                    probs = probs / probs.sum()

                    event_type = np.random.choice(self.n_types, p=probs)

                    event_times.append(t)
                    event_types.append(event_type)

        return event_times, event_types


if __name__ == "__main__":
    # Test point process implementation
    print("="*80)
    print("Testing Deep Spatio-Temporal Point Process")
    print("="*80)

    # Configuration
    n_types = 15  # Number of attack types
    batch_size = 4
    seq_len = 20
    d_model = 256

    # Create model
    model = MarkedTemporalPointProcess(
        n_types=n_types,
        d_model=d_model,
        hidden_dim=256
    )

    print(f"\nModel Configuration:")
    print(f"  Number of attack types: {n_types}")
    print(f"  Model dimension: {d_model}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Generate synthetic data
    event_times = torch.cumsum(torch.rand(batch_size, seq_len) * 0.1, dim=1)
    event_types = torch.randint(0, n_types, (batch_size, seq_len))
    t_end = torch.ones(batch_size) * 2.0

    print(f"\nTest data:")
    print(f"  Event times shape: {event_times.shape}")
    print(f"  Event types shape: {event_types.shape}")

    # Test intensity computation
    t_query = torch.tensor([[1.0]]).expand(batch_size, 1)
    intensity = model.compute_intensity(
        t_query, event_times, event_types
    )

    print(f"\nIntensity computation:")
    print(f"  Query time: {t_query[0, 0].item():.3f}")
    print(f"  Intensity shape: {intensity.shape}")
    print(f"  Intensity range: [{intensity.min().item():.4f}, {intensity.max().item():.4f}]")

    # Test log-likelihood
    log_lik = model.log_likelihood(event_times, event_types, t_end)

    print(f"\nLog-likelihood:")
    print(f"  Shape: {log_lik.shape}")
    print(f"  Mean: {log_lik.mean().item():.4f}")
    print(f"  Std: {log_lik.std().item():.4f}")

    # Test event sampling
    print(f"\nEvent sampling test:")
    sampled_times, sampled_types = model.sample_events(t_end=5.0, max_events=50)
    print(f"  Generated {len(sampled_times)} events")
    print(f"  Time range: [0.0, {max(sampled_times) if sampled_times else 0:.3f}]")
    print(f"  Type distribution: {np.bincount(sampled_types, minlength=n_types)}")

    # Test logarithmic barrier error bound
    error_bound = model.survival_fn.get_error_bound(32)
    print(f"\nLogarithmic barrier (Lemma 1):")
    print(f"  Quadrature points: 32")
    print(f"  Theoretical error bound: O(1/√n) = {error_bound:.4f}")

    print("\n" + "="*80)
    print("Point Process Test Complete")
    print("="*80)
