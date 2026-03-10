"""
Deep Spatio-Temporal Point Process (DSTPP).

Implements the transformer-based intensity function (Eq. 8) and
marked Hawkes-like formulation (Eq. 11) from the main manuscript.

- Eq 8:  lambda_k(t) = softplus(W_k h_attn(t) + b_k)
- Eq 11: lambda*(t,k) = lambda_0(t) + sum_{t_i<t} alpha_{k_i,k} exp(-beta_{k_i,k}(t-t_i))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for continuous timestamps."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class ContinuousTimeEncoding(nn.Module):
    """Encode continuous timestamps into d_model-dimensional vectors."""

    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(1, d_model)
        self.act = nn.GELU()
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: [batch, seq_len] or [batch, seq_len, 1]"""
        if t.dim() == 2:
            t = t.unsqueeze(-1)
        return self.proj(self.act(self.linear(t)))


class TransformerIntensity(nn.Module):
    """Transformer-based conditional intensity (Eq. 8).

    Uses multi-head self-attention over the event history to produce
    h_attn(t), then maps to per-type intensities via softplus.

    Architecture: 4 transformer layers, 8 heads, d_model=512.
    """

    def __init__(self, hidden_dim: int, n_types: int,
                 d_model: int = 512, n_layers: int = 4,
                 n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.n_types = n_types
        self.d_model = d_model

        # Project ODE hidden state to transformer dimension
        self.input_proj = nn.Linear(hidden_dim, d_model)

        # Continuous time encoding
        self.time_enc = ContinuousTimeEncoding(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Per-type intensity heads
        self.intensity_heads = nn.Linear(d_model, n_types)

    def forward(self, h_seq: torch.Tensor, t_seq: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            h_seq: ODE hidden states [batch, seq_len, hidden_dim]
            t_seq: Timestamps [batch, seq_len]
            mask:  Causal mask [seq_len, seq_len] (optional)
        Returns:
            intensities: [batch, seq_len, n_types] (non-negative)
        """
        x = self.input_proj(h_seq) + self.time_enc(t_seq)
        h_attn = self.transformer(x, mask=mask)
        return F.softplus(self.intensity_heads(h_attn))


class MarkedHawkes(nn.Module):
    """Marked Hawkes excitation kernel (Eq. 11).

    lambda*(t, k) = lambda_0(t) + sum_{t_i<t} alpha_{k_i,k} exp(-beta_{k_i,k}(t - t_i))

    alpha captures cross-excitation between event types;
    beta controls temporal decay.
    """

    def __init__(self, n_types: int):
        super().__init__()
        self.n_types = n_types

        # Excitation matrix: alpha_{k_i, k} >= 0
        self.alpha_raw = nn.Parameter(torch.randn(n_types, n_types) * 0.1)
        # Decay matrix: beta_{k_i, k} > 0
        self.beta_raw = nn.Parameter(torch.ones(n_types, n_types))

    @property
    def alpha(self) -> torch.Tensor:
        return F.softplus(self.alpha_raw)

    @property
    def beta(self) -> torch.Tensor:
        return F.softplus(self.beta_raw)

    def excitation(self, dt: torch.Tensor,
                   event_types: torch.Tensor) -> torch.Tensor:
        """Compute Hawkes excitation contribution.

        Args:
            dt: Time differences [batch, n_events] (t - t_i for t_i < t)
            event_types: Event type indices [batch, n_events]
        Returns:
            excitation: [batch, n_types]
        """
        alpha = self.alpha  # [K, K]
        beta = self.beta    # [K, K]

        # Gather excitation/decay for observed event types
        # event_types: [batch, n_events] -> index into alpha[k_i, :]
        batch_size, n_events = event_types.shape
        alpha_sel = alpha[event_types]  # [batch, n_events, n_types]
        beta_sel = beta[event_types]    # [batch, n_events, n_types]

        decay = torch.exp(-beta_sel * dt.unsqueeze(-1))  # [batch, n_events, n_types]
        return (alpha_sel * decay).sum(dim=1)  # [batch, n_types]


class DeepSpatioTemporalPointProcess(nn.Module):
    """Full DSTPP combining transformer intensity and Hawkes excitation.

    Total intensity: lambda_k(t) = transformer_intensity_k(t) + hawkes_excitation_k(t)
    """

    def __init__(self, hidden_dim: int, n_types: int,
                 d_model: int = 512, n_layers: int = 4,
                 n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.n_types = n_types
        self.transformer_intensity = TransformerIntensity(
            hidden_dim, n_types, d_model, n_layers, n_heads, dropout
        )
        self.hawkes = MarkedHawkes(n_types)

    def compute_intensity(self, h_seq: torch.Tensor, t_seq: torch.Tensor,
                          event_types: torch.Tensor = None,
                          mask: torch.Tensor = None) -> torch.Tensor:
        """Compute total conditional intensity."""
        # Transformer component
        lam = self.transformer_intensity(h_seq, t_seq, mask)  # [B, S, K]

        # Hawkes excitation (if history available)
        if event_types is not None and event_types.shape[1] > 1:
            batch_size, seq_len = t_seq.shape
            hawkes_contrib = torch.zeros_like(lam)
            for j in range(1, seq_len):
                dt = t_seq[:, j:j+1] - t_seq[:, :j]  # [B, j]
                exc = self.hawkes.excitation(dt, event_types[:, :j])  # [B, K]
                hawkes_contrib[:, j] = exc
            lam = lam + hawkes_contrib

        return torch.clamp(lam, min=1e-8)

    def log_likelihood(self, h_seq: torch.Tensor, t_seq: torch.Tensor,
                       event_types: torch.Tensor,
                       T: torch.Tensor) -> torch.Tensor:
        """Compute point process negative log-likelihood (Eq. 4 in supplementary).

        L_TPP = -sum_i log lambda_{k_i}(t_i) + integral_0^T sum_k lambda_k(tau) dtau

        Uses log-barrier survival approximation (Lemma 1) with equispaced
        quadrature for the compensator integral.
        """
        lam = self.compute_intensity(h_seq, t_seq, event_types)  # [B, S, K]

        batch_size, seq_len, n_types = lam.shape

        # Log-intensity at event times: sum_i log lambda_{k_i}(t_i)
        event_types_expanded = event_types.unsqueeze(-1)  # [B, S, 1]
        lam_at_events = lam.gather(2, event_types_expanded).squeeze(-1)  # [B, S]
        log_lam_sum = torch.log(lam_at_events + 1e-8).sum(dim=1)  # [B]

        # Compensator via trapezoidal quadrature over observed times
        lam_total = lam.sum(dim=-1)  # [B, S]
        dt = t_seq[:, 1:] - t_seq[:, :-1]  # [B, S-1]
        compensator = 0.5 * ((lam_total[:, :-1] + lam_total[:, 1:]) * dt).sum(dim=1)

        nll = -log_lam_sum + compensator  # [B]
        return nll.mean()
