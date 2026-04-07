"""
Transformer-based models for encrypted traffic analysis

Implements transformer architectures adapted for encrypted traffic:
- TransECA-Net: Transformer with Efficient Channel Attention (98.94% on ISCX-VPN)
- FlowTransformer: Multi-head self-attention for flow analysis (97.4% on CICIDS2018)

These models leverage self-attention mechanisms to capture long-range dependencies
in packet sequences without sequential processing bottlenecks.

References:
    Paper Section 3.2 - Transformer Architecture
    Liu et al. (2024) - TransECA-Net achieving 98.94% accuracy
    Alkanhel et al. (2023) - FlowTransformer achieving 93-97% accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from .base import BaseModel


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer models.

    Adds position information to packet embeddings, allowing the model
    to utilize sequence ordering information.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class EfficientChannelAttention(nn.Module):
    """
    Efficient Channel Attention (ECA) module.

    Enhances important channels while suppressing less relevant ones,
    improving feature discrimination without significant computational overhead.

    Reference: Wang et al. (2020) - ECA-Net
    """

    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super(EfficientChannelAttention, self).__init__()

        kernel_size = int(abs((math.log2(channels) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class TransformerEncoderLayer(nn.Module):
    """
    Custom Transformer Encoder Layer with optional ECA.

    Extends standard transformer encoder with Efficient Channel Attention
    for improved feature representation in encrypted traffic analysis.
    """

    def __init__(self, d_model: int, nhead: int,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 use_eca: bool = True):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

        self.use_eca = use_eca
        if use_eca:
            self.eca = EfficientChannelAttention(d_model)

    def forward(self, src: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Multi-head self-attention
        src2, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Apply ECA if enabled
        if self.use_eca:
            src_permuted = src.permute(0, 2, 1)
            src_permuted = self.eca(src_permuted)
            src = src_permuted.permute(0, 2, 1)

        # Feedforward network
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src, attn_weights


class TransECANet(BaseModel):
    """
    Transformer with Efficient Channel Attention for encrypted traffic.

    Achieves 98.94% accuracy on ISCX-VPN encrypted traffic classification
    by combining transformer self-attention with efficient channel attention.

    Architecture:
    1. Input embedding
    2. Positional encoding
    3. Multiple transformer encoder layers with ECA
    4. Global pooling (avg + max)
    5. Classification head

    Reference: Paper Section 3.2, Liu et al. (2024)
    """

    def __init__(self, input_dim: int = 88, num_classes: int = 6,
                 d_model: int = 512, nhead: int = 8,
                 num_encoder_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 max_seq_length: int = 100, use_eca: bool = True):
        super(TransECANet, self).__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.d_model = d_model

        self.input_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout)

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, use_eca)
            for _ in range(num_encoder_layers)
        ])

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

        self.attention_weights = []

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        x = self.input_embedding(x)
        x = self.pos_encoder(x)

        self.attention_weights = []
        for layer in self.encoder_layers:
            x, attn = layer(x)
            if return_attention:
                self.attention_weights.append(attn)

        x = x.permute(0, 2, 1)
        avg_pool = self.global_avg_pool(x).squeeze(-1)
        max_pool = self.global_max_pool(x).squeeze(-1)

        x = torch.cat([avg_pool, max_pool], dim=1)
        logits = self.classifier(x)

        return logits

    def get_config(self) -> dict:
        return {
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
            'd_model': self.d_model
        }


class FlowTransformer(BaseModel):
    """
    Flow Transformer for encrypted traffic analysis.

    Simpler transformer architecture focusing on multi-head self-attention
    for capturing relationships across packet sequences.

    Achieves 97.4% accuracy on CICIDS2018 encrypted sessions.

    Reference: Paper Section 3.2, Alkanhel et al. (2023)
    """

    def __init__(self, input_dim: int = 88, num_classes: int = 6,
                 d_model: int = 256, nhead: int = 8,
                 num_layers: int = 4, dropout: float = 0.1,
                 max_seq_length: int = 100):
        super(FlowTransformer, self).__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.d_model = d_model

        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Mean pooling
        logits = self.classifier(x)
        return logits

    def get_config(self) -> dict:
        return {
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
            'd_model': self.d_model
        }
