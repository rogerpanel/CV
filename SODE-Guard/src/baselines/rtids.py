"""RTIDS Transformer baseline (Wu, Jiang, Yang, Yu, He, "RTIDS:
A robust transformer-based intrusion detection system", IEEE Access 2022)."""
from __future__ import annotations
import torch
import torch.nn as nn


class RTIDSTransformer(nn.Module):
    def __init__(self, feature_dim: int = 83, num_classes: int = 34,
                 d_model: int = 128, nhead: int = 8, num_layers: int = 4):
        super().__init__()
        self.proj = nn.Linear(feature_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                               dim_feedforward=4 * d_model,
                                               activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        h = self.proj(x).unsqueeze(1)                  # (B, 1, d)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        h = torch.cat([cls, h], dim=1)
        h = self.encoder(h)[:, 0]
        return self.head(h)

    forward_mc = forward
