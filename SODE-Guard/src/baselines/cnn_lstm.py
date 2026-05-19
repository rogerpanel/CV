"""CNN-LSTM intrusion detector (Vinayakumar et al., 2017)."""
from __future__ import annotations
import torch
import torch.nn as nn


class CNN_LSTM(nn.Module):
    def __init__(self, feature_dim: int = 83, num_classes: int = 34,
                 cnn_channels: int = 64, lstm_hidden: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, cnn_channels, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.lstm = nn.LSTM(input_size=cnn_channels, hidden_size=lstm_hidden,
                            num_layers=2, batch_first=True, bidirectional=True)
        self.head = nn.Linear(2 * lstm_hidden, num_classes)

    def forward(self, x):
        z = self.conv(x.unsqueeze(1))                   # (B, C, F)
        z = z.transpose(1, 2)                            # (B, F, C)
        out, _ = self.lstm(z)
        return self.head(out[:, -1])

    forward_mc = forward
