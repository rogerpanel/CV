"""Trace-Level Encoder with time-aware GCN for distributed request flows."""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_add_pool, AttentionalAggregation


class TraceLevelEncoder(nn.Module):
    def __init__(self, hidden_dims=[128, 256, 512], gru_hidden_size=256, dropout=0.3):
        super().__init__()

        # Time-aware GCN layers
        self.gcn_layers = nn.ModuleList([
            GCNConv(64, hidden_dims[0]),  # 64 span features
            GCNConv(hidden_dims[0], hidden_dims[1]),
            GCNConv(hidden_dims[1], hidden_dims[2])
        ])

        # Attention-weighted readout
        self.readout = AttentionalAggregation(nn.Linear(hidden_dims[2], 1))

        # GRU for temporal modeling
        self.gru = nn.GRU(hidden_dims[2], gru_hidden_size, num_layers=2, batch_first=True)
        self.output_proj = nn.Linear(gru_hidden_size, hidden_dims[2])

    def forward(self, batch):
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch

        # GCN layers
        for gcn in self.gcn_layers:
            x = gcn(x, edge_index)
            x = torch.relu(x)

        # Attention-weighted pooling
        graph_emb = self.readout(x, batch_idx)

        # GRU
        output, _ = self.gru(graph_emb.unsqueeze(1))
        return self.output_proj(output.squeeze(1))
