"""Service-Level Encoder with GAT + GRU for aggregate service behaviors."""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool


class ServiceLevelEncoder(nn.Module):
    def __init__(self, hidden_dims=[128, 256, 512], gru_hidden_size=256,
                 num_attention_heads=8, dropout=0.3):
        super().__init__()

        # GAT layers
        self.gat_layers = nn.ModuleList([
            GATConv(47, hidden_dims[0], heads=num_attention_heads, concat=True, dropout=dropout),
            GATConv(hidden_dims[0]*num_attention_heads, hidden_dims[1], heads=num_attention_heads, concat=True, dropout=dropout),
            GATConv(hidden_dims[1]*num_attention_heads, hidden_dims[2], heads=1, concat=False, dropout=dropout)
        ])

        # GRU for temporal modeling
        self.gru = nn.GRU(hidden_dims[2], gru_hidden_size, num_layers=2, batch_first=True)
        self.output_proj = nn.Linear(gru_hidden_size, hidden_dims[2])

    def forward(self, batch):
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch

        # GAT layers
        for gat in self.gat_layers:
            x = gat(x, edge_index)
            x = torch.relu(x)

        # Global pooling
        graph_emb = global_mean_pool(x, batch_idx)

        # GRU (assuming temporal dimension exists)
        output, hidden = self.gru(graph_emb.unsqueeze(1))
        return self.output_proj(output.squeeze(1))
