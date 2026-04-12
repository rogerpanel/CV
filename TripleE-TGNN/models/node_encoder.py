"""Node-Level Encoder with temporal GCN + LSTM for pod dynamics."""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool


class NodeLevelEncoder(nn.Module):
    def __init__(self, hidden_dims=[128, 256, 512], lstm_hidden_size=256, dropout=0.3):
        super().__init__()

        # GCN layers for pod interaction graph
        self.gcn_layers = nn.ModuleList([
            GCNConv(32, hidden_dims[0]),  # 32 pod features
            GCNConv(hidden_dims[0], hidden_dims[1]),
            GCNConv(hidden_dims[1], hidden_dims[2])
        ])

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(hidden_dims[2], lstm_hidden_size, num_layers=2, batch_first=True)
        self.output_proj = nn.Linear(lstm_hidden_size, hidden_dims[2])

    def forward(self, batch):
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch

        # GCN layers
        for gcn in self.gcn_layers:
            x = gcn(x, edge_index)
            x = torch.relu(x)

        # Global pooling
        graph_emb = global_mean_pool(x, batch_idx)

        # LSTM
        output, _ = self.lstm(graph_emb.unsqueeze(1))
        return self.output_proj(output.squeeze(1))
