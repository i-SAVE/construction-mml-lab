from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class RiskGNN(torch.nn.Module):
    """Simple GATv2 model for node-level risk prediction."""

    def __init__(self, in_channels: int, hidden_channels: int = 32, out_channels: int = 2, heads: int = 4):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, dropout=0.2)
        self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=1, dropout=0.2)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        return self.lin(x)
