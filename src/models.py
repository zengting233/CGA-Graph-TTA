from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

class MeanConcatSAGELayer(nn.Module):
    """GraphSAGE-style mean aggregation with self-neighbor concatenation.

    h'_v = Linear([h_v || mean_{u in N(v)} h_u])
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.lin = nn.Linear(2 * in_channels, out_channels, bias=bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        row, col = edge_index
        neigh_sum = torch.zeros_like(x)
        neigh_sum.index_add_(0, col, x[row])
        deg = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        deg.index_add_(0, col, torch.ones_like(col, dtype=x.dtype))
        neigh_mean = neigh_sum / deg.clamp_min(1.0).unsqueeze(-1)
        return self.lin(torch.cat([x, neigh_mean], dim=-1))

class GraphSAGE(nn.Module):
    """2-layer GraphSAGE-style encoder plus a linear classifier head."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = MeanConcatSAGELayer(in_channels, hidden_channels)
        self.conv2 = MeanConcatSAGELayer(hidden_channels, hidden_channels)
        self.head = nn.Linear(hidden_channels, out_channels)
        self.dropout = float(dropout)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x, edge_index))
