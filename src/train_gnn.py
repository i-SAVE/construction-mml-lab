from __future__ import annotations

import sys
from pathlib import Path

import torch

if __package__ in {None, ""}:
    # Allow running as: python src/train_gnn.py
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.gnn_model import RiskGNN


def main() -> None:
    print("GNN scaffold is ready. Add your graph dataset and masks here.")
    num_nodes = 8
    in_channels = 3
    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7]], dtype=torch.long)
    model = RiskGNN(in_channels=in_channels, hidden_channels=32, out_channels=2, heads=4)
    out = model(x, edge_index)
    print("Output tensor shape:", tuple(out.shape))


if __name__ == "__main__":
    main()
