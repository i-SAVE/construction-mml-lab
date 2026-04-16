from src.models.gnn_model import RiskGNN
import torch


def main():
    print('GNN scaffold is ready. Add your graph dataset and masks here.')
    num_nodes = 8
    in_channels = 3
    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7]], dtype=torch.long)
    model = RiskGNN(in_channels=in_channels, hidden_channels=32, out_channels=2, heads=4)
    out = model(x, edge_index)
    print('Output tensor shape:', tuple(out.shape))


if __name__ == '__main__':
    main()
