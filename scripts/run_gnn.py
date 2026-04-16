from src.models.gnn_model import RiskGNN


def main():
    print("GNN runner scaffold is ready. Connect your graph dataset here.")
    model = RiskGNN(in_channels=3, hidden_channels=32, out_channels=2, heads=4)
    print(model)


if __name__ == "__main__":
    main()
