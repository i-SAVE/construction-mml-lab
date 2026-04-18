# Подключаем будущие аннотации типов.
from __future__ import annotations

# Импортируем torch как базовую библиотеку тензоров.
import torch
# Импортируем функциональные операции (активации, dropout).
import torch.nn.functional as F
# Импортируем GATv2-слой из PyTorch Geometric.
from torch_geometric.nn import GATv2Conv


# Класс графовой модели для оценки риска на уровне узлов.
class RiskGNN(torch.nn.Module):
    # Докстринг кратко поясняет идею модели.
    """Simple GATv2 model for node-level risk prediction."""

    # Инициализатор модели с основными гиперпараметрами.
    def __init__(self, in_channels: int, hidden_channels: int = 32, out_channels: int = 2, heads: int = 4):
        # Вызываем инициализацию базового класса.
        super().__init__()
        # Первый слой многоголового attention.
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, dropout=0.2)
        # Второй слой attention, сжатие в 1 голову.
        self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=1, dropout=0.2)
        # Финальный линейный слой для выходов модели.
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    # Прямой проход модели.
    def forward(self, x, edge_index):
        # Применяем первый graph attention слой.
        x = self.conv1(x, edge_index)
        # Нелинейность ELU после первого слоя.
        x = F.elu(x)
        # Dropout для регуляризации в обучении.
        x = F.dropout(x, p=0.2, training=self.training)
        # Применяем второй graph attention слой.
        x = self.conv2(x, edge_index)
        # Повторная нелинейность.
        x = F.elu(x)
        # Возвращаем финальный прогноз.
        return self.lin(x)
