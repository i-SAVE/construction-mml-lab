# Подключаем будущие аннотации типов.
from __future__ import annotations

# Импортируем sys для модификации пути импорта при запуске файла напрямую.
import sys
# Импортируем Path для безопасной работы с путями.
from pathlib import Path

# Импортируем PyTorch для тензоров и вычислений.
import torch

# Если модуль запущен как файл, а не как пакет, поправляем sys.path.
if __package__ in {None, ""}:
    # Добавляем корень репозитория в путь импорта.
    sys.path.append(str(Path(__file__).resolve().parents[1]))

# Импортируем GNN-модель риска.
from src.models.gnn_model import RiskGNN


# Основная функция демонстрационного запуска GNN.
def main() -> None:
    # Поясняем, что это каркас и куда подключать реальные данные.
    print("GNN scaffold is ready. Add your graph dataset and masks here.")
    # Указываем число узлов в игрушечном графе.
    num_nodes = 8
    # Указываем число входных признаков на узел.
    in_channels = 3
    # Генерируем случайные признаки узлов.
    x = torch.randn(num_nodes, in_channels)
    # Описываем простые направленные рёбра цепочкой.
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7]], dtype=torch.long)
    # Создаём модель GATv2 для узловой классификации/оценки риска.
    model = RiskGNN(in_channels=in_channels, hidden_channels=32, out_channels=2, heads=4)
    # Получаем выход модели.
    out = model(x, edge_index)
    # Печатаем форму выхода для контроля корректности.
    print("Output tensor shape:", tuple(out.shape))


# Точка входа при запуске: python -m src.train_gnn.
if __name__ == "__main__":
    # Запускаем демонстрацию.
    main()
