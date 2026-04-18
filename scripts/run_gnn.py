# Импортируем модель GNN для демонстрационного запуска.
from src.models.gnn_model import RiskGNN


# Основная функция запуска.
def main():
    # Сообщаем, что это каркас для подключения реальных графовых данных.
    print("GNN runner scaffold is ready. Connect your graph dataset here.")
    # Создаём экземпляр модели с базовыми параметрами.
    model = RiskGNN(in_channels=3, hidden_channels=32, out_channels=2, heads=4)
    # Печатаем архитектуру модели.
    print(model)


# Точка входа для запуска скрипта.
if __name__ == "__main__":
    # Запускаем main.
    main()
