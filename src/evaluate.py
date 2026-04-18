# Подключаем будущие аннотации типов.
from __future__ import annotations

# Импортируем функцию вычисления метрик регрессии.
from src.utils.metrics import regression_metrics


# Функция печати метрик регрессии в удобном виде.
def print_regression_metrics(y_true, y_pred):
    # Считаем словарь метрик.
    metrics = regression_metrics(y_true, y_pred)
    # Выводим метрики по одной строке.
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
