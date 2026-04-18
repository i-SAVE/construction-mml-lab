# Подключаем будущие аннотации типов.
from __future__ import annotations

# Импортируем numpy для корня и числовых операций.
import numpy as np
# Импортируем стандартные метрики регрессии из sklearn.
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Функция считает набор метрик регрессии.
def regression_metrics(y_true, y_pred) -> dict:
    # Докстринг поясняет назначение функции.
    """Return common regression metrics."""
    # RMSE считаем как корень из MSE.
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    # Возвращаем словарь ключевых метрик.
    return {
        # Средняя абсолютная ошибка.
        "mae": float(mean_absolute_error(y_true, y_pred)),
        # Корень из средней квадратичной ошибки.
        "rmse": rmse,
        # Коэффициент детерминации.
        "r2": float(r2_score(y_true, y_pred)),
    }
