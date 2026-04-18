# Импортируем функцию расчёта метрик.
from src.utils.metrics import regression_metrics


# Дымовой тест: проверяем, что возвращаются ожидаемые ключи метрик.
def test_regression_metrics_keys():
    # Считаем метрики на простом идеальном примере.
    metrics = regression_metrics([1, 2, 3], [1, 2, 3])
    # Убеждаемся, что набор ключей стабильный.
    assert set(metrics.keys()) == {"mae", "rmse", "r2"}
