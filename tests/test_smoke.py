from src.utils.metrics import regression_metrics


def test_regression_metrics_keys():
    metrics = regression_metrics([1, 2, 3], [1, 2, 3])
    assert set(metrics.keys()) == {"mae", "rmse", "r2"}
