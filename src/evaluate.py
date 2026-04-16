from __future__ import annotations

from src.utils.metrics import regression_metrics


def print_regression_metrics(y_true, y_pred):
    metrics = regression_metrics(y_true, y_pred)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
