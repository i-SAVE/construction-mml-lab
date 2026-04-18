# Подключаем будущие аннотации типов.
from __future__ import annotations

# Импортируем Path для кросс-платформенной работы с путями.
from pathlib import Path
# Импортируем pandas для чтения и обработки CSV.
import pandas as pd


# Функция загрузки CSV в DataFrame.
def load_csv(data_path: str | Path) -> pd.DataFrame:
    # Докстринг описывает назначение функции.
    """Load a CSV file into a pandas DataFrame."""
    # Преобразуем путь к единому типу Path.
    path = Path(data_path)
    # Проверяем существование файла.
    if not path.exists():
        # Если файл не найден — выбрасываем ошибку.
        raise FileNotFoundError(f"CSV file not found: {path}")
    # Читаем CSV и возвращаем таблицу.
    return pd.read_csv(path)


# Функция разделения признаков и целевой переменной.
def split_features_target(df: pd.DataFrame, target: str):
    # Докстринг объясняет задачу функции.
    """Split a dataframe into features and target."""
    # Проверяем наличие целевой колонки.
    if target not in df.columns:
        # Если колонки нет — сообщаем ошибку.
        raise ValueError(f"Target column '{target}' not found in dataframe.")
    # Берём все колонки, кроме target, как признаки.
    X = df.drop(columns=[target])
    # Берём target как вектор ответов.
    y = df[target]
    # Возвращаем пару (X, y).
    return X, y
