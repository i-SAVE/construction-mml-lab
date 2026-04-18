from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class DataQualitySummary:
    rows_before: int
    rows_after: int
    duplicates_removed: int
    dropped_columns: list[str]


def clean_training_frame(df: pd.DataFrame, target: str, missing_threshold: float = 0.4) -> tuple[pd.DataFrame, DataQualitySummary]:
    """Basic cleaning for tabular training frames.

    Steps:
    1) Remove exact duplicate rows.
    2) Drop feature columns with too many missing values.
    """
    # Проверяем, что target есть в таблице.
    if target not in df.columns:
        # Бросаем понятную ошибку, если target не найден.
        raise ValueError(f"Target column '{target}' not found")

    # Запоминаем размер до очистки.
    rows_before = len(df)
    # Удаляем полные дубликаты строк.
    deduped = df.drop_duplicates().copy()
    # Считаем, сколько дублей удалено.
    duplicates_removed = rows_before - len(deduped)

    # Для признаков (без target) считаем долю пропусков.
    feature_missing = deduped.drop(columns=[target]).isna().mean()
    # Определяем колонки, где доля пропусков выше порога.
    dropped_columns = sorted(feature_missing[feature_missing > missing_threshold].index.tolist())
    # Удаляем найденные проблемные колонки.
    cleaned = deduped.drop(columns=dropped_columns)

    # Формируем сводку очистки.
    summary = DataQualitySummary(
        # Передаём число строк до очистки.
        rows_before=rows_before,
        # Передаём число строк после очистки.
        rows_after=len(cleaned),
        # Передаём число удалённых дублей.
        duplicates_removed=duplicates_removed,
        # Передаём список удалённых колонок.
        dropped_columns=dropped_columns,
    )
    # Возвращаем очищенный датафрейм и summary.
    return cleaned, summary


# Функция строит компактный отчёт о качестве данных.
def build_data_report(df: pd.DataFrame, target: str) -> dict[str, object]:
    # Докстринг кратко описывает назначение отчёта.
    """Generate short dataset diagnostics for training logs."""
    # Проверяем, что target существует.
    if target not in df.columns:
        # Если нет — выбрасываем ошибку.
        raise ValueError(f"Target column '{target}' not found")

    # Считаем процент пропусков по всем колонкам.
    missing_share = (df.isna().mean() * 100).sort_values(ascending=False)
    # Берём топ-10 самых проблемных колонок по пропускам.
    top_missing = {col: round(val, 2) for col, val in missing_share.head(10).items()}

    # Собираем итоговый словарь отчёта.
    report: dict[str, object] = {
        # Количество строк.
        "n_rows": len(df),
        # Количество колонок.
        "n_cols": df.shape[1],
        # Число числовых признаков.
        "n_numeric": int(df.select_dtypes(include=["number"]).shape[1]),
        # Число категориальных признаков.
        "n_categorical": int(df.select_dtypes(exclude=["number"]).shape[1]),
        # Процент пропусков в target.
        "target_missing_pct": round(float(df[target].isna().mean() * 100), 4),
        # Словарь топа колонок по проценту пропусков.
        "top_missing_pct": top_missing,
    }
    # Возвращаем отчёт.
    return report
