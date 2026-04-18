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
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found")

    rows_before = len(df)
    deduped = df.drop_duplicates().copy()
    duplicates_removed = rows_before - len(deduped)

    feature_missing = deduped.drop(columns=[target]).isna().mean()
    dropped_columns = sorted(feature_missing[feature_missing > missing_threshold].index.tolist())
    cleaned = deduped.drop(columns=dropped_columns)

    summary = DataQualitySummary(
        rows_before=rows_before,
        rows_after=len(cleaned),
        duplicates_removed=duplicates_removed,
        dropped_columns=dropped_columns,
    )
    return cleaned, summary


def build_data_report(df: pd.DataFrame, target: str) -> dict[str, object]:
    """Generate short dataset diagnostics for training logs."""
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found")

    missing_share = (df.isna().mean() * 100).sort_values(ascending=False)
    top_missing = {col: round(val, 2) for col, val in missing_share.head(10).items()}

    report: dict[str, object] = {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "n_numeric": int(df.select_dtypes(include=["number"]).shape[1]),
        "n_categorical": int(df.select_dtypes(exclude=["number"]).shape[1]),
        "target_missing_pct": round(float(df[target].isna().mean() * 100), 4),
        "top_missing_pct": top_missing,
    }
    return report
