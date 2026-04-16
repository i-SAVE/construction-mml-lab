from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_csv(data_path: str | Path) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path)


def split_features_target(df: pd.DataFrame, target: str):
    """Split a dataframe into features and target."""
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe.")
    X = df.drop(columns=[target])
    y = df[target]
    return X, y
