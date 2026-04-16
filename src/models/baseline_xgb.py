from __future__ import annotations

from xgboost import XGBRegressor


def build_xgb_regressor(random_state: int = 42) -> XGBRegressor:
    """Create a reasonable baseline regressor for tabular housing-like data."""
    return XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=random_state,
    )
