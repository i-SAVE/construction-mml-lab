from __future__ import annotations

from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor


def build_xgb_regressor(random_state: int = 42) -> XGBRegressor:
    """Create a reasonable XGBoost baseline regressor for tabular data."""
    return XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=random_state,
    )


def build_gradient_boosting_regressor(random_state: int = 42) -> GradientBoostingRegressor:
    """Create a lightweight sklearn gradient boosting fallback."""
    return GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=3,
        random_state=random_state,
    )


def build_regressor(model_name: str = "xgboost", random_state: int = 42):
    """Return a regressor by model name."""
    normalized = model_name.strip().lower()
    if normalized in {"xgboost", "xgb"}:
        return build_xgb_regressor(random_state=random_state)
    if normalized in {"gradient_boosting", "gbr", "sklearn_gbr"}:
        return build_gradient_boosting_regressor(random_state=random_state)
    raise ValueError(
        "Unknown model_name. Use one of: xgboost, xgb, gradient_boosting, gbr, sklearn_gbr."
    )
