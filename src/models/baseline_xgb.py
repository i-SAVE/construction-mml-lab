# Подключаем будущие аннотации типов для более аккуратной типизации.
from __future__ import annotations

from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

# Пробуем импортировать CatBoost (опционально).
try:
    # Импортируем CatBoostRegressor, если библиотека установлена.
    from catboost import CatBoostRegressor
except Exception:  # noqa: BLE001
    # Если catboost не установлен, оставляем заглушку None.
    CatBoostRegressor = None

# Функция создаёт конфигурацию XGBoost для регрессии.
def build_xgb_regressor(random_state: int = 42) -> XGBRegressor:
    """Create a reasonable XGBoost baseline regressor for tabular data."""
    return XGBRegressor(
        # Число деревьев в ансамбле.
        n_estimators=1200,
        # Максимальная глубина дерева (контроль сложности).
        max_depth=6,
        # Шаг обучения (меньше -> стабильнее, но дольше обучается).
        learning_rate=0.02,
        # Доля объектов для каждого дерева (bagging-эффект).
        subsample=0.85,
        # Доля признаков для каждого дерева.
        colsample_bytree=0.8,
        # Минимальная сумма весов в листе (защита от переобучения).
        min_child_weight=2,
        # Минимальное уменьшение ошибки для разбиения.
        gamma=0.05,
        # L2-регуляризация весов листьев.
        reg_lambda=1.2,
        # L1-регуляризация весов листьев.
        reg_alpha=0.02,
        # Функция потерь для регрессии.
        objective="reg:squarederror",
        # Оптимизированный метод построения деревьев.
        tree_method="hist",
        # Случайность для воспроизводимости.
        random_state=random_state,
        # Количество потоков CPU.
        n_jobs=-1,
    )


# Функция создаёт альтернативный sklearn-бустинг.
def build_gradient_boosting_regressor(random_state: int = 42) -> GradientBoostingRegressor:
    # Кратко описываем смысл альтернативы.
    """Create a stronger sklearn gradient boosting fallback."""
    # Возвращаем модель с настройками, близкими к стабильному baseline.
    return GradientBoostingRegressor(
        # Количество слабых моделей.
        n_estimators=700,
        # Шаг обучения.
        learning_rate=0.03,
        # Глубина базовых деревьев.
        max_depth=3,
        # Доля обучающей выборки на итерацию.
        subsample=0.9,
        # Критерий выбора разбиений.
        criterion="friedman_mse",
        # Случайность для воспроизводимости.
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
