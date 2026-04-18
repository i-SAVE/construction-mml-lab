# Подключаем будущие аннотации типов для более аккуратной типизации.
from __future__ import annotations

# Импортируем градиентный бустинг из sklearn как альтернативную модель.
from sklearn.ensemble import GradientBoostingRegressor
# Импортируем регрессионную модель XGBoost как основной baseline.
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
    # Докстринг поясняет назначение функции.
    """Create an improved XGBoost baseline regressor for tabular data."""
    # Возвращаем экземпляр модели с более сильной регуляризацией и устойчивыми настройками.
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


# Функция создаёт CatBoost-модель (если пакет доступен).
def build_catboost_regressor(random_state: int = 42):
    # Проверяем, доступен ли класс CatBoostRegressor.
    if CatBoostRegressor is None:
        # Даём понятную ошибку с инструкцией по установке.
        raise ImportError("catboost is not installed. Install it via: pip install catboost")
    # Возвращаем CatBoost с устойчивыми дефолтами для регрессии.
    return CatBoostRegressor(
        iterations=1200,
        learning_rate=0.03,
        depth=6,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=random_state,
        verbose=False,
    )


# Универсальная фабрика моделей по имени.
def build_regressor(model_name: str = "xgboost", random_state: int = 42):
    # Докстринг объясняет, что возвращается модель по строковому ключу.
    """Return a regressor by model name."""
    # Нормализуем имя: убираем пробелы и переводим в нижний регистр.
    normalized = model_name.strip().lower()
    # Если пользователь запросил XGBoost — возвращаем его.
    if normalized in {"xgboost", "xgb"}:
        # Создаём и возвращаем XGBoost-модель.
        return build_xgb_regressor(random_state=random_state)
    # Если пользователь запросил sklearn-boosting — возвращаем его.
    if normalized in {"gradient_boosting", "gbr", "sklearn_gbr"}:
        # Создаём и возвращаем sklearn-модель.
        return build_gradient_boosting_regressor(random_state=random_state)
    # Если пользователь запросил CatBoost — возвращаем его.
    if normalized in {"catboost", "cat"}:
        # Создаём и возвращаем CatBoost-модель.
        return build_catboost_regressor(random_state=random_state)
    # Если имя неизвестно — бросаем понятную ошибку.
    raise ValueError(
        # Сообщаем допустимые варианты имени модели.
        "Unknown model_name. Use one of: xgboost, xgb, gradient_boosting, gbr, sklearn_gbr, catboost, cat."
    )
