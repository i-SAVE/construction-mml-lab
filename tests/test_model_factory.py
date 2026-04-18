# Импортируем фабрику моделей.
from src.models.baseline_xgb import build_regressor


# Проверяем алиас xgb -> XGBRegressor.
def test_build_regressor_xgb_alias():
    # Создаём модель через короткое имя.
    model = build_regressor("xgb")
    # Проверяем класс модели.
    assert model.__class__.__name__ == "XGBRegressor"


# Проверяем алиас gbr -> GradientBoostingRegressor.
def test_build_regressor_gbr_alias():
    # Создаём модель через короткое имя.
    model = build_regressor("gbr")
    # Проверяем класс модели.
    assert model.__class__.__name__ == "GradientBoostingRegressor"
