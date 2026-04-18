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


# Проверяем, что неизвестное имя модели даёт ошибку.
def test_build_regressor_unknown_model_raises_value_error():
    # Проверяем сообщение об ошибке.
    try:
        build_regressor("unknown_model")
        raise AssertionError("Expected ValueError was not raised")
    except ValueError as exc:
        assert "Unknown model_name" in str(exc)
