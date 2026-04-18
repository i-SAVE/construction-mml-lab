from src.models.baseline_xgb import build_regressor


def test_build_regressor_xgb_alias():
    model = build_regressor("xgb")
    assert model.__class__.__name__ == "XGBRegressor"


def test_build_regressor_gbr_alias():
    model = build_regressor("gbr")
    assert model.__class__.__name__ == "GradientBoostingRegressor"
