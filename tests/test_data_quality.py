# Импортируем pandas для подготовки тестового датафрейма.
import pandas as pd

# Импортируем функцию очистки данных.
from src.data.quality import clean_training_frame


# Проверяем удаление дублей и колонок с высокой долей пропусков.
def test_clean_training_frame_removes_duplicates_and_high_missing_columns():
    # Формируем небольшой тестовый набор.
    df = pd.DataFrame(
        {
            "feature_ok": [1, 1, 2],
            "feature_bad": [None, None, 5],
            "SalePrice": [10, 10, 20],
        }
    )

    # Запускаем очистку с порогом пропусков.
    cleaned, summary = clean_training_frame(df, target="SalePrice", missing_threshold=0.49)

    # Проверяем число удалённых дублей.
    assert summary.duplicates_removed == 1
    # Проверяем, что плохая колонка удалена.
    assert "feature_bad" in summary.dropped_columns
    # Проверяем, что в таблице колонка отсутствует.
    assert "feature_bad" not in cleaned.columns
