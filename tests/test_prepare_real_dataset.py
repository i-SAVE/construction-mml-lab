# Импортируем pandas для тестового датафрейма.
import pandas as pd

# Импортируем функцию подготовки реальных данных.
from src.data.prepare_real_dataset import improve_real_dataset


# Проверяем, что очистка удаляет дубли/плохие колонки и заполняет пропуски.
def test_improve_real_dataset_basic_cleanup():
    # Формируем небольшой искусственный набор.
    df = pd.DataFrame(
        {
            "SalePrice": [100, 100, 200, None],
            "feature 1": [1.0, 1.0, 10_000.0, 5.0],
            "mostly_nan": [None, None, None, 1.0],
            "const_col": ["a", "a", "a", "a"],
        }
    )

    # Запускаем подготовку.
    clean, summary = improve_real_dataset(df, target="SalePrice", missing_threshold=0.7)

    # Проверяем удаление дублей.
    assert summary.duplicates_removed == 1
    # Проверяем удаление строки с пустым target.
    assert summary.empty_target_rows_removed == 1
    # Проверяем, что колонка с очень большим числом пропусков удалена.
    assert "mostly_nan" in summary.high_missing_columns_removed
    # Проверяем, что константная колонка удалена.
    assert summary.constant_columns_removed >= 1
    # Проверяем, что пробелы в названиях колонок нормализуются.
    assert "feature_1" in clean.columns
