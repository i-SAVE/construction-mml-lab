# Подключаем будущие аннотации типов.
from __future__ import annotations

# Импортируем argparse для CLI.
import argparse
# Импортируем json для сохранения результатов сравнения.
import json
# Импортируем Path для файловой системы.
from pathlib import Path

# Импортируем numpy для лог-трансформации target.
import numpy as np
# Импортируем pandas для сохранения таблицы результатов.
import pandas as pd
# Импортируем препроцессинг.
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Импортируем модуль очистки данных.
from src.data.quality import clean_training_frame
# Импортируем загрузчик CSV.
from src.data.tabular_loader import load_csv
# Импортируем фабрику моделей.
from src.models.baseline_xgb import build_regressor


# Парсинг аргументов командной строки.
def parse_args() -> argparse.Namespace:
    # Создаём parser.
    parser = argparse.ArgumentParser()
    # Путь к датасету.
    parser.add_argument("--data", type=str, required=True)
    # Название target.
    parser.add_argument("--target", type=str, default="SalePrice")
    # Список моделей для сравнения.
    parser.add_argument(
        "--models",
        nargs="+",
        default=["xgboost", "gradient_boosting", "catboost"],
        help="Models to compare: xgboost gradient_boosting catboost",
    )
    # Количество fold-ов.
    parser.add_argument("--cv-folds", type=int, default=5)
    # Папка вывода отчёта.
    parser.add_argument("--save-dir", type=str, default="outputs/model_benchmark")
    # Возвращаем аргументы.
    return parser.parse_args()


# Функция строит общий препроцессор.
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    # Собираем числовые признаки.
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    # Собираем категориальные признаки.
    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()
    # Возвращаем трансформер.
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                cat_cols,
            ),
        ]
    )


# Функция запуска сравнения моделей.
def main() -> None:
    # Читаем аргументы.
    args = parse_args()
    # Загружаем исходный CSV.
    raw = load_csv(args.data)
    # Чистим данные.
    clean_df, _ = clean_training_frame(raw, target=args.target, missing_threshold=0.4)

    # Разделяем признаки/target.
    X = clean_df.drop(columns=[args.target])
    y = np.log1p(clean_df[args.target])

    # Если есть Id — удаляем.
    if "Id" in X.columns:
        X = X.drop(columns=["Id"])

    # Строим препроцессор.
    preprocessor = build_preprocessor(X)

    # Настраиваем CV.
    cv = KFold(n_splits=args.cv_folds, shuffle=True, random_state=42)

    # Контейнер результатов.
    rows: list[dict[str, float | str]] = []

    # Проходим по выбранным моделям.
    for model_name in args.models:
        # Пробуем создать модель.
        try:
            model = build_regressor(model_name=model_name)
        except ImportError as exc:
            # Если библиотека модели не установлена, пропускаем модель с пояснением.
            print(f"Skip model '{model_name}': {exc}")
            continue

        # Собираем pipeline.
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])

        # Считаем кросс-валидационные метрики.
        scores = cross_validate(
            pipe,
            X,
            y,
            cv=cv,
            scoring={
                "rmse": "neg_root_mean_squared_error",
                "mae": "neg_mean_absolute_error",
                "r2": "r2",
            },
            n_jobs=-1,
        )

        # Добавляем агрегированный результат.
        rows.append(
            {
                "model": model_name,
                "cv_rmse_log_mean": float(-scores["test_rmse"].mean()),
                "cv_mae_log_mean": float(-scores["test_mae"].mean()),
                "cv_r2_mean": float(scores["test_r2"].mean()),
            }
        )

    # Формируем таблицу результатов.
    result_df = pd.DataFrame(rows).sort_values(by="cv_rmse_log_mean", ascending=True)

    # Готовим директорию вывода.
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Сохраняем CSV-таблицу.
    result_df.to_csv(save_dir / "benchmark_results.csv", index=False)
    # Сохраняем JSON.
    (save_dir / "benchmark_results.json").write_text(result_df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")

    # Печатаем лидера сравнения.
    if not result_df.empty:
        best = result_df.iloc[0]
        print("Best model:", best["model"])
        print(result_df)
    else:
        print("No models were evaluated. Check installed libraries and --models list.")


# Точка входа.
if __name__ == "__main__":
    # Запускаем сравнение.
    main()
