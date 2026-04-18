# Подключаем будущие аннотации типов.
from __future__ import annotations

# Импорт для CLI-аргументов.
import argparse
# Импорт JSON для сохранения итоговых метрик в файл.
import json
# Импорт класса Path для удобной работы с директориями.
from pathlib import Path

# Импорт matplotlib для графиков результатов обучения.
import matplotlib.pyplot as plt
# Импорт numpy для log1p/expm1.
import numpy as np
# Импорт pandas для сохранения предсказаний в CSV.
import pandas as pd
# Импорт трансформера под разные типы колонок.
from sklearn.compose import ColumnTransformer
# Импорт импутации пропусков.
from sklearn.impute import SimpleImputer
# Импорт инструментов для split/CV/тюнинга.
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
# Импорт Pipeline.
from sklearn.pipeline import Pipeline
# Импорт one-hot encoder.
from sklearn.preprocessing import OneHotEncoder

# Импорт отчёта и очистки качества данных.
from src.data.quality import build_data_report, clean_training_frame
# Импорт загрузки CSV.
from src.data.tabular_loader import load_csv
# Импорт фабрики моделей.
from src.models.baseline_xgb import build_regressor
# Импорт вычисления метрик.
from src.utils.metrics import regression_metrics


# Функция парсинга аргументов командной строки.
def parse_args():
    # Создаём parser.
    parser = argparse.ArgumentParser()
    # Путь к данным.
    parser.add_argument("--data", type=str, required=True)
    # Название целевого столбца.
    parser.add_argument("--target", type=str, default="SalePrice")
    # Выбор семейства модели.
    parser.add_argument(
        "--model",
        type=str,
        default="xgboost",
        choices=["xgboost", "gradient_boosting"],
        help="Model to train: xgboost or sklearn gradient boosting.",
    )
    # Порог пропусков, выше которого признаки удаляются.
    parser.add_argument(
        "--missing-threshold",
        type=float,
        default=0.4,
        help="Drop feature columns with missing share greater than this value.",
    )
    # Включение гиперпараметрического тюнинга.
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable RandomizedSearchCV hyperparameter tuning.",
    )
    # Количество случайных конфигураций для тюнинга.
    parser.add_argument(
        "--n-iter",
        type=int,
        default=20,
        help="Number of random parameter combinations for tuning.",
    )
    # Количество fold-ов в KFold.
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of CV folds for hyperparameter tuning.",
    )
    # Директория для сохранения артефактов (метрики, графики, предсказания).
    parser.add_argument(
        "--save-dir",
        type=str,
        default="outputs/tabular",
        help="Directory to save metrics, predictions and training plots.",
    )
    # Возвращаем распарсенные аргументы.
    return parser.parse_args()


# Функция строит пространство поиска гиперпараметров.
def build_param_distributions(model_name: str) -> dict[str, list]:
    # Для XGBoost используем расширенную сетку.
    if model_name == "xgboost":
        # Возвращаем пространство по префиксу pipeline шага model__.
        return {
            "model__n_estimators": [600, 900, 1200, 1500],
            "model__max_depth": [4, 5, 6, 7],
            "model__learning_rate": [0.01, 0.02, 0.03, 0.05],
            "model__subsample": [0.75, 0.85, 0.95],
            "model__colsample_bytree": [0.7, 0.8, 0.9],
            "model__min_child_weight": [1, 2, 4],
            "model__gamma": [0.0, 0.05, 0.1],
            "model__reg_lambda": [0.8, 1.2, 2.0],
            "model__reg_alpha": [0.0, 0.02, 0.08],
        }
    # Для sklearn gradient boosting используем свою сетку.
    return {
        "model__n_estimators": [300, 500, 700, 900],
        "model__learning_rate": [0.01, 0.02, 0.03, 0.05],
        "model__max_depth": [2, 3, 4],
        "model__subsample": [0.7, 0.85, 1.0],
    }


# Функция сохраняет график "факт vs прогноз" для статьи.
def save_prediction_plot(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    # Создаём фигуру.
    plt.figure(figsize=(7, 7))
    # Точки реальных/предсказанных значений.
    plt.scatter(y_true, y_pred, alpha=0.5, s=18)
    # Берём общий минимум для диагонали идеального прогноза.
    min_val = min(float(np.min(y_true)), float(np.min(y_pred)))
    # Берём общий максимум для диагонали идеального прогноза.
    max_val = max(float(np.max(y_true)), float(np.max(y_pred)))
    # Рисуем линию y=x.
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    # Подпись оси X.
    plt.xlabel("True values")
    # Подпись оси Y.
    plt.ylabel("Predicted values")
    # Заголовок графика.
    plt.title("Validation: true vs predicted")
    # Добавляем сетку.
    plt.grid(alpha=0.3)
    # Плотно упаковываем подписи.
    plt.tight_layout()
    # Сохраняем картинку.
    plt.savefig(output_path, dpi=150)
    # Закрываем фигуру для освобождения памяти.
    plt.close()


# Главная функция обучения.
def main():
    # Читаем аргументы.
    args = parse_args()
    # Загружаем исходные данные.
    raw_df = load_csv(args.data)

    # Строим отчёт по качеству данных.
    report = build_data_report(raw_df, args.target)
    # Печатаем заголовок.
    print("Dataset report:")
    # Печатаем каждый пункт отчёта.
    for key, value in report.items():
        print(f"{key}: {value}")

    # Очищаем данные.
    df, quality_summary = clean_training_frame(
        raw_df,
        target=args.target,
        missing_threshold=args.missing_threshold,
    )
    # Печатаем результаты очистки.
    print("\nCleaning summary:")
    print(f"rows_before: {quality_summary.rows_before}")
    print(f"rows_after: {quality_summary.rows_after}")
    print(f"duplicates_removed: {quality_summary.duplicates_removed}")
    print(f"dropped_columns: {quality_summary.dropped_columns}")

    # Проверяем наличие target.
    if args.target not in df.columns:
        # Бросаем ошибку, если target отсутствует.
        raise ValueError(f"Target column '{args.target}' not found")

    # Проверяем пропуски в target.
    if df[args.target].isna().any():
        # Останавливаем обучение, если target содержит NaN.
        raise ValueError("Target contains missing values after cleaning. Fill or remove them first.")

    # Формируем матрицу признаков.
    X = df.drop(columns=[args.target])
    # Лог-трансформируем целевую для стабильного обучения.
    y = np.log1p(df[args.target])

    # Удаляем технический ID, чтобы не ловить утечку.
    if "Id" in X.columns:
        X = X.drop(columns=["Id"])

    # Получаем список числовых колонок.
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    # Получаем список категориальных колонок.
    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    # Создаём общий препроцессор.
    preprocessor = ColumnTransformer(
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

    # Создаём модель.
    model = build_regressor(model_name=args.model)
    # Объединяем препроцессинг и модель в один pipeline.
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    # Делим на train/valid.
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    # Если включён тюнинг — запускаем RandomizedSearchCV.
    if args.tune:
        # Настройка кросс-валидации.
        cv = KFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
        # Пространство поиска параметров.
        distributions = build_param_distributions(args.model)
        # Инициализируем RandomizedSearchCV.
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=distributions,
            n_iter=args.n_iter,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            cv=cv,
            verbose=1,
            random_state=42,
        )
        # Обучаем подбор параметров.
        search.fit(X_train, y_train)
        # Забираем лучшую модель.
        pipeline = search.best_estimator_
        # Печатаем лучший CV RMSE.
        print("\nBest CV RMSE (log-space):", -search.best_score_)
        # Печатаем лучшие параметры.
        print("Best params:", search.best_params_)
    else:
        # Без тюнинга просто обучаем базовый pipeline.
        pipeline.fit(X_train, y_train)

    # Прогнозы в log-space.
    pred_log = pipeline.predict(X_valid)
    # Обратное преобразование истинных значений.
    y_true = np.expm1(y_valid)
    # Обратное преобразование прогнозов.
    y_pred = np.expm1(pred_log)
    # Считаем метрики.
    metrics = regression_metrics(y_true, y_pred)

    # Создаём директорию под артефакты.
    save_dir = Path(args.save_dir)
    # Создаём директорию даже если её нет.
    save_dir.mkdir(parents=True, exist_ok=True)

    # Сохраняем метрики в JSON.
    with (save_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Сохраняем таблицу предсказаний.
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(save_dir / "validation_predictions.csv", index=False)

    # Сохраняем parity-график для статьи.
    save_prediction_plot(y_true=y_true, y_pred=y_pred, output_path=save_dir / "validation_parity_plot.png")

    # Печатаем выбранную модель.
    print("\nModel:", args.model)
    # Печатаем режим тюнинга.
    print("Tuning:", args.tune)
    # Печатаем куда сохранены результаты.
    print("Saved artifacts:", str(save_dir))
    # Печатаем метрики.
    print("Validation metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


# Точка входа.
if __name__ == "__main__":
    # Запуск обучения.
    main()
