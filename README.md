# Construction MML Lab

Репозиторий для ML/MML-моделей в задачах строительства.

## Коротко: что лучше сравнивать и надо ли это

Да, **сравнение методов нужно**, если цель — научная статья или обоснованный выбор production-модели.
Практичный минимум для ваших данных:
1. `XGBoost` (сильный baseline для табличных данных);
2. `CatBoost` (часто лучший на смешанных числовых+категориальных фичах);
3. `GradientBoostingRegressor` из `scikit-learn` (прозрачный классический baseline).

`TensorFlow/Keras/FastAI` для этого табличного кейса обычно не обязательны. Их стоит добавлять, если есть отдельная DL-гипотеза (например, multimodal данные, изображения со стройки и т.д.).

## Где XGBoost

- XGBoost — дефолтная модель в запуске обучения (`--model xgboost`).
- В `scripts/run_tabular.sh` зафиксирован запуск именно на XGBoost.

## Production-flow без notebooks

### 1) Подготовка реального датасета
```bash
bash scripts/prepare_real_data.sh
```

### 2) Обучение XGBoost
```bash
python -m src.train_tabular \
  --data data/processed/real_construction_data_clean.csv \
  --target SalePrice \
  --model xgboost \
  --tune \
  --n-iter 20 \
  --cv-folds 5 \
  --save-dir outputs/tabular
```

### 3) Сравнение моделей (XGBoost / CatBoost / sklearn GBR)
```bash
python -m src.benchmark_tabular_models \
  --data data/processed/real_construction_data_clean.csv \
  --target SalePrice \
  --models xgboost catboost gradient_boosting \
  --cv-folds 5 \
  --save-dir outputs/model_benchmark
```

## Что делает подготовка реальных данных

`src/data/prepare_real_dataset.py`:
- удаляет дубликаты;
- удаляет строки с пустым target;
- удаляет признаки с высокой долей пропусков;
- удаляет константные признаки;
- импутирует пропуски;
- клиппирует выбросы по квантилям;
- сохраняет очищенный CSV + summary JSON.

## Нужны ли notebook-файлы

Нет, не обязательно. Основной рабочий контур полностью запускается через `scripts/` и `src/`.

## Артефакты

- `outputs/tabular/metrics.json`
- `outputs/tabular/validation_predictions.csv`
- `outputs/tabular/validation_parity_plot.png`
- `outputs/model_benchmark/benchmark_results.csv`
- `outputs/model_benchmark/benchmark_results.json`
