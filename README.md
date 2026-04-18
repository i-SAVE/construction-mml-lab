# Construction MML Lab

Репозиторий для MML/ML-моделей по направлению **системный анализ и управление**.

## Главное по вашему вопросу

- **XGBoost никуда не делся**: это модель по умолчанию (`--model xgboost`).
- Для реальных данных добавлен отдельный этап улучшения датасета перед обучением.
- Jupyter-ноутбуки — **опциональны** и не нужны для запуска пайплайна.

## Быстрый рабочий контур (без ноутбуков)

### 1) Подготовка реальных данных
```bash
bash scripts/prepare_real_data.sh
```

### 2) Обучение XGBoost на очищенных данных
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

## Что делает улучшение датасета

Скрипт `src/data/prepare_real_dataset.py`:
- удаляет дубликаты;
- удаляет строки с пустым `target`;
- удаляет признаки с очень большим числом пропусков;
- удаляет константные признаки;
- заполняет пропуски (числовые — медианой, категориальные — модой);
- делает клиппинг выбросов по квантилям.

## Нужны ли notebook-файлы?

Нет, они не обязательны для обучения и деплоя.
- Если мешают, можно не использовать `notebooks/` вообще.
- Весь основной pipeline работает только через `scripts/` и `src/`.

## Артефакты обучения

После запуска `train_tabular.py` в `outputs/tabular`:
- `metrics.json`;
- `validation_predictions.csv`;
- `validation_parity_plot.png`.
