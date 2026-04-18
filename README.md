# Construction MML Lab

Репозиторий для MML/ML-моделей по направлению **системный анализ и управление**.

## Что внутри

1. **Базовый рабочий контур для табличных данных**  
   Подходит для твоего текущего `unknown_data.csv` с целевой переменной `SalePrice`.  
   Модель: `XGBoost` или `GradientBoostingRegressor` как fallback.

2. **Шаблон графовой модели для строительных проектов**  
   Подходит под научную статью по теме:
   - узлы = работы / ресурсы / рисковые события;
   - рёбра = технологические, ресурсные и информационные связи;
   - модель = `GATv2 / GraphSAGE` в `PyTorch Geometric`.

## Структура

```text
construction-mml-repo/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── default.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── notebooks/
│   └── README.md
├── scripts/
│   ├── run_tabular.sh
│   └── run_gnn.sh
├── src/
│   ├── train_tabular.py
│   ├── train_gnn.py
│   ├── evaluate.py
│   ├── data/
│   │   ├── tabular_loader.py
│   │   └── graph_builder.py
│   ├── models/
│   │   ├── baseline_xgb.py
│   │   └── gnn_model.py
│   └── utils/
│       └── metrics.py
└── tests/
    └── test_smoke.py
```

## Быстрый старт

### 1. Табличная модель на твоём CSV
```bash
python -m src.train_tabular --data data/raw/unknown_data.csv --target SalePrice --model xgboost
```



### 1.1. Что именно за модель сейчас
- По умолчанию запускается **XGBoost** (`--model xgboost`).
- Можно переключить на `sklearn`-вариант: `--model gradient_boosting`.
- Перед обучением скрипт печатает отчёт по качеству данных и удаляет:
  - дубликаты строк;
  - признаки с большой долей пропусков (`--missing-threshold`, по умолчанию `0.4`).

### 2. Шаблон GNN
```bash
python -m src.train_gnn
```

## Научная логика

Для статьи я рекомендую связку:
- **baseline:** XGBoost + SHAP;
- **основная модель:** GNN (GraphSAGE / GATv2);
- **сравнение:** качество + интерпретируемость + системные связи.

## Что нужно сделать после создания GitHub-репозитория

1. Создать пустой приватный репозиторий, например `construction-mml-lab`
2. Загрузить туда содержимое этой папки
3. При необходимости я дополню:
   - EDA-ноутбук,
   - обучение на реальных строительных данных,
   - SHAP-анализ,
   - графовую схему проекта,
   - текст методики для статьи.
