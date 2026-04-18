Папка `notebooks/` опциональна.

Если вы работаете только через скрипты (`scripts/`) и модули (`src/`), эти файлы не нужны.

Рекомендуемый production-путь без notebook:
1. `bash scripts/prepare_real_data.sh`
2. `python -m src.train_tabular --data data/processed/real_construction_data_clean.csv --target SalePrice --model xgboost --tune`
