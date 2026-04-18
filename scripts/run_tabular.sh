#!/usr/bin/env bash
# Строгий режим: при ошибке команда завершится сразу.
set -euo pipefail

# Переходим в корень репозитория независимо от точки запуска.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Активируем каталог репозитория.
cd "$REPO_ROOT"

# Запускаем табличное обучение с тюнингом и сохранением графиков/метрик для статьи.
python -m src.train_tabular \
  --data data/raw/unknown_data.csv \
  --target SalePrice \
  --model xgboost \
  --tune \
  --n-iter 20 \
  --cv-folds 5 \
  --save-dir outputs/tabular
