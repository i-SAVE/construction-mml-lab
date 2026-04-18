#!/usr/bin/env bash
# Строгий режим shell.
set -euo pipefail

# Переходим в корень репозитория.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Подготавливаем реальный датасет для обучения.
python -m src.data.prepare_real_dataset \
  --input data/raw/real_construction_data.csv \
  --output data/processed/real_construction_data_clean.csv \
  --target SalePrice
