#!/usr/bin/env bash
set -e

python -m src.train_tabular --data data/raw/unknown_data.csv --target SalePrice
