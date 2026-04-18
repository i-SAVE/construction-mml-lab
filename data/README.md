Сюда класть данные.

- `raw/` — исходные данные без изменений.
- `processed/` — очищенные/подготовленные наборы.

Для реального обучения:
- вход: `data/raw/real_construction_data.csv`
- выход после подготовки: `data/processed/real_construction_data_clean.csv`

Подготовка запускается так:
```bash
bash scripts/prepare_real_data.sh
```
