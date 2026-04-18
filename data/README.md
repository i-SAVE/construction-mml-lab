Сюда класть данные.

- `raw/` — исходные данные без изменений.
- `processed/` — подготовленные наборы.

Минимальный набор для табличного обучения:
- `unknown_data.csv` или `real_construction_data.csv`.

Рекомендуемые CSV для научного контура:
- `real_construction_data.csv` — реальные табличные данные для `train_tabular.py`;
- `graph_nodes.csv` — узлы графа проекта (должен содержать `node_id`);
- `graph_edges.csv` — рёбра графа (`source`, `target`, опц. `edge_type`).
