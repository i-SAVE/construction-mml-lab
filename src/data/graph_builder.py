# Подключаем будущие аннотации типов.
from __future__ import annotations

# Импортируем библиотеку для графов.
import networkx as nx
# Импортируем pandas для табличных данных узлов/рёбер.
import pandas as pd
# Импортируем torch для тензоров.
import torch
# Импортируем контейнер графовых данных из PyG.
from torch_geometric.data import Data


# Функция строит ориентированный граф WBS из двух таблиц.
def build_wbs_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> nx.DiGraph:
    # Докстринг кратко описывает назначение функции.
    """Build a directed WBS graph from node and edge tables."""
    # Создаём пустой ориентированный граф.
    graph = nx.DiGraph()

    # Проходим по строкам таблицы узлов.
    for _, row in nodes_df.iterrows():
        # Берём ID узла.
        node_id = row["node_id"]
        # Все остальные колонки считаем атрибутами узла.
        attrs = row.drop(labels=["node_id"]).to_dict()
        # Добавляем узел с атрибутами.
        graph.add_node(node_id, **attrs)

    # Проходим по строкам таблицы рёбер.
    for _, row in edges_df.iterrows():
        # Добавляем ребро source -> target и тип связи.
        graph.add_edge(row["source"], row["target"], edge_type=row.get("edge_type", "dependency"))

    # Возвращаем собранный граф.
    return graph


# Функция-конвертер из NetworkX в минимальный Data PyG.
def nx_to_pyg_stub(graph: nx.DiGraph) -> Data:
    # Докстринг объясняет, что это стартовый каркас.
    """Convert a NetworkX graph to a minimal PyG Data object stub.

    This is a scaffold and should be extended once node features are finalized.
    """
    # Получаем список узлов в фиксированном порядке.
    nodes = list(graph.nodes())
    # Строим отображение node_id -> индекс строки в тензоре признаков.
    node_index = {node: idx for idx, node in enumerate(nodes)}
    # Преобразуем рёбра в пары индексов.
    edge_pairs = [[node_index[u], node_index[v]] for u, v in graph.edges()]
    # Собираем edge_index формата [2, E], либо пустой тензор если рёбер нет.
    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous() if edge_pairs else torch.empty((2, 0), dtype=torch.long)
    # Создаём заглушку признаков узлов (1 признак на узел, пока нули).
    x = torch.zeros((len(nodes), 1), dtype=torch.float32)
    # Возвращаем объект PyG Data.
    return Data(x=x, edge_index=edge_index)
