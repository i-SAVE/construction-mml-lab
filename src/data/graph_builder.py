from __future__ import annotations

import networkx as nx
import pandas as pd
import torch
from torch_geometric.data import Data


def build_wbs_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> nx.DiGraph:
    """Build a directed WBS graph from node and edge tables."""
    graph = nx.DiGraph()

    for _, row in nodes_df.iterrows():
        node_id = row["node_id"]
        attrs = row.drop(labels=["node_id"]).to_dict()
        graph.add_node(node_id, **attrs)

    for _, row in edges_df.iterrows():
        graph.add_edge(row["source"], row["target"], edge_type=row.get("edge_type", "dependency"))

    return graph


def nx_to_pyg_stub(graph: nx.DiGraph) -> Data:
    """Convert a NetworkX graph to a minimal PyG Data object stub.

    This is a scaffold and should be extended once node features are finalized.
    """
    nodes = list(graph.nodes())
    node_index = {node: idx for idx, node in enumerate(nodes)}
    edge_pairs = [[node_index[u], node_index[v]] for u, v in graph.edges()]
    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous() if edge_pairs else torch.empty((2, 0), dtype=torch.long)
    x = torch.zeros((len(nodes), 1), dtype=torch.float32)
    return Data(x=x, edge_index=edge_index)
