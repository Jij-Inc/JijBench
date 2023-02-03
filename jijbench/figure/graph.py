from enum import Enum, auto
import networkx as nx
import numpy.typing as npt
from typing import Dict, List, Optional, Union

from jijbench.figure.interface import Figure


class GraphType(Enum):
    UNDIRECTED = auto()
    DIRECTED = auto()


graph_modules = {
    GraphType.UNDIRECTED: nx.Graph,
    GraphType.DIRECTED: nx.DiGraph,
}


class Graph(Figure):
    def __init__(
        self,
        G: nx.Graph,
    ):
        self.G = G

    def show(
        self,
        node_pos,
    ):
        G = self.G

        nx.draw_networkx_nodes(G, node_pos)

    @classmethod
    def from_edge_list(
        cls,
        edge_list: List[List[int]],
        graphtype: GraphType,
    ):
        G = graph_modules[graphtype]()
        node_list = list(set([node for edge in edge_list for node in edge]))
        G.add_nodes_from(node_list)
        G.add_edges_from(edge_list)
        return cls(G)

    @classmethod
    def from_distance_matrix(
        cls,
        distance_matrix: Union[List[List], npt.NDArray],
        graphtype: GraphType,
    ):
        G = graph_modules[graphtype]()
        node_list = [i for i, _ in enumerate(distance_matrix)]
        edge_list = [
            (i, j, w)
            for i, ws in enumerate(distance_matrix)
            for j, w in enumerate(ws)
            if i != j
        ]
        G.add_nodes_from(node_list)
        G.add_weighted_edges_from(edge_list)
        return cls(G)
