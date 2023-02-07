from enum import Enum, auto
import matplotlib.pyplot as plt
import networkx as nx
import numpy.typing as npt
from typing import Dict, List, Optional, Tuple, Union

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
        figsize: Optional[Tuple[Union[int, float]]] = None,
        title: Optional[str] = None,
        node_pos=None,
        node_color: Optional[Union[str, List[str]]] = None,
        edge_color: Optional[Union[str, List[str]]] = None,
        node_labels: Optional[Dict] = None,
        edge_labels: Optional[Dict] = None,
    ):
        G = self.G

        if title is None:
            title = "graph"

        if node_pos is None:
            node_pos = nx.spring_layout(G, seed=1)

        if node_color is None:
            node_color = "#1f78b4"

        if edge_color is None:
            edge_color = "k"

        if node_labels is None:
            node_labels = {node: str(node) for node in G.nodes}

        if edge_labels is None:
            # If G is not weighted graph, edge_labels will be {}.
            edge_labels = nx.get_edge_attributes(G, "weight")

        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle(title)

        nx.draw_networkx_nodes(
            G=G,
            pos=node_pos,
            node_color=node_color,
            ax=ax,
        )
        nx.draw_networkx_edges(
            G=G,
            pos=node_pos,
            edge_color=edge_color,
            ax=ax,
        )
        nx.draw_networkx_labels(
            G=G,
            pos=node_pos,
            labels=node_labels,
            ax=ax,
        )
        nx.draw_networkx_edge_labels(
            G=G,
            pos=node_pos,
            edge_labels=edge_labels,
            ax=ax,
        )

        self._fig_ax = (fig, ax)

    @property
    def fig_ax(self):
        if self._fig_ax is None:
            raise AttributeError(
                "fig_ax attribute is available after show method is called."
            )
        else:
            return self._fig_ax

    @classmethod
    def from_edge_list(
        cls,
        edge_list: List[List[int]],
        graphtype: GraphType,
    ):
        G = graph_modules[graphtype]()
        node_list = sorted(set([node for edge in edge_list for node in edge]))
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
