from __future__ import annotations

from enum import Enum, auto
import matplotlib.pyplot as plt
import networkx as nx
import numpy.typing as npt
from typing import cast

from jijbench.visualization.figure.interface import Figure

# TODO: node_posなどの型アノテーションを行う


class GraphType(Enum):
    UNDIRECTED = auto()
    DIRECTED = auto()


graph_modules = {
    GraphType.UNDIRECTED: nx.Graph,
    GraphType.DIRECTED: nx.DiGraph,
}


class Graph(Figure):
    """Visualize graph.

    You can also instantiate from edge_list, distance matrix by class method.
    Attributes:
        G (nx.Graph): the networkX Graph instance.
        fig_ax (tuple[matplotlib.figure.Figure, matplotlib.axes.Subplot]): Figure and Axes of matplotlib. Available after show method is called.
    Example:
        The code below visualizes an undirected graph given by edge list.
        The style of the graph (e.g. color) can be changed by arguments of the show method.
        ```python
        >>> from jijbench.figure.graph import Graph, GraphType

        >>> graph = Graph.from_edge_list([[1, 2], [2, 3]], GraphType.UNDIRECTED)
        >>> graph.show(edge_color=["black", "red"])
        ```
    """

    def __init__(
        self,
        G: nx.Graph,
    ):
        self.G = G

    def show(
        self,
        figsize: tuple[int | float] | None = None,
        title: str | None = None,
        node_pos: dict | None = None,
        node_color: (str | list[str]) | None = None,
        edge_color: (str | list[str]) | None = None,
        node_labels: dict | None = None,
        edge_labels: dict | None = None,
    ):
        """Visualize graph.

        The arguments of the show method are passed to the plot of matplotlib, networkx.

        Args:
            figsize (tuple[int | float] | None): the size of figure. The default uses matplotlib's default value.
            title (str | None): the title of figure. Defaults to "graph".
            node_pos (dict | None): dict where the key is node and the value is position (np,array([x, y])). The default uses the return of networkx.spring_layout(self.G, seed=1).
            node_color ((str | list[str]) | None): string or list of node color. Defaults to "#1f78b4".
            edge_color ((str | list[str]) | None): string or list of edge color. Defaults to "k".
            node_labels (dict | None): dict where the key is node and the value is label. The default is {node: str(node) for node in self.G.nodes}.
            edge_labels (dict | None): dict where the key is node and the value is label. The default is {}, or weights for weighted graphs.
        """
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

        # execute typing.cast to the avoid type checking error caused by the networkx interface.
        node_color = cast("str", node_color)
        nx.draw_networkx_nodes(
            G=G,
            pos=node_pos,
            node_color=node_color,
            ax=ax,
        )

        # execute typing.cast to the avoid type checking error caused by the networkx interface.
        edge_color = cast("str", edge_color)
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
        edge_list: list[list[int]],
        graphtype: GraphType,
    ):
        """To Graph instance from edge list.

        Args:
            edge_list (list[list[int]]): list of edges.
            graphtype (GraphType): GraphType instance of jijbench.
        Example:
            ```python
            >>> from jijbench.figure.graph import Graph, GraphType

            >>> edge_list = [[1, 2], [2, 3], [2, 1]]
            >>> graph = Graph.from_edge_list(edge_list, GraphType.DIRECTED)
            ```
        """
        G = graph_modules[graphtype]()
        node_list = sorted(set([node for edge in edge_list for node in edge]))
        G.add_nodes_from(node_list)
        G.add_edges_from(edge_list)
        return cls(G)

    @classmethod
    def from_distance_matrix(
        cls,
        distance_matrix: list[list[int | float]] | npt.NDArray,
        graphtype: GraphType,
    ):
        """To Graph instance from distance matrix.

        Args:
            distance_matrix (list[list[int | float]] | npt.NDArray): distance matrix. No self-loop is added, so the diagonal can be any number.
            graphtype (GraphType): GraphType instance of jijbench.
        Example:
            ```python
            >>> from jijbench.figure.graph import Graph, GraphType

            >>> distance_matrix = [[-1, 2], [2, -1]]
            >>> graph = Graph.from_distance_matrix(distance_matrix, GraphType.UNDIRECTED)
            ```
        """
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
