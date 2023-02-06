from collections import OrderedDict
import matplotlib
import networkx as nx
import numpy as np
import pytest

from jijbench.figure.graph import Graph, GraphType
from jijbench.figure.schedule import Schedule
from jijbench.figure.timeseries import TimeSeries

# TODO: colorについて、strだけじゃなくてrgb tupleにも対応させる（テストの追記と型アノテーションでいけると想定）
# 参考: rgb tupleについての言及 https://networkx.org/documentation/stable/reference/generated/networkx.drawing.nx_pylab.draw_networkx_nodes.html

params = {
    "list case": ("data", [1, 2], [3, 4]),
    "np.ndarray case": ("data", np.array([1, 2]), np.array([3, 4])),
}


@pytest.mark.parametrize(
    "label, plot_x, plot_y",
    list(params.values()),
    ids=list(params.keys()),
)
def test_timeseries_add_data(label, plot_x, plot_y):
    timeseries = TimeSeries()
    timeseries.add_data(label, plot_x, plot_y)

    assert timeseries.data == OrderedDict([("data", ([1, 2], [3, 4]))])


def test_timeseries_add_data_not_same_length():
    timeseries = TimeSeries()

    with pytest.raises(ValueError):
        timeseries.add_data("data", [1, 2], [3, 4, 5])


def test_timeseries_fig_ax_attribute():
    timeseries = TimeSeries()
    timeseries.add_data("data", [1, 2], [3, 4])
    timeseries.show()
    fig, ax = timeseries.fig_ax

    assert type(fig) == matplotlib.figure.Figure
    assert type(ax) == matplotlib.axes.Subplot


def test_timeseries_fig_ax_attribute_before_show():
    timeseries = TimeSeries()
    timeseries.add_data("data", [1, 2], [3, 4])

    with pytest.raises(AttributeError):
        timeseries.fig_ax


def test_timeseries_show_no_plot_data():
    timeseries = TimeSeries()
    with pytest.raises(RuntimeError):
        timeseries.show()


def test_timeseries_show_title():
    title = "title"

    timeseries = TimeSeries()
    timeseries.add_data("data", [1, 2], [3, 4])
    timeseries.show(title=title)
    fig, ax = timeseries.fig_ax

    assert fig.texts[0].get_text() == "title"


def test_timeseries_show_title_default():
    timeseries = TimeSeries()
    timeseries.add_data("data", [1, 2], [3, 4])
    timeseries.show()
    fig, ax = timeseries.fig_ax

    assert fig.texts[0].get_text() == "time series"


def test_timeseries_show_x_and_y():
    x1, y1 = [1, 2], [3, 4]
    x2, y2 = [5, 6], [7, 8]

    timeseries = TimeSeries()
    timeseries.add_data("data1", x1, y1)
    timeseries.add_data("data2", x2, y2)
    timeseries.show()
    fig, ax = timeseries.fig_ax

    assert (ax.get_lines()[0].get_xdata() == np.array(x1)).all()
    assert (ax.get_lines()[0].get_ydata() == np.array(y1)).all()
    assert (ax.get_lines()[1].get_xdata() == np.array(x2)).all()
    assert (ax.get_lines()[1].get_ydata() == np.array(y2)).all()


def test_timeseries_show_arg_figsize():
    figwidth, figheight = 8, 4

    timeseries = TimeSeries()
    timeseries.add_data("data", [1, 2], [3, 4])
    timeseries.show(figsize=tuple([figwidth, figheight]))
    fig, ax = timeseries.fig_ax

    assert fig.get_figwidth() == 8
    assert fig.get_figheight() == 4


def test_timeseries_show_arg_color_list():
    color_list = ["r", "#e41a1c"]

    timeseries = TimeSeries()
    timeseries.add_data("data0", [1, 2], [3, 4])
    timeseries.add_data("data1", [1, 2], [3, 4])
    timeseries.show(color_list=color_list)
    fig, ax = timeseries.fig_ax

    assert ax.get_lines()[0].get_color() == "r"
    assert ax.get_lines()[1].get_color() == "#e41a1c"


def test_timeseries_show_arg_color_list_invalid_length():
    color_list = ["r", "g", "b"]

    timeseries = TimeSeries()
    timeseries.add_data("data0", [1, 2], [3, 4])
    timeseries.add_data("data1", [1, 2], [3, 4])

    with pytest.raises(ValueError):
        timeseries.show(color_list=color_list)


def test_timeseries_show_arg_alpha_list():
    alpha_list = [0.5, 0.7]

    timeseries = TimeSeries()
    timeseries.add_data("data0", [1, 2], [3, 4])
    timeseries.add_data("data1", [1, 2], [3, 4])
    timeseries.show(alpha_list=alpha_list)
    fig, ax = timeseries.fig_ax

    assert ax.get_lines()[0].get_alpha() == 0.5
    assert ax.get_lines()[1].get_alpha() == 0.7


def test_timeseries_show_arg_alpha_list_default():
    timeseries = TimeSeries()
    timeseries.add_data("data0", [1, 2], [3, 4])
    timeseries.add_data("data1", [1, 2], [3, 4])
    timeseries.show()
    fig, ax = timeseries.fig_ax

    assert ax.get_lines()[0].get_alpha() == 1.0
    assert ax.get_lines()[1].get_alpha() == 1.0


def test_timeseries_show_arg_alpha_list_invalid_length():
    alpha_list = [0.1, 0.1, 0.1]

    timeseries = TimeSeries()
    timeseries.add_data("data0", [1, 2], [3, 4])
    timeseries.add_data("data1", [1, 2], [3, 4])

    with pytest.raises(ValueError):
        timeseries.show(alpha_list=alpha_list)


def test_timeseries_show_arg_linestyle_list():
    linestyle_list = ["-", "--"]

    timeseries = TimeSeries()
    timeseries.add_data("data0", [1, 2], [3, 4])
    timeseries.add_data("data1", [1, 2], [3, 4])
    timeseries.show(linestyle_list=linestyle_list)
    fig, ax = timeseries.fig_ax

    assert ax.get_lines()[0].get_linestyle() == "-"
    assert ax.get_lines()[1].get_linestyle() == "--"


def test_timeseries_show_arg_linestyle_list_default():
    timeseries = TimeSeries()
    timeseries.add_data("data0", [1, 2], [3, 4])
    timeseries.add_data("data1", [1, 2], [3, 4])
    timeseries.show()
    fig, ax = timeseries.fig_ax

    assert ax.get_lines()[0].get_linestyle() == "-"
    assert ax.get_lines()[1].get_linestyle() == "-"


def test_timeseries_show_arg_linestyle_list_invalid_length():
    linestyle_list = ["--", "--", "--"]

    timeseries = TimeSeries()
    timeseries.add_data("data0", [1, 2], [3, 4])
    timeseries.add_data("data1", [1, 2], [3, 4])

    with pytest.raises(ValueError):
        timeseries.show(linestyle_list=linestyle_list)


def test_timeseries_show_arg_marker_list():
    marker_list = ["v", "d"]

    timeseries = TimeSeries()
    timeseries.add_data("data0", [1, 2], [3, 4])
    timeseries.add_data("data1", [1, 2], [3, 4])
    timeseries.show(marker_list=marker_list)
    fig, ax = timeseries.fig_ax

    assert ax.get_lines()[0].get_marker() == "v"
    assert ax.get_lines()[1].get_marker() == "d"


def test_timeseries_show_arg_marker_list_default():
    timeseries = TimeSeries()
    timeseries.add_data("data0", [1, 2], [3, 4])
    timeseries.add_data("data1", [1, 2], [3, 4])
    timeseries.show()
    fig, ax = timeseries.fig_ax

    assert ax.get_lines()[0].get_marker() == "o"
    assert ax.get_lines()[1].get_marker() == "o"


def test_timeseries_show_arg_marker_list_invalid_length():
    marker_list = ["v", "v", "v"]

    timeseries = TimeSeries()
    timeseries.add_data("data0", [1, 2], [3, 4])
    timeseries.add_data("data1", [1, 2], [3, 4])

    with pytest.raises(ValueError):
        timeseries.show(marker_list=marker_list)


def test_timeseries_show_arg_xlabel():
    xlabel = "xlabel"

    timeseries = TimeSeries()
    timeseries.add_data("data", [1, 2], [3, 4])
    timeseries.show(xlabel=xlabel)
    fig, ax = timeseries.fig_ax

    assert ax.get_xlabel() == "xlabel"


def test_timeseries_show_arg_ylabel():
    ylabel = "ylabel"

    timeseries = TimeSeries()
    timeseries.add_data("data", [1, 2], [3, 4])
    timeseries.show(ylabel=ylabel)
    fig, ax = timeseries.fig_ax

    assert ax.get_ylabel() == "ylabel"


def test_timeseries_show_arg_xticks():
    xticks = [1.0, 1.5, 2.0]

    timeseries = TimeSeries()
    timeseries.add_data("data", [1, 2], [3, 4])
    timeseries.show(xticks=xticks)
    fig, ax = timeseries.fig_ax

    assert (ax.get_xticks() == np.array([1.0, 1.5, 2.0])).all()


def test_timeseries_show_arg_yticks():
    yticks = [3.0, 3.5, 4.0]

    timeseries = TimeSeries()
    timeseries.add_data("data", [1, 2], [3, 4])
    timeseries.show(yticks=yticks)
    fig, ax = timeseries.fig_ax

    assert (ax.get_yticks() == np.array([3.0, 3.5, 4.0])).all()


params = {
    "list case": ("data", [1, 2], [3, 4], [5.5, 6.6]),
    "np.ndarray case": (
        "data",
        np.array([1, 2]),
        np.array([3, 4]),
        np.array([5.5, 6.6]),
    ),
}


@pytest.mark.parametrize(
    "task_label, workers, start_times, time_lengths",
    list(params.values()),
    ids=list(params.keys()),
)
def test_schedule_add_data(task_label, workers, start_times, time_lengths):
    schedule = Schedule()
    schedule.add_data(task_label, workers, start_times, time_lengths)

    assert schedule.data == OrderedDict([("data", ([1, 2], [3, 4], [5.5, 6.6]))])


params = {
    "workers and start_times are different": ("data", [1, 2], [3, 4, 5], [5.5, 6.6]),
    "workers and time_lengths are different": (
        "data",
        np.array([1, 2]),
        np.array([3, 4]),
        np.array([5.5, 6.6, 7.7]),
    ),
}


@pytest.mark.parametrize(
    "task_label, workers, start_times, time_lengths",
    list(params.values()),
    ids=list(params.keys()),
)
def test_schedule_add_data_not_same_length(
    task_label, workers, start_times, time_lengths
):
    schedule = Schedule()

    with pytest.raises(ValueError):
        schedule.add_data(task_label, workers, start_times, time_lengths)


def test_schedule_fig_ax_attribute():
    workers, start_times, time_lengths = [1, 2], [1, 2], [3, 4]

    schedule = Schedule()
    schedule.add_data("data", workers, start_times, time_lengths)
    schedule.show()
    fig, ax = schedule.fig_ax

    assert type(fig) == matplotlib.figure.Figure
    assert type(ax) == matplotlib.axes.Subplot


def test_schedule_fig_ax_attribute_before_show():
    workers, start_times, time_lengths = [1, 2], [1, 2], [3, 4]

    schedule = Schedule()
    schedule.add_data("data", workers, start_times, time_lengths)

    with pytest.raises(AttributeError):
        schedule.fig_ax


def test_schedule_show_no_plot_data():
    schedule = Schedule()
    with pytest.raises(RuntimeError):
        schedule.show()


def test_schedule_show_title():
    title = "title"

    schedule = Schedule()
    schedule.add_data("data", [1, 2], [3, 4], [5, 6])
    schedule.show(title=title)
    fig, ax = schedule.fig_ax

    assert fig.texts[0].get_text() == "title"


def test_schedule_show_title_default():
    schedule = Schedule()
    schedule.add_data("data", [1, 2], [3, 4], [5, 6])
    schedule.show()
    fig, ax = schedule.fig_ax

    assert fig.texts[0].get_text() == "schedule"


def test_schedule_show_bar():
    workers1, start_times1, time_lengths1 = [1, 2], [1, 2], [3, 4]
    workers2, start_times2, time_lengths2 = [2], [1], [1]

    schedule = Schedule()
    schedule.add_data("data1", workers1, start_times1, time_lengths1)
    schedule.add_data("data2", workers2, start_times2, time_lengths2)
    schedule.show()
    fig, ax = schedule.fig_ax

    assert (ax.containers[0].get_children()[0].get_center() == np.array([2.5, 1])).all()
    assert (ax.containers[0].get_children()[1].get_center() == np.array([4, 2])).all()
    assert (ax.containers[1].get_children()[0].get_center() == np.array([1.5, 2])).all()


def test_schedule_show_text():
    workers, start_times, time_lengths = [1, 2], [1, 2], [3, 4]

    schedule = Schedule()
    schedule.add_data("data", workers, start_times, time_lengths)
    schedule.show()
    fig, ax = schedule.fig_ax

    assert ax.texts[0].get_position() == (2.5, 1)
    assert ax.texts[0].get_text() == "3"
    assert ax.texts[1].get_position() == (4, 2)
    assert ax.texts[1].get_text() == "4"


def test_schedule_show_arg_figsize():
    figwidth, figheight = 8, 4

    schedule = Schedule()
    schedule.add_data("data", [1, 2], [1, 2], [3, 4])
    schedule.show(figsize=tuple([figwidth, figheight]))
    fig, ax = schedule.fig_ax

    assert fig.get_figwidth() == 8
    assert fig.get_figheight() == 4


def test_schedule_show_arg_color_list():
    color_list = ["r", "b"]

    schedule = Schedule()
    schedule.add_data("data0", [1, 2], [1, 2], [3, 4])
    schedule.add_data("data1", [1, 2], [1, 2], [3, 4])
    schedule.show(color_list=color_list)
    fig, ax = schedule.fig_ax

    assert ax.containers[0].get_children()[0].get_facecolor()[0] == 1.0
    assert ax.containers[1].get_children()[0].get_facecolor()[2] == 1.0


def test_schedule_show_arg_color_list_invalid_length():
    color_list = ["r", "g", "b"]

    schedule = Schedule()
    schedule.add_data("data0", [1, 2], [1, 2], [3, 4])
    schedule.add_data("data1", [1, 2], [1, 2], [3, 4])

    with pytest.raises(ValueError):
        schedule.show(color_list=color_list)


def test_schedule_show_arg_alpha_list():
    alpha_list = [0.5, 0.7]

    schedule = Schedule()
    schedule.add_data("data0", [1, 2], [1, 2], [3, 4])
    schedule.add_data("data1", [1, 2], [1, 2], [3, 4])
    schedule.show(alpha_list=alpha_list)
    fig, ax = schedule.fig_ax

    assert ax.containers[0].get_children()[0].get_alpha() == 0.5
    assert ax.containers[1].get_children()[0].get_alpha() == 0.7


def test_schedule_show_arg_alpha_list_default():
    schedule = Schedule()
    schedule.add_data("data0", [1, 2], [1, 2], [3, 4])
    schedule.add_data("data1", [1, 2], [1, 2], [3, 4])
    schedule.show()
    fig, ax = schedule.fig_ax

    assert ax.containers[0].get_children()[0].get_alpha() == 0.5
    assert ax.containers[1].get_children()[0].get_alpha() == 0.5


def test_schedule_show_arg_alpha_list_invalid_length():
    alpha_list = [0.1, 0.1, 0.1]

    schedule = Schedule()
    schedule.add_data("data0", [1, 2], [1, 2], [3, 4])
    schedule.add_data("data1", [1, 2], [1, 2], [3, 4])

    with pytest.raises(ValueError):
        schedule.show(alpha_list=alpha_list)


def test_schedule_show_arg_xlabel():
    xlabel = "xlabel"

    schedule = Schedule()
    schedule.add_data("data", [1, 2], [1, 2], [3, 4])
    schedule.show(xlabel=xlabel)
    fig, ax = schedule.fig_ax

    assert ax.get_xlabel() == "xlabel"


def test_schedule_show_arg_ylabel():
    ylabel = "ylabel"

    schedule = Schedule()
    schedule.add_data("data", [1, 2], [1, 2], [3, 4])
    schedule.show(ylabel=ylabel)
    fig, ax = schedule.fig_ax

    assert ax.get_ylabel() == "ylabel"


def test_schedule_show_arg_xticks():
    xticks = [1, 2, 3, 4, 5, 6]

    schedule = Schedule()
    schedule.add_data("data", [1, 2], [1, 2], [3, 4])
    schedule.show(xticks=xticks)
    fig, ax = schedule.fig_ax

    assert (ax.get_xticks() == np.array([1, 2, 3, 4, 5, 6])).all()


def test_schedule_show_arg_yticks():
    yticks = [1, 2]

    schedule = Schedule()
    schedule.add_data("data", [1, 2], [1, 2], [3, 4])
    schedule.show(yticks=yticks)
    fig, ax = schedule.fig_ax

    assert (ax.get_yticks() == np.array([1, 2])).all()


params = {
    "undirected case": ([[0, 1], [1, 2]], GraphType.UNDIRECTED, nx.Graph),
    "directed case": ([[0, 1], [1, 2]], GraphType.DIRECTED, nx.DiGraph),
}


@pytest.mark.parametrize(
    "edge_list, graphtype, expect_type",
    list(params.values()),
    ids=list(params.keys()),
)
def test_graph_from_edge_list(edge_list, graphtype, expect_type):
    graph = Graph.from_edge_list(edge_list, graphtype)
    G = graph.G

    assert type(G) == expect_type
    assert len(G.edges()) == 2


params = {
    "undirected case": ([[-1, 1], [1, -1]], GraphType.UNDIRECTED, nx.Graph, 1),
    "directed case": ([[-1, 1], [2, -1]], GraphType.DIRECTED, nx.DiGraph, 2),
    "numpy case": (np.array([[-1, 1], [1, -1]]), GraphType.UNDIRECTED, nx.Graph, 1),
}


@pytest.mark.parametrize(
    "distance_matrix, graphtype, expect_type, expect_edge_num",
    list(params.values()),
    ids=list(params.keys()),
)
def test_graph_from_distance_matrix(
    distance_matrix, graphtype, expect_type, expect_edge_num
):
    graph = Graph.from_distance_matrix(distance_matrix, graphtype)
    G = graph.G

    assert type(G) == expect_type
    assert len(G.edges()) == expect_edge_num


def test_graph_fig_ax_attribute():
    graph = Graph.from_edge_list([[1, 2]], GraphType.UNDIRECTED)
    graph.show()
    fig, ax = graph.fig_ax

    assert type(fig) == matplotlib.figure.Figure
    assert type(ax) == matplotlib.axes.Subplot


def test_graph_fig_ax_attribute_before_show():
    graph = Graph.from_edge_list([[1, 2]], GraphType.UNDIRECTED)

    with pytest.raises(AttributeError):
        graph.fig_ax


def test_graph_show_title():
    title = "title"

    graph = Graph.from_edge_list([[1, 2]], GraphType.UNDIRECTED)
    graph.show(title=title)
    fig, ax = graph.fig_ax

    assert fig.texts[0].get_text() == "title"


def test_graph_show_title_default():
    graph = Graph.from_edge_list([[1, 2]], GraphType.UNDIRECTED)
    graph.show()
    fig, ax = graph.fig_ax

    assert fig.texts[0].get_text() == "graph"


def test_graph_show_arg_figsize():
    figwidth, figheight = 8, 4

    graph = Graph.from_edge_list([[1, 2]], GraphType.UNDIRECTED)
    graph.show(figsize=tuple([figwidth, figheight]))
    fig, ax = graph.fig_ax

    assert fig.get_figwidth() == 8
    assert fig.get_figheight() == 4


def test_graph_show_node():
    graph = Graph.from_edge_list([[1, 2], [2, 3]], GraphType.UNDIRECTED)
    graph.show()
    fig, ax = graph.fig_ax

    acutal_node_num = ax.get_children()[0].get_offsets().data.shape[0]
    expect_node_num = 3

    assert acutal_node_num == expect_node_num


def test_graph_show_node_pos():
    pos1 = np.array([1, 1])
    pos2 = np.array([-1, -1])
    node_pos = {1: pos1, 2: pos2}

    graph = Graph.from_edge_list([[1, 2]], GraphType.UNDIRECTED)
    graph.show(node_pos=node_pos)
    fig, ax = graph.fig_ax

    actual_pos = ax.get_children()[0].get_offsets().data
    expect_pos = np.vstack([pos1, pos2])

    assert (actual_pos == expect_pos).all()


def test_graph_show_node_pos_default():
    graph = Graph.from_edge_list([[1, 2]], GraphType.UNDIRECTED)
    graph.show()
    fig, ax = graph.fig_ax

    default_pos = nx.spring_layout(graph.G, seed=1)
    expect_pos = np.vstack(list(default_pos.values()))

    actual_pos = ax.get_children()[0].get_offsets().data

    assert (np.abs(actual_pos - expect_pos) < 0.0001).all()


def test_graph_show_node_color():
    node_color = ["r", "b"]

    graph = Graph.from_edge_list([[1, 2]], GraphType.UNDIRECTED)
    graph.show(node_color=node_color)
    fig, ax = graph.fig_ax

    actual_color_node1, actual_color_node2 = ax.get_children()[0].get_facecolor()
    expect_color_node1 = np.array([1.0, 0.0, 0.0, 1.0])  # color "r"
    expect_color_node2 = np.array([0.0, 0.0, 1.0, 1.0])  # color "g"

    assert (actual_color_node1 == expect_color_node1).all()
    assert (actual_color_node2 == expect_color_node2).all()


def test_graph_show_node_color_default():
    default_color = "#1f78b4"

    graph = Graph.from_edge_list([[1, 2]], GraphType.UNDIRECTED)
    graph.show()
    fig, ax = graph.fig_ax

    actual_color = ax.get_children()[0].get_facecolor()[0][:-1]
    expect_color = np.array(matplotlib.colors.to_rgb(default_color))

    assert (np.abs(actual_color - expect_color) < 0.0001).all()


def test_graph_show_node_labels():
    node_labels = {1: "node1", 2: "node2"}

    graph = Graph.from_edge_list([[1, 2]], GraphType.UNDIRECTED)
    graph.show(node_labels=node_labels)
    fig, ax = graph.fig_ax

    assert ax.texts[0].get_text() == "node1"
    assert ax.texts[1].get_text() == "node2"


def test_graph_show_node_labels_default():
    node1, node2 = 1, 2

    graph = Graph.from_edge_list([[node1, node2]], GraphType.UNDIRECTED)
    graph.show()
    fig, ax = graph.fig_ax

    assert ax.texts[0].get_text() == str(node1)
    assert ax.texts[1].get_text() == str(node2)
