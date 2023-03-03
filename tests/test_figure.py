from collections import OrderedDict
import matplotlib
from matplotlib import axes, collections, figure, patches
import networkx as nx
import numpy as np
import pytest

from jijbench.visualization.figure.graph import Graph, GraphType
from jijbench.visualization.figure.schedule import Schedule
from jijbench.visualization.figure.timeseries import TimeSeries

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

    assert type(fig) == figure.Figure
    assert type(ax) == axes.Subplot


def test_timeseries_fig_ax_attribute_before_show():
    timeseries = TimeSeries()
    timeseries.add_data("data", [1, 2], [3, 4])

    with pytest.raises(AttributeError):
        timeseries.fig_ax


def test_timeseries_show_no_plot_data():
    timeseries = TimeSeries()
    with pytest.raises(RuntimeError):
        timeseries.show()


params = {
    "give argument": ("title", "title"),
    "default": (None, "time series"),
}


@pytest.mark.parametrize(
    "title, expect",
    list(params.values()),
    ids=list(params.keys()),
)
def test_timeseries_show_arg_title(title, expect):
    timeseries = TimeSeries()
    timeseries.add_data("data", [1, 2], [3, 4])
    timeseries.show(title=title)
    fig, ax = timeseries.fig_ax

    success_show_title = False
    for obj in fig.texts:
        actual_title = obj.get_text()
        if actual_title == expect:
            success_show_title = True

    assert success_show_title


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


params = {
    "give argument": ([0.5, 0.7], 0.5, 0.7),
    "default": (None, 1.0, 1.0),
}


@pytest.mark.parametrize(
    "alpha_list, expect_alpha_1, expect_alpha_2",
    list(params.values()),
    ids=list(params.keys()),
)
def test_timeseries_show_arg_alpha_list(alpha_list, expect_alpha_1, expect_alpha_2):
    timeseries = TimeSeries()
    timeseries.add_data("data0", [1, 2], [3, 4])
    timeseries.add_data("data1", [1, 2], [3, 4])
    timeseries.show(alpha_list=alpha_list)
    fig, ax = timeseries.fig_ax

    assert ax.get_lines()[0].get_alpha() == expect_alpha_1
    assert ax.get_lines()[1].get_alpha() == expect_alpha_2


def test_timeseries_show_arg_alpha_list_invalid_length():
    alpha_list = [0.1, 0.1, 0.1]

    timeseries = TimeSeries()
    timeseries.add_data("data0", [1, 2], [3, 4])
    timeseries.add_data("data1", [1, 2], [3, 4])

    with pytest.raises(ValueError):
        timeseries.show(alpha_list=alpha_list)


params = {
    "give argument": (["-", "--"], "-", "--"),
    "default": (None, "-", "-"),
}


@pytest.mark.parametrize(
    "linestyle_list, expect_linestyle_1, expect_linestyle_2",
    list(params.values()),
    ids=list(params.keys()),
)
def test_timeseries_show_arg_linestyle_list(
    linestyle_list, expect_linestyle_1, expect_linestyle_2
):
    timeseries = TimeSeries()
    timeseries.add_data("data0", [1, 2], [3, 4])
    timeseries.add_data("data1", [1, 2], [3, 4])
    timeseries.show(linestyle_list=linestyle_list)
    fig, ax = timeseries.fig_ax

    assert ax.get_lines()[0].get_linestyle() == expect_linestyle_1
    assert ax.get_lines()[1].get_linestyle() == expect_linestyle_2


def test_timeseries_show_arg_linestyle_list_invalid_length():
    linestyle_list = ["--", "--", "--"]

    timeseries = TimeSeries()
    timeseries.add_data("data0", [1, 2], [3, 4])
    timeseries.add_data("data1", [1, 2], [3, 4])

    with pytest.raises(ValueError):
        timeseries.show(linestyle_list=linestyle_list)


params = {
    "give argument": (["v", "d"], "v", "d"),
    "default": (None, "o", "o"),
}


@pytest.mark.parametrize(
    "marker_list, expect_marker_1, expect_marker_2",
    list(params.values()),
    ids=list(params.keys()),
)
def test_timeseries_show_arg_marker_list(marker_list, expect_marker_1, expect_marker_2):
    timeseries = TimeSeries()
    timeseries.add_data("data0", [1, 2], [3, 4])
    timeseries.add_data("data1", [1, 2], [3, 4])
    timeseries.show(marker_list=marker_list)
    fig, ax = timeseries.fig_ax

    assert ax.get_lines()[0].get_marker() == expect_marker_1
    assert ax.get_lines()[1].get_marker() == expect_marker_2


def test_timeseries_show_arg_marker_list_invalid_length():
    marker_list = ["v", "v", "v"]

    timeseries = TimeSeries()
    timeseries.add_data("data0", [1, 2], [3, 4])
    timeseries.add_data("data1", [1, 2], [3, 4])

    with pytest.raises(ValueError):
        timeseries.show(marker_list=marker_list)


params = {
    "give argument": ("xlabel", "xlabel"),
    "default": (None, "time"),
}


@pytest.mark.parametrize(
    "xlabel, expect",
    list(params.values()),
    ids=list(params.keys()),
)
def test_timeseries_show_arg_xlabel(xlabel, expect):
    timeseries = TimeSeries()
    timeseries.add_data("data", [1, 2], [3, 4])
    timeseries.show(xlabel=xlabel)
    fig, ax = timeseries.fig_ax

    assert ax.get_xlabel() == expect


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


def test_schedule_add_data_attribute_workers():
    schedule = Schedule()
    schedule.add_data("data1", [1, 2], [3, 4], [2, 4])
    schedule.add_data("data2", [2, 3], [7, 8], [9, 10])

    assert schedule.workers == [1, 2, 3]


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

    assert type(fig) == figure.Figure
    assert type(ax) == axes.Subplot


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


params = {
    "give argument": ("title", "title"),
    "default": (None, "schedule"),
}


@pytest.mark.parametrize(
    "title, expect",
    list(params.values()),
    ids=list(params.keys()),
)
def test_schedule_show_arg_title(title, expect):
    schedule = Schedule()
    schedule.add_data("data", [1, 2], [3, 4], [5, 6])
    schedule.show(title=title)
    fig, ax = schedule.fig_ax

    success_show_title = False
    for obj in fig.texts:
        actual_title = obj.get_text()
        if actual_title == expect:
            success_show_title = True

    assert success_show_title


def test_schedule_show_bar():
    workers1, start_times1, time_lengths1 = [1, 2], [1, 2], [3, 4]
    workers2, start_times2, time_lengths2 = [2], [1], [1]

    schedule = Schedule()
    schedule.add_data("data1", workers1, start_times1, time_lengths1)
    schedule.add_data("data2", workers2, start_times2, time_lengths2)

    schedule.show()
    fig, ax = schedule.fig_ax

    # Check that the children of ax contain the expected bar information
    expect_center_1 = np.array(
        [2.5, 1]
    )  # np.array([start_times1[0] + (time_length1[0] / 2), workers1[0]])
    success_show_bar_1 = False
    for obj in ax.get_children():
        if type(obj) == patches.Rectangle:
            actual_center_1 = obj.get_center()
            if (np.abs(actual_center_1 - expect_center_1) < 0.0001).all():
                success_show_bar_1 = True
    assert success_show_bar_1

    expect_center_2 = np.array(
        [4, 2]
    )  # np.array([start_times1[1] + (time_length1[1] / 2), workers1[1]])
    success_show_bar_2 = False
    for obj in ax.get_children():
        if type(obj) == patches.Rectangle:
            actual_center_2 = obj.get_center()
            if (np.abs(actual_center_2 - expect_center_2) < 0.0001).all():
                success_show_bar_2 = True
    assert success_show_bar_2

    expect_center_3 = np.array(
        [1.5, 2]
    )  # np.array([start_times2[0] + (time_length2[0] / 2), workers2[0]])
    success_show_bar_3 = False
    for obj in ax.get_children():
        if type(obj) == patches.Rectangle:
            actual_center_3 = obj.get_center()
            if (np.abs(actual_center_3 - expect_center_3) < 0.0001).all():
                success_show_bar_3 = True
    assert success_show_bar_3


def test_schedule_show_text():
    workers, start_times, time_lengths = [1, 2], [1, 2], [3, 4]

    schedule = Schedule()
    schedule.add_data("data", workers, start_times, time_lengths)
    schedule.show()
    fig, ax = schedule.fig_ax

    # Check that the ax.texts contain the expected text information
    expect_text_1 = "3"  # time_length[0]
    expect_center_1 = (2.5, 1)  # (start_times[0] + (time_length[0] / 2), workers[0])
    success_text_1 = False
    for obj in ax.texts:
        actual_text_1 = obj.get_text()
        actual_center_1 = obj.get_position()
        if not (actual_text_1 == expect_text_1):
            continue
        if not (actual_center_1 == expect_center_1):
            continue
        success_text_1 = True
    assert success_text_1

    expect_text_2 = "4"  # time_length[1]
    expect_center_2 = (4, 2)  # (start_times[1] + (time_length[1] / 2), workers[1])
    success_text_2 = False
    for obj in ax.texts:
        actual_text_2 = obj.get_text()
        actual_center_2 = obj.get_position()
        if not (actual_text_2 == expect_text_2):
            continue
        if not (actual_center_2 == expect_center_2):
            continue
        success_text_2 = True
    assert success_text_2


def test_schedule_show_arg_figsize():
    figwidth, figheight = 8, 4

    schedule = Schedule()
    schedule.add_data("data", [1, 2], [1, 2], [3, 4])
    schedule.show(figsize=tuple([figwidth, figheight]))
    fig, ax = schedule.fig_ax

    assert fig.get_figwidth() == 8
    assert fig.get_figheight() == 4


def test_schedule_show_arg_color_list():
    color_list = ["red", "blue"]
    workers1, start_times1, time_lengths1 = [1], [1], [3]
    workers2, start_times2, time_lengths2 = [2], [2], [4]

    schedule = Schedule()
    schedule.add_data("data1", workers1, start_times1, time_lengths1)
    schedule.add_data("data2", workers2, start_times2, time_lengths2)
    schedule.show(color_list=color_list)
    fig, ax = schedule.fig_ax

    # Check that the children of ax contain the expected color information
    expect_color_1 = np.array([1.0, 0.0, 0.0])  # red
    expect_center_1 = np.array(
        [2.5, 1]
    )  # np.array([start_times1[0] + (time_length1[0] / 2), workers1[0]])
    success_coloring_1 = False
    for obj in ax.get_children():
        if type(obj) == patches.Rectangle:
            actual_color_1 = obj.get_facecolor()[:-1]
            actual_center_1 = obj.get_center()
            if not (actual_color_1 == expect_color_1).all():
                continue
            if not (np.abs(actual_center_1 - expect_center_1) < 0.0001).all():
                continue
            success_coloring_1 = True
    assert success_coloring_1

    expect_color_2 = np.array([0.0, 0.0, 1.0])  # blue
    expect_center_2 = np.array(
        [4, 2]
    )  # np.array([start_times2[0] + (time_length2[0] / 2), workers2[0]])
    success_coloring_2 = False
    for obj in ax.get_children():
        if type(obj) == patches.Rectangle:
            actual_color_2 = obj.get_facecolor()[:-1]
            actual_center_2 = obj.get_center()
            if not (actual_color_2 == expect_color_2).all():
                continue
            if not (np.abs(actual_center_2 - expect_center_2) < 0.0001).all():
                continue
            success_coloring_2 = True
    assert success_coloring_2


def test_schedule_show_arg_color_list_invalid_length():
    color_list = ["r", "g", "b"]

    schedule = Schedule()
    schedule.add_data("data0", [1, 2], [1, 2], [3, 4])
    schedule.add_data("data1", [1, 2], [1, 2], [3, 4])

    with pytest.raises(ValueError):
        schedule.show(color_list=color_list)


params = {
    "give argument": ([0.3, 0.7], 0.3, 0.7),
    "default": (None, 0.5, 0.5),
}


@pytest.mark.parametrize(
    "alpha_list, expect_alpha_1, expect_alpha_2",
    list(params.values()),
    ids=list(params.keys()),
)
def test_schedule_show_arg_alpha_list(alpha_list, expect_alpha_1, expect_alpha_2):
    workers1, start_times1, time_lengths1 = [1], [1], [3]
    workers2, start_times2, time_lengths2 = [2], [2], [4]

    schedule = Schedule()
    schedule.add_data("data1", workers1, start_times1, time_lengths1)
    schedule.add_data("data2", workers2, start_times2, time_lengths2)
    schedule.show(alpha_list=alpha_list)
    fig, ax = schedule.fig_ax

    # Check that the children of ax contain the expected alpha information
    expect_center_1 = np.array(
        [2.5, 1]
    )  # np.array([start_times1[0] + (time_length1[0] / 2), workers1[0]])
    success_alpha_1 = False
    for obj in ax.get_children():
        if type(obj) == patches.Rectangle:
            actual_alpha_1 = obj.get_alpha()
            actual_center_1 = obj.get_center()
            if not actual_alpha_1 == expect_alpha_1:
                continue
            if not (np.abs(actual_center_1 - expect_center_1) < 0.0001).all():
                continue
            success_alpha_1 = True
    assert success_alpha_1

    expect_center_2 = np.array(
        [4, 2]
    )  # np.array([start_times1[0] + (time_length1[0] / 2), workers1[0]])
    success_alpha_2 = False
    for obj in ax.get_children():
        if type(obj) == patches.Rectangle:
            actual_alpha_2 = obj.get_alpha()
            actual_center_2 = obj.get_center()
            if not actual_alpha_2 == expect_alpha_2:
                continue
            if not (np.abs(actual_center_2 - expect_center_2) < 0.0001).all():
                continue
            success_alpha_2 = True
    assert success_alpha_2


def test_schedule_show_arg_alpha_list_invalid_length():
    alpha_list = [0.1, 0.1, 0.1]

    schedule = Schedule()
    schedule.add_data("data0", [1, 2], [1, 2], [3, 4])
    schedule.add_data("data1", [1, 2], [1, 2], [3, 4])

    with pytest.raises(ValueError):
        schedule.show(alpha_list=alpha_list)


params = {
    "give argument": ("xlabel", "xlabel"),
    "default": (None, "time"),
}


@pytest.mark.parametrize(
    "xlabel, expect",
    list(params.values()),
    ids=list(params.keys()),
)
def test_schedule_show_arg_xlabel(xlabel, expect):
    schedule = Schedule()
    schedule.add_data("data", [1, 2], [1, 2], [3, 4])
    schedule.show(xlabel=xlabel)
    fig, ax = schedule.fig_ax

    assert ax.get_xlabel() == expect


params = {
    "give argument": ("ylabel", "ylabel"),
    "default": (None, "worker"),
}


@pytest.mark.parametrize(
    "ylabel, expect",
    list(params.values()),
    ids=list(params.keys()),
)
def test_schedule_show_arg_ylabel(ylabel, expect):
    schedule = Schedule()
    schedule.add_data("data", [1, 2], [1, 2], [3, 4])
    schedule.show(ylabel=ylabel)
    fig, ax = schedule.fig_ax

    assert ax.get_ylabel() == expect


def test_schedule_show_arg_xticks():
    xticks = [1, 2, 3, 4, 5, 6]

    schedule = Schedule()
    schedule.add_data("data", [1, 2], [1, 2], [3, 4])
    schedule.show(xticks=xticks)
    fig, ax = schedule.fig_ax

    assert (ax.get_xticks() == np.array([1, 2, 3, 4, 5, 6])).all()


params = {
    "give argument": ([1, 2, 3], np.array([1, 2, 3])),
    "default": (None, np.array([1, 2])),
}


@pytest.mark.parametrize(
    "yticks, expect",
    list(params.values()),
    ids=list(params.keys()),
)
def test_schedule_show_arg_yticks(yticks, expect):
    schedule = Schedule()
    schedule.add_data("data", [1, 2], [1, 2], [3, 4])
    schedule.show(yticks=yticks)
    fig, ax = schedule.fig_ax

    assert (ax.get_yticks() == expect).all()


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

    assert type(fig) == figure.Figure
    assert type(ax) == axes.Subplot


def test_graph_fig_ax_attribute_before_show():
    graph = Graph.from_edge_list([[1, 2]], GraphType.UNDIRECTED)

    with pytest.raises(AttributeError):
        graph.fig_ax


params = {
    "give argument": ("title", "title"),
    "default": (None, "graph"),
}


@pytest.mark.parametrize(
    "title, expect",
    list(params.values()),
    ids=list(params.keys()),
)
def test_graph_show_arg_title(title, expect):
    graph = Graph.from_edge_list([[1, 2]], GraphType.UNDIRECTED)
    graph.show(title=title)
    fig, ax = graph.fig_ax

    # Check that the ax.texts contain the expected node label information
    success_show_title = False
    for obj in fig.texts:
        actual_title = obj.get_text()
        if actual_title == expect:
            success_show_title = True

    assert success_show_title


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

    # Check that the children of ax contain the expected node information
    expect_node_num = 3
    success_show_node = False
    for obj in ax.get_children():
        if type(obj) == collections.PathCollection:
            actual_node_num = obj.get_offsets().data.shape[0]
            if (np.abs(actual_node_num - expect_node_num) < 0.0001).all():
                success_show_node = True
    assert success_show_node


def test_graph_show_arg_node_pos():
    pos1 = np.array([1, 1])
    pos2 = np.array([-1, -1])
    node_pos = {1: pos1, 2: pos2}

    # Check that the children of ax contain the expected node position information
    graph = Graph.from_edge_list([[1, 2]], GraphType.UNDIRECTED)
    graph.show(node_pos=node_pos)
    fig, ax = graph.fig_ax

    expect_pos = np.vstack([pos1, pos2])
    success_set_pos = False
    for obj in ax.get_children():
        if type(obj) == collections.PathCollection:
            actual_pos = obj.get_offsets().data
            if (np.abs(actual_pos - expect_pos) < 0.0001).all():
                success_set_pos = True
    assert success_set_pos


def test_graph_show_arg_node_pos_default():
    graph = Graph.from_edge_list([[1, 2]], GraphType.UNDIRECTED)
    graph.show()
    fig, ax = graph.fig_ax

    # Check that the children of ax contain the expected node position information
    default_pos = nx.spring_layout(graph.G, seed=1)
    expect_pos = np.vstack(list(default_pos.values()))
    success_set_pos = False
    for obj in ax.get_children():
        if type(obj) == collections.PathCollection:
            actual_pos = obj.get_offsets().data
            if (np.abs(actual_pos - expect_pos) < 0.0001).all():
                success_set_pos = True
    assert success_set_pos


def test_graph_show_arg_node_color():
    node_color = ["r", "b"]

    graph = Graph.from_edge_list([[1, 2]], GraphType.UNDIRECTED)
    graph.show(node_color=node_color)
    fig, ax = graph.fig_ax

    # Check that the children of ax contain the expected node color information
    expect_color_node1 = np.array(matplotlib.colors.to_rgb("r"))
    success_coloring_node1 = False
    for obj in ax.get_children():
        if type(obj) == collections.PathCollection:
            actual_color_node1 = obj.get_facecolor()[0][:-1]
            if (np.abs(actual_color_node1 - expect_color_node1) < 0.0001).all():
                success_coloring_node1 = True
    assert success_coloring_node1

    expect_color_node2 = np.array(matplotlib.colors.to_rgb("b"))
    success_coloring_node2 = False
    for obj in ax.get_children():
        if type(obj) == collections.PathCollection:
            actual_color_node2 = obj.get_facecolor()[1][:-1]
            if (np.abs(actual_color_node2 - expect_color_node2) < 0.0001).all():
                success_coloring_node2 = True
    assert success_coloring_node2


def test_graph_show_arg_node_color_default():
    default_color = "#1f78b4"

    graph = Graph.from_edge_list([[1, 2]], GraphType.UNDIRECTED)
    graph.show()
    fig, ax = graph.fig_ax

    expect_color = np.array(matplotlib.colors.to_rgb(default_color))

    # Check that the children of ax contain the expected node color information
    success_coloring = False
    for obj in ax.get_children():
        if type(obj) == collections.PathCollection:
            actual_color = obj.get_facecolor()[0][:-1]
            if (np.abs(actual_color - expect_color) < 0.0001).all():
                success_coloring = True
    assert success_coloring


def test_graph_show_arg_node_labels():
    node1, node2 = 1, 2
    node_labels = {node1: "node1", node2: "node2"}
    node_pos = {node1: np.array([1, 1]), node2: np.array([-1, -1])}

    graph = Graph.from_edge_list([[1, 2]], GraphType.UNDIRECTED)
    graph.show(node_pos=node_pos, node_labels=node_labels)
    fig, ax = graph.fig_ax

    # Check that the ax.texts contain the expected node label information
    expect_info_node1 = (node_pos[node1][0], node_pos[node1][1], "node1")
    success_show_node1_label = False
    for obj in ax.texts:
        actual_info_node1 = (*obj.get_position(), obj.get_text())
        if actual_info_node1 == expect_info_node1:
            success_show_node1_label = True
    assert success_show_node1_label

    expect_info_node2 = (node_pos[node2][0], node_pos[node2][1], "node2")
    success_show_node2_label = False
    for obj in ax.texts:
        actual_info_node2 = (*obj.get_position(), obj.get_text())
        if actual_info_node2 == expect_info_node2:
            success_show_node2_label = True
    assert success_show_node2_label


def test_graph_show_arg_node_labels_default():
    node1, node2 = 1, 2
    node_pos = {node1: np.array([1, 1]), node2: np.array([-1, -1])}

    graph = Graph.from_edge_list([[node1, node2]], GraphType.UNDIRECTED)
    graph.show(node_pos=node_pos)
    fig, ax = graph.fig_ax

    # Check that the ax.texts contain the expected node label information
    expect_info_node1 = (node_pos[node1][0], node_pos[node1][1], str(node1))
    success_show_node1_label = False
    for obj in ax.texts:
        actual_info_node1 = (*obj.get_position(), obj.get_text())
        if actual_info_node1 == expect_info_node1:
            success_show_node1_label = True
    assert success_show_node1_label

    expect_info_node2 = (node_pos[node2][0], node_pos[node2][1], str(node2))
    success_show_node2_label = False
    for obj in ax.texts:
        actual_info_node2 = (*obj.get_position(), obj.get_text())
        if actual_info_node2 == expect_info_node2:
            success_show_node2_label = True
    assert success_show_node2_label


def test_graph_show_arg_edge():
    edge_list = [[1, 2], [2, 3]]

    graph = Graph.from_edge_list(edge_list, GraphType.UNDIRECTED)
    graph.show()
    fig, ax = graph.fig_ax

    expect_edge_num = 2

    # Check that the children of ax contain the expected edge information
    success_show_edge = False
    for obj in ax.get_children():
        if type(obj) == collections.LineCollection:
            actual_edge_num = len(obj.get_paths())
            if actual_edge_num == expect_edge_num:
                success_show_edge = True
    assert success_show_edge


def test_graph_show_arg_edge_color():
    edge_color = ["r"]

    graph = Graph.from_edge_list([[1, 2]], GraphType.UNDIRECTED)
    graph.show(edge_color=edge_color)
    fig, ax = graph.fig_ax

    expect_color = np.array([1.0, 0.0, 0.0])

    # Check that the children of ax contain the expected edge color information
    success_coloring = False
    for obj in ax.get_children():
        if type(obj) == collections.LineCollection:
            actual_color = obj.get_color()[0][:-1]
            if (actual_color == expect_color).all():
                success_coloring = True
    assert success_coloring


def test_graph_show_arg_edge_color_default():
    graph = Graph.from_edge_list([[1, 2]], GraphType.UNDIRECTED)
    graph.show()
    fig, ax = graph.fig_ax

    expect_color = np.array([0.0, 0.0, 0.0])  # "black"

    # Check that the children of ax contain the expected edge color information
    success_coloring = False
    for obj in ax.get_children():
        if type(obj) == collections.LineCollection:
            actual_color = obj.get_color()[0][:-1]
            if (actual_color == expect_color).all():
                success_coloring = True
    assert success_coloring


def test_graph_show_arg_edge_labels():
    node_pos = {
        0: np.array([-1, -1]),
        1: np.array([1, 1]),
    }
    edge_labels = {(0, 1): "edge"}

    graph = Graph.from_edge_list([[0, 1]], GraphType.UNDIRECTED)
    graph.show(node_pos=node_pos, edge_labels=edge_labels)
    fig, ax = graph.fig_ax

    # Check that the ax.texts contain the expected edge label information
    expect_label_x, expect_label_y = (node_pos[0] + node_pos[1]) / 2
    expect_label = edge_labels[(0, 1)]
    expect_info = (expect_label_x, expect_label_y, expect_label)

    success_show_edge_label = False
    for obj in ax.texts:
        actual_info = (*obj.get_position(), obj.get_text())
        if actual_info == expect_info:
            success_show_edge_label = True

    assert success_show_edge_label


def test_graph_show_arg_edge_labels_default_weighted_edge_case():
    node_pos = {0: np.array([1, 1]), 1: np.array([-1, -1])}
    weight = 5

    graph = Graph.from_distance_matrix(
        [[-1, weight], [weight, -1]], GraphType.UNDIRECTED
    )
    graph.show(node_pos=node_pos)
    fig, ax = graph.fig_ax

    # Check that the ax.texts contain the expected edge label information
    expect_label_x, expect_label_y = (node_pos[0] + node_pos[1]) / 2
    expect_label = str(weight)
    expect_info = (expect_label_x, expect_label_y, expect_label)

    success_show_edge_label = False
    for obj in ax.texts:
        actual_info = (*obj.get_position(), obj.get_text())
        if actual_info == expect_info:
            success_show_edge_label = True

    assert success_show_edge_label
