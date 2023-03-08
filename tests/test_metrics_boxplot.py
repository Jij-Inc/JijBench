import os, shutil
import jijmodeling as jm
from matplotlib import axes, figure

import numpy as np
import pandas as pd
import pytest

import jijbench as jb

from jijbench.visualization.metrics.plot import _get_violations_dict, MetricsPlot


@pytest.fixture(scope="function", autouse=True)
def pre_post_process():
    # preprocess
    yield
    # postprocess
    norm_path = os.path.normcase("./.jb_results")
    if os.path.exists(norm_path):
        shutil.rmtree(norm_path)


def solve():
    d = jm.Placeholder("d", dim=2)
    x = jm.Binary("x", shape=d.shape)
    i = jm.Element("i", d.shape[0])
    j = jm.Element("j", d.shape[1])
    problem = jm.Problem("simple_problem")
    problem += jm.Sum([i, j], d[i, j] * x[i, j])
    problem += jm.Constraint("onehot1", x[i, :] == 1, forall=i)
    problem += jm.Constraint("onehot2", x[:, j] == 1, forall=j)
    jm_sampleset_dict = {
        "record": {
            "solution": {
                "x": [
                    (([0, 1], [0, 1]), [1, 1], (2, 2)),
                    (([0, 1], [1, 0]), [1, 1], (2, 2)),
                    (([], []), [], (2, 2)),
                    (([0, 1], [0, 0]), [1, 1], (2, 2)),
                ]
            },
            "num_occurrences": [4, 3, 2, 1],
        },
        "evaluation": {
            "energy": [3.0, 24.0, 0.0, 20.0],
            "objective": [3.0, 24.0, 0.0, 17.0],
            "constraint_violations": {
                "onehot1": [0.0, 0.0, 2.0, 0.0],
                "onehot2": [0.0, 0.0, 2.0, 2.0],
            },
            "penalty": {},
        },
        "measuring_time": {"solve": None, "system": None, "total": None},
    }
    jm_sampleset = jm.SampleSet.from_serializable(jm_sampleset_dict)
    solving_time = jm.SolvingTime(
        **{"preprocess": 1.0, "solve": 1.0, "postprocess": 1.0}
    )
    jm_sampleset.measuring_time.solve = solving_time
    return jm_sampleset


def simple_func_for_boxplot(x: pd.Series) -> dict:
    return {"data1": np.array([1, 2, 3, 4]), "data2": np.array([1, 2, 3, 4])}


def test_metrics_plot_get_violations_dict():
    series = pd.Series(
        data=[np.array([1, 1]), np.array([0, 2])],
        index=["num_occurrences", "onehot_violations"],
    )
    violations_dict = _get_violations_dict(series)

    assert len(violations_dict.keys()) == 1
    assert (violations_dict["onehot_violations"] == np.array([0, 2])).all()


def test_metrics_plot_boxplot_return_value():
    num_multipliers = 2
    params = {"multipliers": [{} for _ in range(num_multipliers)]}
    solver_list = [solve]

    expect_num_fig = len(params["multipliers"]) * len(solver_list)

    bench = jb.Benchmark(
        params=params,
        solver=solver_list,
    )
    result = bench()
    mplot = MetricsPlot(result)
    fig_ax_tuple = mplot.boxplot(f=simple_func_for_boxplot)
    assert len(fig_ax_tuple) == expect_num_fig
    assert type(fig_ax_tuple[0][0]) == figure.Figure
    assert type(fig_ax_tuple[0][1]) == axes.Subplot


def test_metrics_plot_boxplot_call_maplotlib_boxplot(mocker):
    m = mocker.patch("matplotlib.axes.Subplot.boxplot")

    num_multipliers = 2
    params = {"multipliers": [{} for _ in range(num_multipliers)]}
    solver_list = [solve]

    expect_num_call = len(params["multipliers"]) * len(solver_list)

    bench = jb.Benchmark(
        params=params,
        solver=solver_list,
    )
    result = bench()
    mplot = MetricsPlot(result)
    mplot.boxplot(f=simple_func_for_boxplot)

    assert m.call_count == expect_num_call


def test_metrics_plot_boxplot_arg_figsize():
    figwidth, figheight = 8, 4

    bench = jb.Benchmark(
        params={},
        solver=[solve],
    )
    result = bench()
    mplot = MetricsPlot(result)
    fig_ax_tuple = mplot.boxplot(
        f=simple_func_for_boxplot,
        figsize=(figwidth, figheight),
    )
    fig, ax = fig_ax_tuple[0]

    assert fig.get_figwidth() == 8
    assert fig.get_figheight() == 4


def test_metrics_plot_boxplot_arg_title(mocker):
    m = mocker.patch("matplotlib.figure.Figure.suptitle")

    title = ["title1", "title2"]

    num_multipliers = 2
    params = {"multipliers": [{} for _ in range(num_multipliers)]}
    solver_list = [solve]

    bench = jb.Benchmark(
        params=params,
        solver=solver_list,
    )
    result = bench()
    mplot = MetricsPlot(result)
    mplot.boxplot(
        f=simple_func_for_boxplot,
        title=title,
    )
    m.assert_has_calls(
        [
            mocker.call("title1", fontsize=None),
            mocker.call("title2", fontsize=None),
        ]
    )


def test_metrics_plot_boxplot_arg_title_fontsize(mocker):
    m = mocker.patch("matplotlib.figure.Figure.suptitle")

    title = ["title1", "title2"]
    title_fontsize = 12

    num_multipliers = 2
    params = {"multipliers": [{} for _ in range(num_multipliers)]}
    solver_list = [solve]

    bench = jb.Benchmark(
        params=params,
        solver=solver_list,
    )
    result = bench()
    mplot = MetricsPlot(result)
    mplot.boxplot(
        f=simple_func_for_boxplot,
        title=title,
        title_fontsize=title_fontsize,
    )
    m.assert_has_calls(
        [
            mocker.call("title1", fontsize=title_fontsize),
            mocker.call("title2", fontsize=title_fontsize),
        ]
    )


def test_metrics_plot_boxplot_arg_xticklabels():
    bench = jb.Benchmark(
        params={},
        solver=[solve],
    )
    result = bench()
    mplot = MetricsPlot(result)
    fig_ax_tuple = mplot.boxplot(
        f=simple_func_for_boxplot,
    )
    ax = fig_ax_tuple[0][1]
    xticklabels = ax.get_xticklabels()

    assert xticklabels[0].get_text() == "data1"
    assert xticklabels[1].get_text() == "data2"


def test_metrics_plot_boxplot_arg_xticklabels_size(mocker):
    m = mocker.patch("matplotlib.axes.Subplot.set_xticklabels")
    xticklabels_size = 10
    data = {"data1": np.array([1, 2, 3, 4]), "data2": np.array([1, 2, 3, 4])}

    bench = jb.Benchmark(
        params={},
        solver=[solve],
    )
    result = bench()
    mplot = MetricsPlot(result)
    mplot.boxplot(
        f=simple_func_for_boxplot,
        xticklabels_size=xticklabels_size,
    )

    m.assert_called_once_with(data.keys(), size=10, rotation=None)


def test_metrics_plot_boxplot_arg_xticklabels_rotation(mocker):
    m = mocker.patch("matplotlib.axes.Subplot.set_xticklabels")
    xticklabels_rotation = 10
    data = {"data1": np.array([1, 2, 3, 4]), "data2": np.array([1, 2, 3, 4])}

    bench = jb.Benchmark(
        params={},
        solver=[solve],
    )
    result = bench()
    mplot = MetricsPlot(result)
    mplot.boxplot(
        f=simple_func_for_boxplot,
        xticklabels_rotation=xticklabels_rotation,
    )

    m.assert_called_once_with(data.keys(), size=None, rotation=10)


def test_metrics_plot_boxplot_arg_ylabel():
    ylabel = "ylabel"

    bench = jb.Benchmark(
        params={},
        solver=[solve],
    )
    result = bench()
    mplot = MetricsPlot(result)
    fig_ax_tuple = mplot.boxplot(
        f=simple_func_for_boxplot,
        ylabel=ylabel,
    )

    ax = fig_ax_tuple[0][1]
    assert ax.get_ylabel() == "ylabel"


def test_metrics_plot_boxplot_arg_ylabel_size(mocker):
    m = mocker.patch("matplotlib.axes.Subplot.set_ylabel")

    ylabel = "ylabel"
    ylabel_size = 10

    bench = jb.Benchmark(
        params={},
        solver=[solve],
    )
    result = bench()
    mplot = MetricsPlot(result)
    mplot.boxplot(f=simple_func_for_boxplot, ylabel=ylabel, ylabel_size=ylabel_size)

    m.assert_called_once_with("ylabel", size=ylabel_size)


def test_metrics_plot_boxplot_arg_yticks_default(mocker):
    m = mocker.patch("matplotlib.ticker.MaxNLocator.__init__", return_value=None)

    bench = jb.Benchmark(
        params={},
        solver=[solve],
    )
    result = bench()
    mplot = MetricsPlot(result)
    mplot.boxplot(f=simple_func_for_boxplot)

    m.assert_called_with(integer=True)


def test_metrics_plot_boxplot_arg_yticks():
    yticks = [0, 10]

    bench = jb.Benchmark(
        params={},
        solver=[solve],
    )
    result = bench()
    mplot = MetricsPlot(result)
    fig_ax_tuple = mplot.boxplot(
        f=simple_func_for_boxplot,
        yticks=yticks,
    )
    ax = fig_ax_tuple[0][1]
    assert (ax.get_yticks() == [0, 10]).all()


def test_metrics_plot_boxplot_arg_kwargs(mocker):
    m = mocker.patch("matplotlib.axes.Subplot.boxplot")

    bench = jb.Benchmark(
        params={},
        solver=[solve],
    )
    result = bench()
    mplot = MetricsPlot(result)
    mplot.boxplot(
        f=simple_func_for_boxplot,
        showmeans=True,
        meanline=True,
    )

    args, kwargs = m.call_args
    assert kwargs["showmeans"] is True
    assert kwargs["meanline"] is True


def test_metrics_plot_boxplot_violations_return_value():
    num_multipliers = 2
    params = {"multipliers": [{} for _ in range(num_multipliers)]}
    solver_list = [solve]

    expect_num_fig = len(params["multipliers"]) * len(solver_list)

    bench = jb.Benchmark(
        params=params,
        solver=solver_list,
    )
    result = bench()
    mplot = MetricsPlot(result)
    fig_ax_tuple = mplot.boxplot_violations()
    assert len(fig_ax_tuple) == expect_num_fig
    assert type(fig_ax_tuple[0][0]) == figure.Figure
    assert type(fig_ax_tuple[0][1]) == axes.Subplot


def test_metrics_plot_boxplot_violations_call_maplotlib_boxplot(mocker):
    m = mocker.patch("matplotlib.axes.Subplot.boxplot")

    num_multipliers = 2
    params = {"multipliers": [{} for _ in range(num_multipliers)]}
    solver_list = [solve]

    expect_num_call = len(params["multipliers"]) * len(solver_list)

    bench = jb.Benchmark(
        params=params,
        solver=solver_list,
    )
    result = bench()
    mplot = MetricsPlot(result)
    mplot.boxplot_violations()

    assert m.call_count == expect_num_call


def test_metrics_plot_boxplot_violations_arg_figsize():
    figwidth, figheight = 8, 4

    bench = jb.Benchmark(
        params={},
        solver=[solve],
    )
    result = bench()
    mplot = MetricsPlot(result)
    fig_ax_tuple = mplot.boxplot_violations(
        figsize=(figwidth, figheight),
    )
    fig, ax = fig_ax_tuple[0]

    assert fig.get_figwidth() == 8
    assert fig.get_figheight() == 4


def test_metrics_plot_boxplot_violations_arg_title(mocker):
    m = mocker.patch("matplotlib.figure.Figure.suptitle")

    title = ["title1", "title2"]

    num_multipliers = 2
    params = {"multipliers": [{} for _ in range(num_multipliers)]}
    solver_list = [solve]

    bench = jb.Benchmark(
        params=params,
        solver=solver_list,
    )
    result = bench()
    mplot = MetricsPlot(result)
    mplot.boxplot_violations(
        title=title,
    )
    m.assert_has_calls(
        [
            mocker.call("title1", fontsize=None),
            mocker.call("title2", fontsize=None),
        ]
    )


def test_metrics_plot_boxplot_violations_arg_title_fontsize(mocker):
    m = mocker.patch("matplotlib.figure.Figure.suptitle")

    title = ["title1", "title2"]
    title_fontsize = 12

    num_multipliers = 2
    params = {"multipliers": [{} for _ in range(num_multipliers)]}
    solver_list = [solve]

    bench = jb.Benchmark(
        params=params,
        solver=solver_list,
    )
    result = bench()
    mplot = MetricsPlot(result)
    mplot.boxplot_violations(
        title=title,
        title_fontsize=title_fontsize,
    )
    m.assert_has_calls(
        [
            mocker.call("title1", fontsize=title_fontsize),
            mocker.call("title2", fontsize=title_fontsize),
        ]
    )


def test_metrics_plot_boxplot_violations_arg_xticklabels():
    bench = jb.Benchmark(
        params={},
        solver=[solve],
    )
    result = bench()
    mplot = MetricsPlot(result)
    fig_ax_tuple = mplot.boxplot_violations()
    ax = fig_ax_tuple[0][1]
    xticklabels = ax.get_xticklabels()

    assert xticklabels[0].get_text() == "onehot1_violations"
    assert xticklabels[1].get_text() == "onehot2_violations"


def test_metrics_plot_boxplot_violations_arg_xticklabels_size(mocker):
    m = mocker.patch("matplotlib.axes.Subplot.set_xticklabels")
    constraint_name_fontsize = 10

    bench = jb.Benchmark(
        params={},
        solver=[solve],
    )
    result = bench()
    mplot = MetricsPlot(result)
    mplot.boxplot_violations(
        constraint_name_fontsize=constraint_name_fontsize,
    )
    args, kwargs = m.call_args
    assert kwargs["size"] == 10


def test_metrics_plot_boxplot_violations_arg_xticklabels_rotation(mocker):
    m = mocker.patch("matplotlib.axes.Subplot.set_xticklabels")
    constraint_name_fontrotation = 10

    bench = jb.Benchmark(
        params={},
        solver=[solve],
    )
    result = bench()
    mplot = MetricsPlot(result)
    mplot.boxplot_violations(
        constraint_name_fontrotation=constraint_name_fontrotation,
    )
    args, kwargs = m.call_args
    assert kwargs["rotation"] == 10


def test_metrics_plot_boxplot_violations_arg_ylabel_default():
    bench = jb.Benchmark(
        params={},
        solver=[solve],
    )
    result = bench()
    mplot = MetricsPlot(result)
    fig_ax_tuple = mplot.boxplot_violations()

    ax = fig_ax_tuple[0][1]
    assert ax.get_ylabel() == "constraint violations"


def test_metrics_plot_boxplot_violations_arg_ylabel():
    ylabel = "ylabel"

    bench = jb.Benchmark(
        params={},
        solver=[solve],
    )
    result = bench()
    mplot = MetricsPlot(result)
    fig_ax_tuple = mplot.boxplot_violations(
        ylabel=ylabel,
    )

    ax = fig_ax_tuple[0][1]
    assert ax.get_ylabel() == "ylabel"


def test_metrics_plot_boxplot_violations_arg_ylabel_size(mocker):
    m = mocker.patch("matplotlib.axes.Subplot.set_ylabel")

    ylabel = "ylabel"
    ylabel_size = 10

    bench = jb.Benchmark(
        params={},
        solver=[solve],
    )
    result = bench()
    mplot = MetricsPlot(result)
    mplot.boxplot_violations(ylabel=ylabel, ylabel_size=ylabel_size)

    m.assert_called_once_with("ylabel", size=ylabel_size)


def test_metrics_plot_boxplot_violations_arg_yticks_default(mocker):
    m = mocker.patch("matplotlib.ticker.MaxNLocator.__init__", return_value=None)
    mocker.patch("matplotlib.axes.Subplot.axhline")

    bench = jb.Benchmark(
        params={},
        solver=[solve],
    )
    result = bench()
    mplot = MetricsPlot(result)
    mplot.boxplot_violations()

    m.assert_called_with(integer=True)


def test_metrics_plot_boxplot_violations_arg_yticks():
    yticks = [0, 10]

    bench = jb.Benchmark(
        params={},
        solver=[solve],
    )
    result = bench()
    mplot = MetricsPlot(result)
    fig_ax_tuple = mplot.boxplot_violations(
        yticks=yticks,
    )
    ax = fig_ax_tuple[0][1]
    assert (ax.get_yticks() == [0, 10]).all()


def test_metrics_plot_boxplot_violations_arg_kwargs_default(mocker):
    m = mocker.patch("matplotlib.axes.Subplot.boxplot")

    bench = jb.Benchmark(
        params={},
        solver=[solve],
    )
    result = bench()
    mplot = MetricsPlot(result)
    mplot.boxplot_violations()

    args, kwargs = m.call_args
    assert kwargs["showmeans"] is True
    assert kwargs["whis"] == [0, 100]


def test_metrics_plot_boxplot_violations_arg_kwargs(mocker):
    m = mocker.patch("matplotlib.axes.Subplot.boxplot")

    bench = jb.Benchmark(
        params={},
        solver=[solve],
    )
    result = bench()
    mplot = MetricsPlot(result)
    mplot.boxplot_violations(
        showmeans=True,
        meanline=True,
    )

    args, kwargs = m.call_args
    assert kwargs["showmeans"] is True
    assert kwargs["meanline"] is True


def test_metrics_plot_boxplot_violations_hline_indicate_no_violation(mocker):
    m = mocker.patch("matplotlib.axes.Subplot.axhline")

    num_multipliers = 2
    params = {"multipliers": [{} for _ in range(num_multipliers)]}
    solver_list = [solve]

    expect_num_call = len(params["multipliers"]) * len(solver_list)

    bench = jb.Benchmark(
        params=params,
        solver=solver_list,
    )
    result = bench()
    mplot = MetricsPlot(result)
    mplot.boxplot_violations()

    m.assert_has_calls(
        [
            mocker.call(0, xmin=0, xmax=1, color="gray", linestyle="dotted"),
            mocker.call(0, xmin=0, xmax=1, color="gray", linestyle="dotted"),
        ]
    )
