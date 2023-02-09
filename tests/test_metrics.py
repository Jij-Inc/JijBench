import jijmodeling as jm
import matplotlib
import numpy as np
import pandas as pd
import pytest

import jijbench as jb

from jijbench.visualize.metrics.plot import get_violations_dict, MetricsPlot
from jijbench.visualize.metrics.utils import (
    construct_experiment_from_sampleset,
    create_fig_title_list,
)


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


def test_utils_construct_experiment_from_sampleset():
    sampleset = jm.SampleSet(
        record=jm.Record(
            solution={"x": [(([3],), [1], (6,))]},
            num_occurrences=[4],
        ),
        evaluation=jm.Evaluation(
            energy=[-8.0],
            objective=[-8.0],
            constraint_violations={"onehot": [0.0]},
            penalty={},
        ),
        measuring_time=jm.MeasuringTime(),
    )

    experiment = construct_experiment_from_sampleset(sampleset)

    assert isinstance(experiment, jb.Experiment)
    assert experiment.table["num_occurrences"].values == [np.array([4])]


params = {
    "input is title list": (["title1", "title2"], ["title1", "title2"]),
    "input is title string": ("title", ["title", "title"]),
    "input is None": (None, ["i: 1", "i: 2"]),
}


@pytest.mark.parametrize(
    "input_title, expect",
    list(params.values()),
    ids=list(params.keys()),
)
def test_utils_create_fig_title_list(input_title, expect):
    series = pd.Series(
        data=[1, 2],
        index=["1", "2"],
    )
    series.index.names = ["i"]

    title_list = create_fig_title_list(
        metrics=series,
        title=input_title,
    )
    assert title_list == expect


def test_utils_create_fig_title_list_for_series_with_no_index():
    series = pd.Series(
        data=[1, 2],
        index=[None, None],
    )
    title_list = create_fig_title_list(
        metrics=series,
        title=None,
    )
    assert title_list == ["", ""]


def test_utils_create_fig_title_list_for_invalid_input():
    invalid_input_title = 0

    series = pd.Series(
        data=[1, 2],
        index=["1", "2"],
    )
    series.index.names = ["i"]
    with pytest.raises(TypeError):
        create_fig_title_list(
            metrics=series,
            title=invalid_input_title,
        )


def test_metrics_plot_get_violations_dict():
    series = pd.Series(
        data=[np.array([1, 1]), np.array([0, 2])],
        index=["num_occurrences", "onehot_violations"],
    )
    violations_dict = get_violations_dict(series)

    assert len(violations_dict.keys()) == 1
    assert (violations_dict["onehot_violations"] == np.array([0, 2])).all()


def test_metrics_plot_boxplot():
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
    assert type(fig_ax_tuple[0][0]) == matplotlib.figure.Figure
    assert type(fig_ax_tuple[0][1]) == matplotlib.axes.Subplot


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
    m.assert_any_call("title1")
    m.assert_any_call("title2")
