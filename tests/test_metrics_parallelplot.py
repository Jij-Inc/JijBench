import jijmodeling as jm

import numpy as np
import pandas as pd
import pytest

import jijbench as jb

from jijbench.visualize.metrics.plot import MetricsPlot


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


def solve_no_constraint():
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
            "constraint_violations": {},
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


def fig_contain_target_data(fig, label, values):
    for data in fig.data[0].dimensions:
        if data["label"] == label and (data["values"] == values).all():
            return True
    return False


def test_metrics_plot_parallelplot_samplemean_objective():
    bench = jb.Benchmark(
        params={"multipliers": [{}, {}]},
        solver=[solve],
    )
    result = bench()
    mplot = MetricsPlot(result)
    fig = mplot.parallelplot_experiment()

    def calc_result_samplemean_objective(result, idx):
        obj = result.table["objective"].values[idx]
        occ = result.table["num_occurrences"].values[idx]
        print(obj, occ)
        return np.sum(occ * obj) / np.sum(occ)

    expect_mean_0 = calc_result_samplemean_objective(result, 0)
    expect_mean_1 = calc_result_samplemean_objective(result, 1)
    expect_arr = np.array(expect_mean_0, expect_mean_1)

    assert fig_contain_target_data(fig, "samplemean_objective", expect_arr)


def test_metrics_plot_parallelplot_samplemean_violations():
    bench = jb.Benchmark(
        params={},
        solver=[solve],
    )
    result = bench()
    mplot = MetricsPlot(result)
    fig = mplot.parallelplot_experiment()

    def calc_result_samplemean_violations(result, idx, constraint_name):
        violation = result.table[f"{constraint_name}_violations"].values[idx]
        occ = result.table["num_occurrences"].values[idx]
        return np.sum(occ * violation) / np.sum(occ)

    expect_mean_onehot1 = calc_result_samplemean_violations(result, 0, "onehot1")
    expect_mean_onehot2 = calc_result_samplemean_violations(result, 0, "onehot2")
    expect_mean_total = expect_mean_onehot1 + expect_mean_onehot2

    assert fig_contain_target_data(
        fig, "samplemean_onehot1_violations", np.array([expect_mean_onehot1])
    )

    assert fig_contain_target_data(
        fig, "samplemean_onehot2_violations", np.array([expect_mean_onehot2])
    )

    assert fig_contain_target_data(
        fig, "samplemean_total_violations", expect_mean_total
    )


def test_metrics_plot_parallelplot_samplemean_violations_no_constraint():
    bench = jb.Benchmark(
        params={},
        solver=[solve_no_constraint],
    )
    result = bench()
    mplot = MetricsPlot(result)
    fig = mplot.parallelplot_experiment()

    # samplemean_total_violationsが含まれていないことを確認する（本テストでは制約がない問題を解く状況を想定しているため）
    def no_samplemean_total_violations(fig):
        for data in fig.data[0].dimensions:
            if data["label"] == "samplemean_total_violations":
                return False
        return True

    assert no_samplemean_total_violations(fig)


def test_metrics_plot_parallelplot_multipliers():
    params = {
        "multipliers": [{"onehot1": 1, "onehot2": 2}, {"onehot1": 3, "onehot2": 4}]
    }

    bench = jb.Benchmark(
        params=params,
        solver=[solve],
    )
    result = bench()
    mplot = MetricsPlot(result)
    fig = mplot.parallelplot_experiment()

    assert fig_contain_target_data(fig, "onehot1_multiplier", np.array([1, 3]))
    assert fig_contain_target_data(fig, "onehot2_multiplier", np.array([2, 4]))


params = {
    "have_constraint_sampleset": (solve, "samplemean_total_violations"),
    "no_constraint_sampleset": (solve_no_constraint, "samplemean_objective"),
}


@pytest.mark.parametrize(
    "solver, expect",
    list(params.values()),
    ids=list(params.keys()),
)
def test_metrics_plot_parallelplot_color_column_default(mocker, solver, expect):
    print(expect)
    m = mocker.patch("plotly.express.parallel_coordinates")
    bench = jb.Benchmark(
        params={},
        solver=[solver],
    )
    result = bench()
    mplot = MetricsPlot(result)
    mplot.parallelplot_experiment()

    args, kwargs = m.call_args

    def valid_call_arg_color(kwargs, color_value):
        if "color" in kwargs:
            if kwargs["color"] == color_value:
                return True
        return False

    assert valid_call_arg_color(kwargs, expect)


# color columnを設定するケースを書く
