import jijmodeling as jm
import numpy as np
import pandas as pd
import pytest

import jijbench as jb

from jijbench.visualize.metrics.plot import get_violations_dict, MetricsPlot
from jijbench.visualize.metrics.utils import construct_experiment_from_sampleset


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


def test_metrics_plot_get_violations_dict():
    series = pd.Series(
        data=[np.array([1, 1]), np.array([0, 2])],
        index=["num_occurrences", "onehot_violations"],
    )
    violations_dict = get_violations_dict(series)

    assert len(violations_dict.keys()) == 1
    assert (violations_dict["onehot_violations"] == np.array([0, 2])).all()


def test_metrics_plot_boxplot_violations():
    expect = 2

    multipliers = [{} for _ in range(expect)]
    bench = jb.Benchmark(
        params={"multipliers": multipliers},
        solver=solve,
    )
    result = bench()
    mplot = MetricsPlot(result)
    fig_ax_tuple = mplot.boxplot_violations()
    assert len(fig_ax_tuple) == expect
