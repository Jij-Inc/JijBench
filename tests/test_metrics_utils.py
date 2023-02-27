import jijmodeling as jm

import numpy as np
import pandas as pd
import pytest

import jijbench as jb

from jijbench.visualize.metrics.utils import (
    construct_experiment_from_samplesets,
    create_fig_title_list,
)


def test_utils_construct_experiment_from_samplesets():
    num_occ1, num_occ2 = 1, 2

    sampleset1 = jm.SampleSet(
        record=jm.Record(
            solution={"x": [(([3],), [1], (6,))]},
            num_occurrences=[num_occ1],
        ),
        evaluation=jm.Evaluation(
            energy=[-8.0],
            objective=[-8.0],
            constraint_violations={"onehot": [0.0]},
            penalty={},
        ),
        measuring_time=jm.MeasuringTime(),
    )

    sampleset2 = jm.SampleSet(
        record=jm.Record(
            solution={"x": [(([3],), [1], (6,))]},
            num_occurrences=[num_occ2],
        ),
        evaluation=jm.Evaluation(
            energy=[-8.0],
            objective=[-8.0],
            constraint_violations={"onehot": [0.0]},
            penalty={},
        ),
        measuring_time=jm.MeasuringTime(),
    )

    samplesets = [sampleset1, sampleset2]

    experiment = construct_experiment_from_samplesets(samplesets)

    assert isinstance(experiment, jb.Experiment)
    assert len(experiment.table) == 2
    assert experiment.table["num_occurrences"].values[0] == np.array([num_occ1])
    assert experiment.table["num_occurrences"].values[1] == np.array([num_occ2])


def test_utils_construct_experiment_from_samplesets_give_raw_sampleset():
    sampleset = jm.SampleSet(
        record=jm.Record(
            solution={"x": [(([3],), [1], (6,))]},
            num_occurrences=[1],
        ),
        evaluation=jm.Evaluation(
            energy=[-8.0],
            objective=[-8.0],
            constraint_violations={"onehot": [0.0]},
            penalty={},
        ),
        measuring_time=jm.MeasuringTime(),
    )

    experiment = construct_experiment_from_samplesets(sampleset)

    assert isinstance(experiment, jb.Experiment)
    assert len(experiment.table) == 1
    assert experiment.table["num_occurrences"].values[0] == np.array([1])


def test_utils_construct_experiment_from_samplesets_additional_data():
    additional_data = {
        "data1": [0, 1],
        "data2": [np.array([2, 3]), np.array([4, 5])],
    }

    sampleset1 = jm.SampleSet(
        record=jm.Record(
            solution={"x": [(([3],), [1], (6,))]},
            num_occurrences=[1],
        ),
        evaluation=jm.Evaluation(
            energy=[-8.0],
            objective=[-8.0],
            constraint_violations={"onehot": [0.0]},
            penalty={},
        ),
        measuring_time=jm.MeasuringTime(),
    )

    sampleset2 = jm.SampleSet(
        record=jm.Record(
            solution={"x": [(([3],), [1], (6,))]},
            num_occurrences=[2],
        ),
        evaluation=jm.Evaluation(
            energy=[-8.0],
            objective=[-8.0],
            constraint_violations={"onehot": [0.0]},
            penalty={},
        ),
        measuring_time=jm.MeasuringTime(),
    )
    samplesets = [sampleset1, sampleset2]

    experiment = construct_experiment_from_samplesets(samplesets, additional_data)

    assert isinstance(experiment, jb.Experiment)
    assert len(experiment.table) == 2
    assert experiment.table["data1"].values[0] == 0
    assert experiment.table["data1"].values[1] == 1
    assert (experiment.table["data2"].values[0] == np.array([2, 3])).all()
    assert (experiment.table["data2"].values[1] == np.array([4, 5])).all()


def test_utils_construct_experiment_from_samplesets_additional_data_invalid_length():
    additional_data = {
        "data1": [0],
    }

    sampleset1 = jm.SampleSet(
        record=jm.Record(
            solution={"x": [(([3],), [1], (6,))]},
            num_occurrences=[1],
        ),
        evaluation=jm.Evaluation(
            energy=[-8.0],
            objective=[-8.0],
            constraint_violations={"onehot": [0.0]},
            penalty={},
        ),
        measuring_time=jm.MeasuringTime(),
    )

    sampleset2 = jm.SampleSet(
        record=jm.Record(
            solution={"x": [(([3],), [1], (6,))]},
            num_occurrences=[2],
        ),
        evaluation=jm.Evaluation(
            energy=[-8.0],
            objective=[-8.0],
            constraint_violations={"onehot": [0.0]},
            penalty={},
        ),
        measuring_time=jm.MeasuringTime(),
    )
    samplesets = [sampleset1, sampleset2]

    with pytest.raises(TypeError):
        construct_experiment_from_samplesets(samplesets, additional_data)


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
