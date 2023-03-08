import jijmodeling as jm

import numpy as np
import pandas as pd
import pytest

import jijbench as jb

from jijbench.visualization.metrics.utils import (
    construct_experiment_from_samplesets,
    _create_fig_title_list,
    _df_has_valid_multipliers_column,
    _df_has_number_array_column_target_name,
    _df_has_number_column_target_name,
)


def test_construct_experiment_from_samplesets():
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


def test_construct_experiment_from_samplesets_give_raw_sampleset():
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


def test_construct_experiment_from_samplesets_additional_data():
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


def test_construct_experiment_from_samplesets_additional_data_invalid_length():
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
def test_create_fig_title_list(input_title, expect):
    series = pd.Series(
        data=[1, 2],
        index=["1", "2"],
    )
    series.index.names = ["i"]

    title_list = _create_fig_title_list(
        metrics=series,
        title=input_title,
    )
    assert title_list == expect


def test_create_fig_title_list_for_series_with_no_index():
    series = pd.Series(
        data=[1, 2],
        index=[None, None],
    )
    title_list = _create_fig_title_list(
        metrics=series,
        title=None,
    )
    assert title_list == ["", ""]


def test_create_fig_title_list_for_series_with_default_index():
    series = pd.Series(
        data=[1, 2],
    )
    title_list = _create_fig_title_list(
        metrics=series,
        title=None,
    )
    assert title_list == ["index: 0", "index: 1"]


def test_create_fig_title_list_for_invalid_input():
    invalid_input_title = 0

    series = pd.Series(
        data=[1, 2],
        index=["1", "2"],
    )
    series.index.names = ["i"]
    with pytest.raises(TypeError):
        _create_fig_title_list(
            metrics=series,
            title=invalid_input_title,
        )


params = {
    "no_multipliers_columns": ([[0, 1], [2, 3]], ["col_0", "col_1"], False),
    "first_multipliers_isnot_dict": ([[0]], ["multipliers"], False),
    "second_multipliers_isnot_dict": ([[{"onehot": 1}], [0]], ["multipliers"], False),
    "multipliers_key_isnot_str": ([[{0: 1}]], ["multipliers"], False),
    "multipliers_value_isnot_number": ([[{"onehot": "1"}]], ["multipliers"], False),
    "constraint_name_is_different": (
        [[{"onehot1": 1}], [{"onehot2": 2}]],
        ["multipliers"],
        False,
    ),
    "valid_case": ([[{"onehot1": 1}], [{"onehot1": 2}]], ["multipliers"], True),
}


@pytest.mark.parametrize(
    "data, columns, expect",
    list(params.values()),
    ids=list(params.keys()),
)
def test_df_has_valid_multipliers_column(data, columns, expect):
    def create_df(data, columns):
        df = pd.DataFrame(columns=columns)
        for i, row in enumerate(data):
            for j, element in enumerate(row):
                df.at[i, columns[j]] = object
                df.at[i, columns[j]] = element
        return df

    df = create_df(data, columns)
    assert _df_has_valid_multipliers_column(df) == expect


params = {
    "no_target_columns": ("number_array", [[0, 1], [2, 3]], ["col_0", "col_1"], False),
    "target_column_isnot_array": ("number_array", [[0]], ["number_array"], False),
    "target_column_element_isnot_number": (
        "number_array",
        [[["a"]]],
        ["number_array"],
        False,
    ),
    "target_column_is_valid_list": (
        "number_array",
        [[[0, 1]], [[2, 3]]],
        ["number_array"],
        True,
    ),
    "target_column_is_valid_nparray": (
        "number_array",
        [[np.array([0, 1])], [np.array([2, 3])]],
        ["number_array"],
        True,
    ),
}


@pytest.mark.parametrize(
    "target_column, data, columns, expect",
    list(params.values()),
    ids=list(params.keys()),
)
def test_df_has_number_array_column_target_name(target_column, data, columns, expect):
    def create_df(data, columns):
        df = pd.DataFrame(columns=columns)
        for i, row in enumerate(data):
            for j, element in enumerate(row):
                df.at[i, columns[j]] = object
                df.at[i, columns[j]] = element
        return df

    df = create_df(data, columns)
    assert (
        _df_has_number_array_column_target_name(df, column_name=target_column) == expect
    )


params = {
    "no_target_columns": ("number", [[0, 1], [2, 3]], ["col_0", "col_1"], False),
    "target_column_is_list": ("number", [[[0]]], ["number"], False),
    "target_column_is_str": ("number", [["0"]], ["number"], False),
    "target_column_is_number": (
        "number",
        [[0], [1]],
        ["number"],
        True,
    ),
}


@pytest.mark.parametrize(
    "target_column, data, columns, expect",
    list(params.values()),
    ids=list(params.keys()),
)
def test_df_has_number_column_target_name(target_column, data, columns, expect):
    def create_df(data, columns):
        df = pd.DataFrame(columns=columns)
        for i, row in enumerate(data):
            for j, element in enumerate(row):
                df.at[i, columns[j]] = object
                df.at[i, columns[j]] = element
        return df

    df = create_df(data, columns)
    assert _df_has_number_column_target_name(df, column_name=target_column) == expect
