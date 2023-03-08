from __future__ import annotations

import os, shutil


import jijmodeling as jm
import pandas as pd
import pytest

import jijbench as jb
from unittest.mock import MagicMock


@pytest.fixture(scope="function", autouse=True)
def pre_post_process():
    # preprocess
    yield
    # postprocess
    norm_path = os.path.normcase("./.jb_results")
    if os.path.exists(norm_path):
        shutil.rmtree(norm_path)


def test_simple_experiment():
    e = jb.Experiment(name="simple_experiment")

    def func(i):
        return i**2

    for i in range(3):
        solver = jb.Solver(func)
        record = solver([jb.Parameter(i, "i")])
        e.append(record)
    e.save()


def test_simple_experiment_with_context_manager():
    e = jb.Experiment(name="simple_experiment_with_context_manager", autosave=True)

    def func(i):
        return i**2

    for i in range(3):
        with e:
            solver = jb.Solver(func)
            record = solver([jb.Parameter(i, "i")])
            e.append(record)


def test_construct_experiment():
    e = jb.Experiment(name="test")

    a = jb.Artifact({"x": {"0": jb.Number(1, "value")}})
    t = jb.Table(pd.DataFrame([[jb.Number(1, "value")]]))
    e.data = (a, t)

    a = e.artifact
    a.update({"y": 2})

    t = e.table
    t["x"] = [1]


def test_jijmodeling(
    sample_model: MagicMock,
    knapsack_problem: jm.Problem,
    knapsack_instance_data: jm.PH_VALUES_INTERFACE,
):
    experiment = jb.Experiment(autosave=False)

    with experiment:
        solver = jb.Solver(sample_model)
        x1 = jb.Parameter(knapsack_problem, name="model")
        x2 = jb.Parameter(knapsack_instance_data, name="feed_dict")
        record = solver([x1, x2])
        record.name = jb.ID().data
        experiment.append(record)

    droped_table = experiment.table.dropna(axis="columns")

    cols = droped_table.columns
    assert "energy" in cols
    assert "num_feasible" in cols

    assert sample_model.call_count == 1
    sample_model.assert_called_with(
        model=knapsack_problem, feed_dict=knapsack_instance_data
    )
