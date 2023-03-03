from __future__ import annotations

import os, pickle, shutil

from json import load

import dimod
import jijmodeling as jm
import numpy as np
import pandas as pd
import pytest

import jijbench as jb
from jijbench.consts.path import DEFAULT_RESULT_DIR
from unittest.mock import MagicMock


@pytest.fixture(scope="function", autouse=True)
def pre_post_process():
    # preprocess
    yield
    # postprocess
    norm_path = os.path.normcase("./.jb_results")
    # if os.path.exists(norm_path):
    #    shutil.rmtree(norm_path)


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


# def test_run_id():
#     experiment = jb.Experiment(autosave=#
#     row_num = #
#     for _ in range(row_num):
#         with experiment:
#             experiment.store_as_table({"num_reads": 10})
#             experiment.store_as_artifact({"dictobj": {"value": 10#
#     cols = experiment.table.columns
#     assert "num_reads" in cols
#     assert "dictobj" not in #
#     assert len(experiment.table.index) == row_num
#     assert len(experiment.artifact) == #
#     assert len(experiment.table["run_id"].unique()) == row_num
#     assert len(experiment.table["experiment_id"].unique()) == # #
# def test_store():
#     experiment = jb.Experiment(autosave=#
#     row_num = #
#     for _ in range(row_num):
#         with experiment:
#             experiment.store({"num_reads": 10, "dictobj": {"value": 10#
#     cols = experiment.table.columns
#     assert "num_reads" in #
#     assert len(experiment.table.index) == row_num
#     assert len(experiment.artifact) == #
#     assert len(experiment.table["run_id"].unique()) == row_num
#     assert len(experiment.table["experiment_id"].unique()) == # #
# def test_openjij():
#     sampler = oj.SASampler()
#     experiment = jb.Experiment(autosave=#
#     with experiment:
#         response = sampler.sample_qubo({(0, 1): 1})
#         experiment.store({"result": response#
#     droped_table = experiment.table.dropna(axis="columns#
#     cols = droped_table.columns
#     "energy" in cols
#     "energy_min" in # #
# def test_openjij_iteration():
#     sampler = oj.SASampler()
#     experiment = jb.Experiment(autosave=#
#     for _ in range(3):
#         with experiment:
#             response = sampler.sample_qubo({(0, 1): 1})
#             experiment.store({"result": response#
#     droped_table = experiment.table.dropna(axis="columns#
#     cols = droped_table.columns
#     "energy" in cols
#     "energy_min" in # #

# 以下のテストコードが通るように修正してください。
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


# 以下のテストコードが通るように修正してください。
# def test_jijmodeling_iteration(
#     sample_model: MagicMock,
#     knapsack_problem: jm.Problem,
#     knapsack_instance_data: jm.PH_VALUES_INTERFACE,
# ):
#     experiment = jb.Experiment(autosave=False)
#     for _ in range(3):
#         with experiment:
#             solver = jb.Solver(sample_model)
#             x1 = jb.Parameter(knapsack_problem, name="")
#             x2 = jb.Parameter(knapsack_instance_data)
#             record = solver([x1, x2])
#             record.name = jb.ID().data
#             experiment.append(record)
# 
#     droped_table = experiment.table.dropna(axis="columns")
# 
#     cols = droped_table.columns
#     assert "energy" in cols
#     assert "num_feasible" in cols
# 
#     assert sample_model.call_count == 3
#     sample_model.assert_called_with(
#         model=knapsack_problem, feed_dict=knapsack_instance_data
#     )


# ここはsample_modelを使わずに適当な関数を作ってsaveしたデータが呼び出せるか確認してください。
# def test_file_save_load(sample_model, knapsack_problem, knapsack_instance_data):
#     experiment = jb.Experiment(autosave=False)
#     for _ in range(3):
#         with experiment:
#             solver = jb.Solver(sample_model)
#             # x1 = jb.Parameter(knapsack_problem, name="")
#             # x2 = jb.Parameter(knapsack_instance_data)
#             record = solver([])
#             record.name = jb.ID().data
#             experiment.append(record)
# 
#     experiment.save()
#     
#     load_experiment = jb.load(DEFAULT_RESULT_DIR)  # ここでエラー出る
# 
#     original_cols = experiment.table.columns
#     load_cols = load_experiment.table.columns
#     for c in original_cols:
#         assert c in load_cols
#     assert len(experiment.table.index) == len(load_experiment.table.index)
#     assert len(experiment.artifact) == len(load_experiment.artifact)
#     for artifact in load_experiment.artifact.values():
#         # assert isinstance(artifact["record"], jm.SampleSet)  # あとで対応
#         print(artifact)  # あとで消す

    # assert experiment.artifact.timestamp == load_experiment.artifact.timestamp  # あとで対応


# def test_auto_save():
#     experiment = jb.Experiment(autosave=True)
#     num_rows = 3
#     for row in range(num_rows):
#         with experiment:
#             response = sample_qubo()
#             experiment.store({"result": response})
#         assert os.path.exists(
#             experiment._dir.artifact_dir
#             + os.path.normcase(f"/{experiment.run_id}/timestamp.txt")
#         )
#         load_experiment = jb.Experiment.load(
#             experiment_id=experiment.experiment_id, benchmark_id=experiment.benchmark_id
#         )
#         assert len(load_experiment.table) == row + # #
# def test_custome_dir_save():
#     custome_dir = os.path.normcase("./custom_result")
#     experiment = jb.Experiment(autosave=True, save_dir=custome_dir)
#     sampler = oj.SASampler()
#     num_rows = 3
#     for row in range(num_rows):
#         with experiment:
#             response = sampler.sample_qubo({(0, 1): 1})
#             experiment.store({"result": response})
#         assert os.path.exists(
#             experiment._dir.artifact_dir
#             + os.path.normcase(f"/{experiment.run_id}/timestamp.txt")
#         )
#         # print(experiment.run_id)
#         load_experiment = jb.Experiment.load(
#             experiment_id=experiment.experiment_id,
#             benchmark_id=experiment.benchmark_id,
#             save_dir=custome_dir,
#         )
#         assert len(load_experiment.table) == row + #
#     assert os.path.exists(custome_dir)
#     shutil.rmtree(# #
# def test_store_same_timestamp():
#     experiment = jb.Experiment(autosave=#
#     for _ in range(3):
#         with experiment:
#             jm_sampleset = decode()
#             experiment.store({"result": jm_sampleset#
#     run_id = list(experiment.artifact.keys())[#
#     artifact_timestamp = experiment._artifact.timestamp[run_id]
#     table_timestamp = experiment.table[experiment.table["run_id"] == run_id][
#         "timestamp"
#     ][#
#     assert artifact_timestamp == # #
# def test_insert_iterobj_into_table():
#     import numpy as #
#     experiment = jb.Experiment(autosave=#
#     with experiment:
#         experiment.table.dropna(axis=1, inplace=True)
#         record = {
#             "1d_list": [1],
#             "2d_list": [[1, 2], [1, 2]],
#             "1d_array": np.zeros(2),
#             "nd_array": np.random.normal(size=(10, 5, 4, 3, 2)),
#             "dict": {"a": {"b": 1}},
#         }
#         experiment.store_as_table(#
#     assert isinstance(experiment.table.loc[0, "1d_list"], list)
#     assert isinstance(experiment.table.loc[0, "2d_list"], list)
#     assert isinstance(experiment.table.loc[0, "1d_array"], np.ndarray)
#     assert isinstance(experiment.table.loc[0, "nd_array"], np.ndarray)
#     assert isinstance(experiment.table.loc[0, "dict"], # #
# def test_load_iterobj():
#     import numpy as #
#     benchmark_id = "example"
#     experiment_id = "test"
#     experiment = jb.Experiment(
#         experiment_id=experiment_id, benchmark_id=benchmark_id, autosave=True
#     #
#     with experiment:
#         experiment.table.dropna(axis=1, inplace=True)
#         record = {
#             "1d_list": [1],
#             "2d_list": [[1, 2], [1, 2]],
#             "1d_array": np.zeros(2),
#             "nd_array": np.random.normal(size=(10, 5, 4, 3, 2)),
#             "dict": {"a": {"b": 1}},
#         }
#         experiment.store_as_table(#
#     experiment.save()
#     del #
#     experiment = jb.Experiment.load(
#         experiment_id=experiment_id, benchmark_id=benchmark_id
#     #
#     assert isinstance(experiment.table.loc[0, "1d_list"], list)
#     assert isinstance(experiment.table.loc[0, "2d_list"], list)
#     assert isinstance(experiment.table.loc[0, "1d_array"], np.ndarray)
#     assert isinstance(experiment.table.loc[0, "nd_array"], np.ndarray)
#     assert isinstance(experiment.table.loc[0, "dict"], # #
# def test_sampling_and_execution_time():
#     experiment = jb.Experiment(autosave=#
#     with experiment:
#         response = sample_qubo()
#         experiment.store({"result": response#
#     assert np.isnan(experiment.table.sampling_time[0])
#     assert isinstance(experiment.table.execution_time[0], np.#
#     experiment = jb.Experiment(autosave=#
#     with experiment.start():
#         jm_sampleset = decode()
#         experiment.store({"result": jm_sampleset#
#     assert np.isnan(experiment.table.sampling_time[0])
#     assert isinstance(experiment.table.execution_time[0], np.# #
# def test_store_as_artifact_for_obj_cannot_pickle():
#     sampler = oj.SASampler#
#     experiment = jb.Experiment()
#     with experiment:
#         experiment.store_as_artifact(
#             {"sampler": sampler, "sample_qubo": sampler.sample_qubo, "value": 1.0}
#         #
#     loaded_experiment = jb.Experiment.load(
#         experiment_id=experiment.experiment_id, benchmark_id=experiment.benchmark_id
#     )
#     run_id = experiment.run_id
#     assert isinstance(loaded_experiment.artifact[run_id]["sampler"], str)
#     assert isinstance(loaded_experiment.artifact[run_id]["sample_qubo"], str)
#     assert loaded_experiment.artifact[run_id]["value"] == 1.0
