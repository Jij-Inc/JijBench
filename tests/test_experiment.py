from __future__ import annotations

import os, shutil

import jijmodeling as jm
import openjij as oj
import pytest

import jijbench as jb


@pytest.fixture(scope="function", autouse=True)
def pre_post_process():
    # preprocess
    yield
    # postprocess
    norm_path = os.path.normcase("./.jb_results")
    if os.path.exists(norm_path):
        shutil.rmtree(norm_path)


def test_run_id():
    experiment = jb.Experiment(autosave=False)

    row_num = 2

    for _ in range(row_num):
        with experiment:
            experiment.store_as_table({"num_reads": 10})
            experiment.store_as_artifact({"dictobj": {"value": 10}})

    cols = experiment.table.columns
    assert "num_reads" in cols
    assert "dictobj" not in cols

    assert len(experiment.table.index) == row_num
    assert len(experiment.artifact) == row_num

    assert len(experiment.table["run_id"].unique()) == row_num
    assert len(experiment.table["experiment_id"].unique()) == 1


def test_store():
    experiment = jb.Experiment(autosave=False)

    row_num = 2

    for _ in range(row_num):
        with experiment:
            experiment.store({"num_reads": 10, "dictobj": {"value": 10}})

    cols = experiment.table.columns
    assert "num_reads" in cols

    assert len(experiment.table.index) == row_num
    assert len(experiment.artifact) == row_num

    assert len(experiment.table["run_id"].unique()) == row_num
    assert len(experiment.table["experiment_id"].unique()) == 1


def test_openjij():
    sampler = oj.SASampler()
    experiment = jb.Experiment(autosave=False)

    with experiment:
        response = sampler.sample_qubo({(0, 1): 1})
        experiment.store({"result": response})

    droped_table = experiment.table.dropna(axis="columns")

    cols = droped_table.columns
    "energy" in cols
    "energy_min" in cols


def test_openjij_iteration():
    sampler = oj.SASampler()
    experiment = jb.Experiment(autosave=False)

    for _ in range(3):
        with experiment:
            response = sampler.sample_qubo({(0, 1): 1})
            experiment.store({"result": response})

    droped_table = experiment.table.dropna(axis="columns")

    cols = droped_table.columns
    "energy" in cols
    "energy_min" in cols


def test_jijmodeling():
    d = jm.Placeholder("d")
    x = jm.Binary("x", shape=(2,))
    problem = jm.Problem("sample")
    problem += x[0] + d * x[1]
    problem += jm.Constraint("onehot", x[:] == 1)

    ph_value = {"d": 2}
    pyq_obj = problem.to_pyqubo(ph_value=ph_value)
    pyq_model = pyq_obj.compile()

    sampler = oj.SASampler()
    experiment = jb.Experiment(autosave=False)

    with experiment:
        bqm = pyq_model.to_bqm(feed_dict={"onehot": 1})
        response = sampler.sample(bqm)
        decoded = problem.decode(response, ph_value=ph_value)
        experiment.store({"result": decoded})

    droped_table = experiment.table.dropna(axis="columns")

    cols = droped_table.columns
    "energy" in cols
    "energy_min" in cols
    "num_feasible" in cols


def test_jijmodeling_iteration():
    d = jm.Placeholder("d")
    x = jm.Binary("x", shape=(2,))
    problem = jm.Problem("sample")
    problem += x[0] + d * x[1]
    problem += jm.Constraint("onehot", x[:] == 1)
    problem += jm.Constraint("onehot2", x[:] == 1)

    ph_value = {"d": 2}
    pyq_obj = problem.to_pyqubo(ph_value=ph_value)
    pyq_model = pyq_obj.compile()

    sampler = oj.SASampler()
    experiment = jb.Experiment(autosave=False)

    for _ in range(3):
        with experiment:
            bqm = pyq_model.to_bqm(feed_dict={"onehot": 1, "onehot2": 2})
            response = sampler.sample(bqm)
            decoded = problem.decode(response, ph_value=ph_value)
            experiment.store({"result": decoded})

    droped_table = experiment.table.dropna(axis="columns")

    cols = droped_table.columns
    "energy" in cols
    "energy_min" in cols
    "num_feasible" in cols


def test_file_save_load():
    d = jm.Placeholder("d")
    x = jm.Binary("x", shape=(2,))
    problem = jm.Problem("sample")
    problem += x[0] + d * x[1]

    ph_value = {"d": 2}
    pyq_obj = problem.to_pyqubo(ph_value=ph_value)
    pyq_model = pyq_obj.compile()

    sampler = oj.SASampler()
    experiment = jb.Experiment(autosave=False)

    for _ in range(3):
        with experiment:
            bqm = pyq_model.to_bqm()
            response = sampler.sample(bqm)
            decoded = problem.decode(response, ph_value=ph_value)
            experiment.store({"result": decoded})

    experiment.save()

    load_experiment = jb.Experiment.load(
        experiment_id=experiment.experiment_id, benchmark_id=experiment.benchmark_id
    )

    original_cols = experiment.table.columns
    load_cols = load_experiment.table.columns
    for c in original_cols:
        c in load_cols

    assert len(experiment.table.index) == len(load_experiment.table.index)
    assert len(experiment.artifact) == len(load_experiment.artifact)
    for artifact in load_experiment.artifact.values():
        assert isinstance(artifact["result"], jm.DecodedSamples)

    assert experiment._artifact.timestamp == load_experiment._artifact.timestamp


def test_auto_save():
    experiment = jb.Experiment(autosave=True)
    sampler = oj.SASampler()
    num_rows = 3
    for row in range(num_rows):
        with experiment:
            response = sampler.sample_qubo({(0, 1): 1})
            experiment.store({"result": response})
        assert os.path.exists(
            experiment._dir.artifact_dir
            + os.path.normcase(f"/{experiment.run_id}/timestamp.txt")
        )
        load_experiment = jb.Experiment.load(
            experiment_id=experiment.experiment_id, benchmark_id=experiment.benchmark_id
        )
        assert len(load_experiment.table) == row + 1


def test_custome_dir_save():
    custome_dir = os.path.normcase("./custom_result")
    experiment = jb.Experiment(autosave=True, save_dir=custome_dir)
    sampler = oj.SASampler()
    num_rows = 3
    for row in range(num_rows):
        with experiment:
            response = sampler.sample_qubo({(0, 1): 1})
            experiment.store({"result": response})
        assert os.path.exists(
            experiment._dir.artifact_dir
            + os.path.normcase(f"/{experiment.run_id}/timestamp.txt")
        )
        # print(experiment.run_id)
        load_experiment = jb.Experiment.load(
            experiment_id=experiment.experiment_id,
            benchmark_id=experiment.benchmark_id,
            save_dir=custome_dir,
        )
        assert len(load_experiment.table) == row + 1

    assert os.path.exists(custome_dir)
    shutil.rmtree(custome_dir)


def test_store_same_timestamp():
    d = jm.Placeholder("d")
    x = jm.Binary("x", shape=(2,))
    problem = jm.Problem("sample")
    problem += x[0] + d * x[1]

    ph_value = {"d": 2}
    pyq_obj = problem.to_pyqubo(ph_value=ph_value)
    pyq_model = pyq_obj.compile()

    sampler = oj.SASampler()
    experiment = jb.Experiment(autosave=False)

    for _ in range(3):
        with experiment:
            bqm = pyq_model.to_bqm()
            response = sampler.sample(bqm)
            decoded = problem.decode(response, ph_value=ph_value)
            experiment.store({"result": decoded})

    run_id = list(experiment.artifact.keys())[0]

    artifact_timestamp = experiment._artifact.timestamp[run_id]
    table_timestamp = experiment.table[experiment.table["run_id"] == run_id][
        "timestamp"
    ][0]

    assert artifact_timestamp == table_timestamp


def test_insert_iterobj_into_table():
    import numpy as np

    experiment = jb.Experiment(autosave=False)

    with experiment:
        experiment.table.dropna(axis=1, inplace=True)
        record = {
            "1d_list": [1],
            "2d_list": [[1, 2], [1, 2]],
            "1d_array": np.zeros(2),
            "nd_array": np.random.normal(size=(10, 5, 4, 3, 2)),
            "dict": {"a": {"b": 1}},
        }
        experiment.store_as_table(record)

    assert isinstance(experiment.table.loc[0, "1d_list"], list)
    assert isinstance(experiment.table.loc[0, "2d_list"], list)
    assert isinstance(experiment.table.loc[0, "1d_array"], np.ndarray)
    assert isinstance(experiment.table.loc[0, "nd_array"], np.ndarray)
    assert isinstance(experiment.table.loc[0, "dict"], dict)


def test_load_iterobj():
    import numpy as np

    benchmark_id = "example"
    experiment_id = "test"
    experiment = jb.Experiment(
        experiment_id=experiment_id, benchmark_id=benchmark_id, autosave=True
    )

    with experiment:
        experiment.table.dropna(axis=1, inplace=True)
        record = {
            "1d_list": [1],
            "2d_list": [[1, 2], [1, 2]],
            "1d_array": np.zeros(2),
            "nd_array": np.random.normal(size=(10, 5, 4, 3, 2)),
            "dict": {"a": {"b": 1}},
        }
        experiment.store_as_table(record)

    experiment.save()
    del experiment

    experiment = jb.Experiment.load(
        experiment_id=experiment_id, benchmark_id=benchmark_id
    )

    assert isinstance(experiment.table.loc[0, "1d_list"], list)
    assert isinstance(experiment.table.loc[0, "2d_list"], list)
    assert isinstance(experiment.table.loc[0, "1d_array"], np.ndarray)
    assert isinstance(experiment.table.loc[0, "nd_array"], np.ndarray)
    assert isinstance(experiment.table.loc[0, "dict"], dict)


def test_sampling_and_execution_time():
    import numpy as np

    sampler = oj.SASampler()
    experiment = jb.Experiment(autosave=False)

    with experiment:
        response = sampler.sample_qubo({(0, 1): 1})
        experiment.store({"result": response})

    experiment.table.dropna(axis="columns", inplace=True)

    assert isinstance(experiment.table.sampling_time[0], np.float64)
    assert isinstance(experiment.table.execution_time[0], np.float64)

    d = jm.Placeholder("d")
    x = jm.Binary("x", shape=(2,))
    problem = jm.Problem("sample")
    problem += x[0] + d * x[1]
    problem += jm.Constraint("onehot", x[:] == 1)

    ph_value = {"d": 2}
    pyq_obj = problem.to_pyqubo(ph_value=ph_value)
    pyq_model = pyq_obj.compile()

    sampler = oj.SASampler()
    experiment = jb.Experiment(autosave=False)

    with experiment.start():
        bqm = pyq_model.to_bqm(feed_dict={"onehot": 1})
        response = sampler.sample(bqm)
        decoded = problem.decode(response, ph_value=ph_value)
        experiment.store({"result": decoded})

    assert np.isnan(experiment.table.sampling_time[0])
    assert np.isnan(experiment.table.execution_time[0])
