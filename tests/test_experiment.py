import os
import shutil
import jijbench as jb
import openjij as oj
import pytest
import jijmodeling as jm


@pytest.fixture(scope="function", autouse=True)
def pre_post_process():
    # preprocess
    yield
    # postprocess
    if os.path.exists("./.jb_results"):
        shutil.rmtree("./.jb_results")


def test_run_id():
    experiment = jb.Experiment(autosave=False)

    row_num = 2

    for _ in range(row_num):
        with experiment.start():
            experiment.store_as_table({"num_reads": 10})
            experiment.store_as_artifact({"dictobj": {"value": 10}})
    
    cols = experiment.table.columns
    assert "num_reads" in cols
    assert "dictobj" not in cols

    assert len(experiment.table.index) == row_num
    assert len(experiment.artifact) == row_num

    assert len(experiment.table['run_id'].unique()) == row_num
    assert len(experiment.table['experiment_id'].unique()) == 1 


def test_store():
    experiment = jb.Experiment(autosave=False)

    row_num = 2

    for _ in range(row_num):
        with experiment.start():
            experiment.store({"num_reads": 10, "dictobj": {"value": 10}})
    
    cols = experiment.table.columns
    assert "num_reads" in cols

    assert len(experiment.table.index) == row_num
    assert len(experiment.artifact) == row_num

    assert len(experiment.table['run_id'].unique()) == row_num
    assert len(experiment.table['experiment_id'].unique()) == 1 


def test_openjij():
    sampler = oj.SASampler()
    experiment = jb.Experiment(autosave=False)

    with experiment.start():
        response = sampler.sample_qubo({(0, 1): 1})
        experiment.store({"result": response})
    

    droped_table = experiment.table.dropna(axis='columns')

    cols = droped_table.columns
    "energy" in cols
    "energy_min" in cols


def test_jijmodeling():
    d = jm.Placeholder("d")
    x = jm.Binary("x", shape=(2,))
    problem = jm.Problem("sample")
    problem += x[0] + d*x[1]

    ph_value = {"d": 2}
    pyq_obj = problem.to_pyqubo(ph_value=ph_value)
    pyq_model = pyq_obj.compile()

    sampler = oj.SASampler()
    experiment = jb.Experiment(autosave=False)

    with experiment.start():
        bqm = pyq_model.to_bqm()
        response = sampler.sample(bqm)
        decoded = problem.decode(response, ph_value=ph_value)
        experiment.store({"result": decoded})
    

    droped_table = experiment.table.dropna(axis='columns')

    cols = droped_table.columns
    "energy" in cols
    "energy_min" in cols
    "num_feasible" in cols


def test_file_save_load():
    d = jm.Placeholder("d")
    x = jm.Binary("x", shape=(2,))
    problem = jm.Problem("sample")
    problem += x[0] + d*x[1]

    ph_value = {"d": 2}
    pyq_obj = problem.to_pyqubo(ph_value=ph_value)
    pyq_model = pyq_obj.compile()

    sampler = oj.SASampler()
    experiment = jb.Experiment(autosave=False)

    for _ in range(3):
        with experiment.start():
            bqm = pyq_model.to_bqm()
            response = sampler.sample(bqm)
            decoded = problem.decode(response, ph_value=ph_value)
            experiment.store({"result": decoded})
    
    experiment.save()

    load_experiment = jb.Experiment.load(experiment_id=experiment.experiment_id, benchmark_id=experiment.benchmark_id)

    original_cols = experiment.table.columns
    load_cols = load_experiment.table.columns
    for c in original_cols:
        c in load_cols
    
    assert len(experiment.table.index) == len(load_experiment.table.index)
    assert len(experiment.artifact) == len(load_experiment.artifact)
    for artifact in load_experiment.artifact.values():
        assert isinstance(artifact["result"], jm.DecodedSamples)
    
    assert experiment._artifact_timestamp == load_experiment._artifact_timestamp

def test_auto_save():
    experiment = jb.Experiment(autosave=True)
    sampler = oj.SASampler()
    num_rows = 3
    for row in range(num_rows):
        with experiment.start():
            response = sampler.sample_qubo({(0, 1): 1})
            experiment.store({"result": response})
        assert os.path.exists(experiment._dirs.artifact_dir + f"/{experiment.run_id}/timestamp.txt")
        load_experiment = jb.Experiment.load(experiment_id=experiment.experiment_id, benchmark_id=experiment.benchmark_id)
        assert len(load_experiment.table) == row + 1


def test_custome_dir_save():
    custome_dir = "./custom_result"
    experiment = jb.Experiment(autosave=True, save_dir=custome_dir)
    sampler = oj.SASampler()
    num_rows = 3
    for row in range(num_rows):
        with experiment.start():
            response = sampler.sample_qubo({(0, 1): 1})
            experiment.store({"result": response})
        assert os.path.exists(experiment._dirs.artifact_dir + f"/{experiment.run_id}/timestamp.txt")
        load_experiment = jb.Experiment.load(experiment_id=experiment.experiment_id, benchmark_id=experiment.benchmark_id, save_dir=custome_dir)
        assert len(load_experiment.table) == row + 1

    assert os.path.exists(custome_dir)
    shutil.rmtree(custome_dir)



def test_store_same_timestamp():
    d = jm.Placeholder("d")
    x = jm.Binary("x", shape=(2,))
    problem = jm.Problem("sample")
    problem += x[0] + d*x[1]

    ph_value = {"d": 2}
    pyq_obj = problem.to_pyqubo(ph_value=ph_value)
    pyq_model = pyq_obj.compile()

    sampler = oj.SASampler()
    experiment = jb.Experiment(autosave=False)

    for _ in range(3):
        with experiment.start():
            bqm = pyq_model.to_bqm()
            response = sampler.sample(bqm)
            decoded = problem.decode(response, ph_value=ph_value)
            experiment.store({"result": decoded})

    run_id = list(experiment.artifact.keys())[0]

    artifact_timestamp = experiment._artifact_timestamp[run_id]
    table_timestamp = experiment.table[experiment.table["run_id"] == run_id]["timestamp"][0]

    assert artifact_timestamp == table_timestamp


    
