import jijbench as jb
import openjij as oj
import jijmodeling as jm


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

    
