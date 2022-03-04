import jijbench as jb


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


