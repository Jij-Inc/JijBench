from __future__ import annotations

import os, shutil

import jijmodeling as jm
import numpy as np
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


def test_matrics_by_openjij():
    sampler = oj.SASampler(num_reads=10)
    experiment = jb.Experiment(autosave=False)
    for _ in range(3):
        with experiment.start():
            response = sampler.sample_qubo({(0, 1): 1})
            experiment.store({"result": response})

    evaluator = jb.Evaluator(experiment)
    opt_value = 1.0
    expand = True
    evaluator.tts(opt_value=opt_value, solution_type="optimal", expand=expand)
    evaluator.tts(solution_type="feasible", expand=expand)
    evaluator.tts(solution_type="derived", expand=expand)
    evaluator.success_probability(opt_value=opt_value, expand=expand)
    evaluator.feasible_rate(expand=expand)
    evaluator.residual_energy(opt_value=opt_value, expand=expand)

    columns = evaluator.table.columns

    assert "TTS(optimal)" in columns
    assert "TTS(feasible)" in columns
    assert "TTS(derived)" in columns
    assert "success_probability" in columns
    assert "feasible_rate" in columns
    assert "residual_energy" in columns


def test_matrics_by_jijmodeling():
    d = jm.Placeholder("d")
    x = jm.Binary("x", shape=(2,))
    problem = jm.Problem("sample")
    problem += x[0] + d * x[1]
    problem += jm.Constraint("onehot", x[:] == 1)
    ph_value = {"d": 2}
    pyq_obj = problem.to_pyqubo(ph_value=ph_value)
    pyq_model = pyq_obj.compile()
    sampler = oj.SASampler(num_reads=10)
    experiment = jb.Experiment(autosave=False)
    for _ in range(3):
        with experiment.start():
            bqm = pyq_model.to_bqm(feed_dict={"onehot": 1})
            response = sampler.sample(bqm)
            decoded = problem.decode(response, ph_value=ph_value)
            experiment.store({"response": response, "result": decoded})

    evaluator = jb.Evaluator(experiment)
    opt_value = 1.0
    expand = True
    evaluator.tts(opt_value=opt_value, solution_type="optimal", expand=expand)
    evaluator.tts(solution_type="feasible", expand=expand)
    evaluator.tts(solution_type="derived", expand=expand)
    evaluator.success_probability(opt_value=opt_value, expand=expand)
    evaluator.feasible_rate(expand=expand)
    evaluator.residual_energy(opt_value=opt_value, expand=expand)

    columns = evaluator.table.columns

    assert "TTS(optimal)" in columns
    assert "TTS(feasible)" in columns
    assert "TTS(derived)" in columns
    assert "success_probability" in columns
    assert "feasible_rate" in columns
    assert "residual_energy" in columns


def test_typical_metrics():
    sampler = oj.SASampler(num_reads=10)
    experiment = jb.Experiment(autosave=False)
    for _ in range(3):
        with experiment.start():
            response = sampler.sample_qubo({(0, 1): 1})
            experiment.store({"result": response})

    evaluator = jb.Evaluator(experiment)
    opt_value = 1.0
    pr = 0.99
    expand = True
    metrics = evaluator.calc_typical_metrics(opt_value=opt_value, pr=pr, expand=expand)

    table_columns = evaluator.table.columns
    metrics_columns = metrics.columns

    assert "TTS(optimal)" in table_columns
    assert "TTS(feasible)" in table_columns
    assert "TTS(derived)" in table_columns
    assert "success_probability" in table_columns
    assert "feasible_rate" in table_columns
    assert "residual_energy" in table_columns

    assert "TTS(optimal)" in metrics_columns
    assert "TTS(feasible)" in metrics_columns
    assert "TTS(derived)" in metrics_columns
    assert "success_probability" in metrics_columns
    assert "feasible_rate" in metrics_columns
    assert "residual_energy" in metrics_columns


def test_matrics_for_nan_column():
    sampler = oj.SASampler(num_reads=10)
    experiment = jb.Experiment(autosave=False)
    for _ in range(3):
        with experiment.start():
            response = sampler.sample_qubo({(0, 1): 1})
            experiment.store({"result": response})

    evaluator = jb.Evaluator(experiment)
    evaluator.table.execution_time = np.nan
    opt_value = 1.0
    pr = 0.99
    expand = False
    tts = evaluator.tts(opt_value=opt_value, pr=pr, expand=expand)
    for v in tts:
        assert np.isnan(v)
