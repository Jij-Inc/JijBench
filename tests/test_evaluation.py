from __future__ import annotations

import os, shutil

import jijmodeling as jm
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


def test_success_probability(jm_sampleset: jm.SampleSet):
    from jijbench.functions.metrics import SuccessProbability

    sampleset = jb.SampleSet(jm_sampleset, "test_sampleset")

    opt_value = 3.0
    f = SuccessProbability()
    ps = f([sampleset], opt_value=opt_value)

    assert ps.data == 0.4


def test_feasible_rate(jm_sampleset: jm.SampleSet):
    from jijbench.functions.metrics import FeasibleRate

    sampleset = jb.SampleSet(jm_sampleset, "test_sampleset")

    f = FeasibleRate()
    feas_rate = f([sampleset])

    assert feas_rate.data == 0.7


def test_time_to_solution(jm_sampleset: jm.SampleSet):
    from jijbench.functions.metrics import TimeToSolution

    sampleset = jb.SampleSet(jm_sampleset, "test_sampleset")

    opt_value = 3.0
    pr = 0.7
    f = TimeToSolution()
    tts = f([sampleset], opt_value=opt_value, pr=pr)

    assert round(tts.data, 3) == 2.357

    tts = f([sampleset], pr, base="feasible")
    assert tts.data == 1.0

    tts = f([sampleset], pr, base="derived")
    assert round(tts.data, 3) == 2.357


def test_residual_energy(jm_sampleset: jm.SampleSet):
    from jijbench.functions.metrics import ResidualEnergy

    sampleset = jb.SampleSet(jm_sampleset, "test_sampleset")

    opt_value = 3.0
    f = ResidualEnergy()
    residual_energy = f([sampleset], opt_value=opt_value)

    assert residual_energy.data == 9.0


def test_evaluate_benchmark_results(sample_model):
    bench = jb.Benchmark({"num_reads": [1, 2]}, solver=sample_model)
    res = bench(autosave=False)

    opt_value = 3.0
    pr = 0.7
    evaluation = jb.Evaluation()
    res = evaluation([res], opt_value=opt_value, pr=pr)
