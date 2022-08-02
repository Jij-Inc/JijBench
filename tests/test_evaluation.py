from __future__ import annotations
from jijbench import evaluation

import pytest
import os, shutil

import dimod
import numpy as np
import openjij as oj

import jijbench as jb
import jijmodeling as jm


@pytest.fixture(scope="function", autouse=True)
def pre_post_process():
    # preprocess
    yield
    # postprocess
    norm_path = os.path.normcase("./.jb_results")
    if os.path.exists(norm_path):
        shutil.rmtree(norm_path)


@pytest.fixture
def bench_ps_eq_1p0():
    def solve(multipliers):
        d = jm.Placeholder("d", dim=1)
        x = jm.Binary("x", shape=(d.shape[0].set_latex("n")))
        i = jm.Element("i", d.shape[0])
        problem = jm.Problem("simple_problem")
        problem += jm.Sum(i, d[i] * x[i])
        problem += jm.Constraint("onehot", jm.Sum(i, d[i] * x[i]) == 1)
        instance_data = {"d": [1, 2, 3]}

        sampleset = dimod.SampleSet.from_samples(
            samples_like=[
                {"x[0]": 1, "x[1]": 0, "x[2]": 0},
            ],
            vartype="BINARY",
            energy=[1],
            num_occurrences=[4],
        )
        decoded_samples = problem.decode(sampleset, instance_data)
        print("fafafaf")
        print(decoded_samples.feasibles())
        print("fffffff")
        return sampleset, decoded_samples

    bench = jb.Benchmark(
        params={"multipliers": [{"onehot": 1}, {"onehot": 2}, {"onehot": 3}]},
        solver=solve,
    )
    bench.run()
    return bench


@pytest.fixture
def bench_ps_eq_0p5():
    def solve(multipliers):
        d = jm.Placeholder("d", dim=1)
        x = jm.Binary("x", shape=(d.shape[0].set_latex("n")))
        i = jm.Element("i", d.shape[0])
        problem = jm.Problem("simple_problem")
        problem += jm.Sum(i, d[i] * x[i])
        problem += jm.Constraint("onehot", jm.Sum(i, d[i] * x[i]) == 1)
        instance_data = {"d": [1, 2, 3]}

        sampleset = dimod.SampleSet.from_samples(
            samples_like=[
                {"0": 1, "1": 0, "2": 0},
                {"0": 0, "1": 0, "2": 0},
            ],
            vartype="BINARY",
            energy=[1],
            num_occurrences=[2, 2],
        )
        decoded_samples = problem.decode(sampleset, instance_data)
        return sampleset, decoded_samples

    bench = jb.Benchmark(
        params={"multipliers": [{"onehot": 1}, {"onehot": 2}, {"onehot": 3}]},
        solver=solve,
    )
    bench.run()
    return bench


@pytest.fixture
def bench_ps_eq_0p0():
    def solve(multipliers):
        d = jm.Placeholder("d", dim=1)
        x = jm.Binary("x", shape=(d.shape[0].set_latex("n")))
        i = jm.Element("i", d.shape[0])
        problem = jm.Problem("simple_problem")
        problem += jm.Sum(i, d[i] * x[i])
        problem += jm.Constraint("onehot", jm.Sum(i, d[i] * x[i]) == 1)
        instance_data = {"d": [1, 2, 3]}

        sampleset = dimod.SampleSet.from_samples(
            samples_like=[
                {"0": 0, "1": 0, "2": 0},
            ],
            vartype="BINARY",
            energy=[1],
            num_occurrences=[4],
        )
        decoded_samples = problem.decode(sampleset, instance_data)
        return sampleset, decoded_samples

    bench = jb.Benchmark(
        params={"multipliers": [{"onehot": 1}, {"onehot": 2}, {"onehot": 3}]},
        solver=solve,
    )
    bench.run()
    return bench


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
    evaluator.optimal_time_to_solution(opt_value=opt_value, expand=expand)
    evaluator.feasible_time_to_solution(expand=expand)
    evaluator.derived_time_to_solution(expand=expand)
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
    evaluator.time_to_solution(
        opt_value=opt_value, solution_type="optimal", expand=expand
    )
    evaluator.time_to_solution(solution_type="feasible", expand=expand)
    evaluator.time_to_solution(solution_type="derived", expand=expand)
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


def test_metrics_for_nan_column():
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


def test_success_probability(
    bench_ps_eq_1p0, bench_ps_eq_0p5, bench_ps_eq_0p0
):
    print()
    print(bench_ps_eq_1p0.table[["objective", "num_occurances", "num_feasible", "num_samples", "onehot_violations"]])
    # 成功確率1.0, 0.5, 0.0となるような解が得られた場合、evaluatorが本当にその値を返すかどうかのテスト。最適値は1.0
    print(bench_ps_eq_1p0.evaluate(opt_value=1.0))
    evaluator = jb.Evaluator(bench_ps_eq_1p0)
    metrics = evaluator.success_probability(opt_value=1)
    print(metrics)


#
# evaluator = jb.Evaluator(bench)
# evaluator.success_probability(opt_value=1)
