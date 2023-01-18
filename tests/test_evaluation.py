from __future__ import annotations

import os, shutil

import dimod
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


@pytest.fixture
def bench_for_success_probability_eq_1p0():
    def solve():
        d = jm.Placeholder("d", dim=1)
        x = jm.Binary("x", shape=(d.shape[0].set_latex("n")))
        i = jm.Element("i", d.shape[0])
        problem = jm.Problem("simple_problem")
        problem += jm.Sum(i, d[i] * x[i])
        problem += jm.Constraint("onehot", jm.Sum(i, d[i] * x[i]) == 1)

        jm_sampleset_dict = {
            "record": {
                "solution": {"x": [(([0],), [1], (3,))]},
                "num_occurrences": [4],
            },
            "evaluation": {
                "energy": [1.0],
                "objective": [1.0],
                "constraint_violations": {"onehot": [0.0]},
                "penalty": {},
            },
            "measuring_time": {
                "solve": None,
                "system": None,
                "total": None,
            },
        }
        jm_sampleset = jm.SampleSet.from_serializable(jm_sampleset_dict)
        solving_time = jm.SolvingTime(
            **{"preprocess": 1.0, "solve": 1.0, "postprocess": 1.0}
        )
        jm_sampleset.measuring_time.solve = solving_time
        return jm_sampleset

    bench = jb.Benchmark(
        params={"multipliers": [{"onehot": 1}, {"onehot": 2}, {"onehot": 3}]},
        solver=solve,
    )
    bench.run()
    return bench


@pytest.fixture
def bench_for_success_probability_eq_0p5():
    def solve():
        d = jm.Placeholder("d", dim=1)
        x = jm.Binary("x", shape=(d.shape[0].set_latex("n")))
        i = jm.Element("i", d.shape[0])
        problem = jm.Problem("simple_problem")
        problem += jm.Sum(i, d[i] * x[i])
        problem += jm.Constraint("onehot", jm.Sum(i, d[i] * x[i]) == 1)

        jm_sampleset_dict = {
            "record": {
                "solution": {"x": [(([0],), [1], (3,)), (([],), [], (3,))]},
                "num_occurrences": [2, 2],
            },
            "evaluation": {
                "energy": [1.0, 1.0],
                "objective": [1.0, 0.0],
                "constraint_violations": {"onehot": [0.0, 1.0]},
                "penalty": {},
            },
            "measuring_time": {
                "solve": None,
                "system": None,
                "total": None,
            },
        }
        jm_sampleset = jm.SampleSet.from_serializable(jm_sampleset_dict)
        solving_time = jm.SolvingTime(
            **{"preprocess": 1.0, "solve": 1.0, "postprocess": 1.0}
        )
        jm_sampleset.measuring_time.solve = solving_time
        return jm_sampleset

    bench = jb.Benchmark(
        params={"multipliers": [{"onehot": 1}, {"onehot": 2}, {"onehot": 3}]},
        solver=solve,
    )
    bench.run()
    return bench


@pytest.fixture
def bench_for_success_probability_eq_0p0():
    def solve():
        d = jm.Placeholder("d", dim=1)
        x = jm.Binary("x", shape=(d.shape[0].set_latex("n")))
        i = jm.Element("i", d.shape[0])
        problem = jm.Problem("simple_problem")
        problem += jm.Sum(i, d[i] * x[i])
        problem += jm.Constraint("onehot", jm.Sum(i, d[i] * x[i]) == 1)

        jm_sampleset_dict = {
            "record": {"solution": {"x": [(([],), [], (3,))]}, "num_occurrences": [4]},
            "evaluation": {
                "energy": [1.0],
                "objective": [0.0],
                "constraint_violations": {"onehot": [1.0]},
                "penalty": {},
            },
            "measuring_time": {
                "solve": None,
                "system": None,
                "total": None,
            },
        }
        jm_sampleset = jm.SampleSet.from_serializable(jm_sampleset_dict)
        solving_time = jm.SolvingTime(
            **{"preprocess": 1.0, "solve": 1.0, "postprocess": 1.0}
        )
        jm_sampleset.measuring_time.solve = solving_time
        return jm_sampleset

    bench = jb.Benchmark(
        params={"multipliers": [{"onehot": 1}, {"onehot": 2}, {"onehot": 3}]},
        solver=solve,
    )
    bench.run()
    return bench


@pytest.fixture
def bench_for_multi_const_problem():
    def solve():
        d = jm.Placeholder("d", dim=2)
        x = jm.Binary("x", shape=d.shape)
        i = jm.Element("i", d.shape[0])
        j = jm.Element("j", d.shape[1])

        problem = jm.Problem("simple_problem")
        problem += jm.Sum([i, j], d[i, j] * x[i, j])
        problem += jm.Constraint("onehot1", x[i, :] == 1, forall=i)
        problem += jm.Constraint("onehot2", x[:, j] == 1, forall=j)

        jm_sampleset_dict = {
            "record": {
                "solution": {
                    "x": [
                        (([0, 1], [0, 1]), [1, 1], (2, 2)),
                        (([0, 1], [1, 0]), [1, 1], (2, 2)),
                        (([], []), [], (2, 2)),
                        (([0, 1], [0, 0]), [1, 1], (2, 2)),
                    ]
                },
                "num_occurrences": [4, 3, 2, 1],
            },
            "evaluation": {
                "energy": [3.0, 24.0, 0.0, 20.0],
                "objective": [3.0, 24.0, 0.0, 17.0],
                "constraint_violations": {
                    "onehot1": [0.0, 0.0, 2.0, 0.0],
                    "onehot2": [0.0, 0.0, 2.0, 2.0],
                },
                "penalty": {},
            },
            "measuring_time": {"solve": None, "system": None, "total": None},
        }
        jm_sampleset = jm.SampleSet.from_serializable(jm_sampleset_dict)
        solving_time = jm.SolvingTime(
            **{"preprocess": 1.0, "solve": 1.0, "postprocess": 1.0}
        )
        jm_sampleset.measuring_time.solve = solving_time
        return jm_sampleset

    bench = jb.Benchmark(
        params={"multipliers": [{"onehot": 1}, {"onehot": 2}, {"onehot": 3}]},
        solver=solve,
    )
    bench.run()
    return bench


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


def test_single_metrics(
    bench_for_success_probability_eq_1p0,
    bench_for_success_probability_eq_0p5,
    bench_for_success_probability_eq_0p0,
):
    opt_value = 1.0
    # 成功確率1.0, 0.5, 0.0となるような解が得られた場合、evaluatorが本当にその値を返すかどうかのテスト。最適値は1.0
    evaluator = jb.Evaluator(bench_for_success_probability_eq_1p0)
    metrics = evaluator.calc_typical_metrics(opt_value=opt_value)
    assert (
        metrics["success_probability"][0]
        == evaluator.success_probability(opt_value=opt_value)[0]
    )
    assert metrics["feasible_rate"][0] == evaluator.feasible_rate()[0]
    assert (
        metrics["residual_energy"][0]
        == evaluator.residual_energy(opt_value=opt_value)[0]
    )
    if np.isnan(metrics["TTS(optimal)"][0]):
        assert np.isnan(evaluator.optimal_time_to_solution(opt_value=opt_value)[0])
    else:
        assert (
            metrics["TTS(optimal)"][0]
            == evaluator.optimal_time_to_solution(opt_value=opt_value)[0]
        )
    if np.isnan(metrics["TTS(feasible)"][0]):
        assert np.isnan(evaluator.feasible_time_to_solution()[0])
    else:
        assert metrics["TTS(feasible)"][0] == evaluator.feasible_time_to_solution()[0]
    if np.isnan(metrics["TTS(derived)"][0]):
        assert np.isnan(evaluator.derived_time_to_solution()[0])
    else:
        assert metrics["TTS(derived)"][0] == evaluator.derived_time_to_solution()[0]

    evaluator = jb.Evaluator(bench_for_success_probability_eq_0p5)
    metrics = evaluator.calc_typical_metrics(opt_value=opt_value)

    assert metrics["success_probability"][0] == 0.5
    assert metrics["feasible_rate"][0] == 0.5
    assert metrics["residual_energy"][0] == 0.0

    evaluator = jb.Evaluator(bench_for_success_probability_eq_0p0)
    metrics = evaluator.calc_typical_metrics(opt_value=opt_value)

    assert metrics["success_probability"][0] == 0.0
    assert metrics["feasible_rate"][0] == 0.0
    assert np.isnan(metrics["residual_energy"][0])
    assert metrics["TTS(optimal)"][0] == np.inf
    assert metrics["TTS(feasible)"][0] == np.inf
    assert metrics["TTS(derived)"][0] == np.inf


def test_metrics_given_dimod_sampleset():
    sampler = oj.SASampler(num_reads=10)
    experiment = jb.Experiment(autosave=False)
    for _ in range(3):
        with experiment.start():
            response = sampler.sample_qubo({(0, 1): 1})
            experiment.store({"result": response})

    evaluator = jb.Evaluator(experiment)
    opt_value = 1.0
    expand = True

    opt_tts = evaluator.optimal_time_to_solution(opt_value=opt_value, expand=expand)
    feas_tts = evaluator.feasible_time_to_solution(expand=expand)
    derived_tts = evaluator.derived_time_to_solution(expand=expand)
    ps = evaluator.success_probability(opt_value=opt_value, expand=expand)
    fr = evaluator.feasible_rate(expand=expand)
    re = evaluator.residual_energy(opt_value=opt_value, expand=expand)

    columns = evaluator.table.columns

    assert "TTS(optimal)" in columns
    assert "TTS(feasible)" in columns
    assert "TTS(derived)" in columns
    assert "success_probability" in columns
    assert "feasible_rate" in columns
    assert "residual_energy" in columns

    assert np.isnan(opt_tts[0])
    assert np.isnan(feas_tts[0])
    assert np.isnan(derived_tts[0])
    assert np.isnan(ps[0])
    assert np.isnan(fr[0])
    assert np.isnan(re[0])


def test_metrics_given_jm_sampleset():
    def solve():
        d = jm.Placeholder("d", dim=2)
        x = jm.Binary("x", shape=d.shape)
        i = jm.Element("i", d.shape[0])
        j = jm.Element("j", d.shape[1])

        problem = jm.Problem("simple_problem")
        problem += jm.Sum([i, j], d[i, j] * x[i, j])
        problem += jm.Constraint("onehot1", x[i, :] == 1, forall=i)
        problem += jm.Constraint("onehot2", x[:, j] == 1, forall=j)

        jm_sampleset_dict = {
            "record": {
                "solution": {
                    "x": [
                        (([0, 1], [0, 1]), [1, 1], (2, 2)),
                        (([0, 1], [1, 0]), [1, 1], (2, 2)),
                        (([], []), [], (2, 2)),
                        (([0, 1], [0, 0]), [1, 1], (2, 2)),
                    ]
                },
                "num_occurrences": [4, 3, 2, 1],
            },
            "evaluation": {
                "energy": [3.0, 24.0, 0.0, 20.0],
                "objective": [3.0, 24.0, 0.0, 17.0],
                "constraint_violations": {
                    "onehot1": [0.0, 0.0, 2.0, 0.0],
                    "onehot2": [0.0, 0.0, 2.0, 2.0],
                },
                "penalty": {},
            },
            "measuring_time": {"solve": None, "system": None, "total": None},
        }
        jm_sampleset = jm.SampleSet.from_serializable(jm_sampleset_dict)
        solving_time = jm.SolvingTime(
            **{"preprocess": 1.0, "solve": 1.0, "postprocess": 1.0}
        )
        jm_sampleset.measuring_time.solve = solving_time
        return jm_sampleset

    experiment = jb.Experiment(autosave=False)
    for _ in range(3):
        with experiment.start():
            sampleset = solve()
            experiment.store({"sampleset": sampleset})

    evaluator = jb.Evaluator(experiment)
    opt_value = 3.0
    expand = False

    opt_tts = evaluator.optimal_time_to_solution(opt_value=opt_value, expand=expand)
    feas_tts = evaluator.feasible_time_to_solution(expand=expand)
    derived_tts = evaluator.derived_time_to_solution(expand=expand)
    ps = evaluator.success_probability(opt_value=opt_value, expand=expand)
    fr = evaluator.feasible_rate(expand=expand)
    re = evaluator.residual_energy(opt_value=opt_value, expand=expand)

    columns = evaluator.table.columns

    assert "TTS(optimal)" not in columns
    assert "TTS(feasible)" not in columns
    assert "TTS(derived)" not in columns
    assert "success_probability" not in columns
    assert "feasible_rate" not in columns
    assert "residual_energy" not in columns

    pr = 0.99
    assert opt_tts[0] == np.log(1 - pr) / np.log(1 - ps[0])
    assert feas_tts[0] == np.log(1 - pr) / np.log(1 - fr[0])
    assert derived_tts[0] == np.log(1 - pr) / np.log(1 - 0.4)
    assert ps[0] == 0.4
    assert fr[0] == 0.7
    assert re[0] == 9.0


def test_evaluate_for_multi_const_problem(bench_for_multi_const_problem):
    # Benchmarkインスタンスのevaluateでテスト
    opt_value = 3.0
    pr = 0.7
    metrics = bench_for_multi_const_problem.evaluate(opt_value=opt_value, pr=pr)

    assert metrics["success_probability"][0] == 0.4
    assert metrics["feasible_rate"][0] == 0.7
    assert metrics["residual_energy"][0] == 9.0
    assert metrics["TTS(optimal)"][0] == np.log(1 - 0.7) / np.log(1 - 0.4)
    assert metrics["TTS(feasible)"][0] == 1.0
    assert metrics["TTS(derived)"][0] == np.log(1 - 0.7) / np.log(1 - 0.4)

    opt_value = 0
    pr = 0.7
    metrics = bench_for_multi_const_problem.evaluate(opt_value=opt_value, pr=pr)

    assert metrics["success_probability"][0] == 0.0
    assert metrics["feasible_rate"][0] == 0.7
    assert metrics["residual_energy"][0] == 12.0
    assert metrics["TTS(optimal)"][0] == np.inf
    assert metrics["TTS(feasible)"][0] == 1.0
    assert metrics["TTS(derived)"][0] == np.log(1 - 0.7) / np.log(1 - 0.4)


def test_warning():
    def solve():
        pass

    opt_value = 0.0
    bench = jb.Benchmark(params={"dummy": [0]}, solver=solve)
    bench.run()

    evaluator = jb.Evaluator(bench)
    evaluator.calc_typical_metrics(opt_value=opt_value)

    evaluator.table.at[0, "objective"] = np.array([0])
    evaluator.calc_typical_metrics(opt_value=opt_value)

    evaluator.table.execution_time = 0
    evaluator.calc_typical_metrics(opt_value=opt_value)

    evaluator.table.at[0, "num_occurrences"] = np.array([0])
    evaluator.calc_typical_metrics(opt_value=opt_value)

    evaluator.table.num_feasible = 0
    evaluator.calc_typical_metrics(opt_value=opt_value, pr=np.nan)


def test_warning_tts():
    def solve(num_reads):
        return np.arange(5, dtype=float), np.array([1]), 2.0, 1

    opt_value = 0.0
    pr = 0.0

    bench = jb.Benchmark(
        params={"num_reads": [1]},
        solver=solve,
        solver_return_name={"solve": ["objective", "num_occurrences", "execution_time", "num_feasible"]},
    )
    bench.run()
    evaluator = jb.Evaluator(bench)

    evaluator.optimal_time_to_solution(opt_value=opt_value, pr=pr)
    evaluator.feasible_time_to_solution(pr=pr)
    evaluator.derived_time_to_solution(pr=pr)
