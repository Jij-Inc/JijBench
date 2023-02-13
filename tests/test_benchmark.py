import os, shutil, time

import jijmodeling as jm
import jijzept as jz
import numpy as np
import pandas as pd
import pytest

import jijbench as jb
from jijbench.exceptions.exceptions import SolverFailedError, ConcurrentFailedError
from pytest_mock import MockerFixture


@pytest.fixture(scope="function", autouse=True)
def pre_post_process():
    # preprocess
    yield
    # postprocess
    norm_path = os.path.normcase("./.jb_results")
    if os.path.exists(norm_path):
        shutil.rmtree(norm_path)


def generate_problem():
    d = jm.Placeholder("d", dim=1)
    x = jm.Binary("x", shape=(d.shape[0].set_latex("n")))
    i = jm.Element("i", d.shape[0])
    problem = jm.Problem("problem")
    problem += jm.Sum(i, d[i] * x[i])
    problem += jm.Constraint("onehot1", jm.Sum(i, x[i]) == 1)
    return problem


# def test_set_problem_in_benchmark(problem, problem_list):
#     bench = jb.Benchmark({"dummy": [1]}, solver="JijSASampler", problem=problem)
#
#     assert isinstance(bench.problem, list)
#     assert isinstance(bench.problem[0], jm.Problem)
#
#     bench = jb.Benchmark({"dummy": [1]}, solver="JijSASampler", problem=problem_list)
#
#     assert isinstance(bench.problem, list)
#     assert isinstance(bench.problem[0], jm.Problem)
#
#
def test_set_instance_data_in_benchmark(ph_value):
    # PH_VALUES_INTERFACE
    instance_data = jb.InstanceData(ph_value, "instance_data")
    bench = jb.Benchmark({"instance_data": [instance_data]}, solver=lambda: ())

    assert instance_data in bench.params[0]


def test_benchmark_params():
    bench = jb.Benchmark(
        {
            "num_reads": [1, 2],
            "num_sweeps": [10],
        },
        solver=sample_model,
    )
    print()
    print(bench.params)


def test_simple_benchmark():
    def func(x):
        return x

    bench = jb.Benchmark({"x": [1, 2]}, solver=func, name="test")

    res = bench()
    columns = res.table.columns

    # ic()
    # ic(res.data[1].data)
    # ic(res.data[0].data)

    assert isinstance(res, jb.Experiment)
    assert "func_return[0]" in columns

    op1 = res.operator
    assert op1 is not None
    assert isinstance(op1.inputs[0], jb.Experiment)
    assert isinstance(op1.inputs[1], jb.Experiment)
    t1 = op1.inputs[0].table
    t2 = op1.inputs[1].table

    assert t1.iloc[0, 0] == 1
    assert t2.iloc[0, 0] == 2


def test_benchmark_jijzept_sampler(problem, ph_value):
    bench = jb.construct_benchmark_for_jijzept_sampler(
        {"num_reads": [5]},
    )

    res = bench()

    # print(res.params_table)


def test_apply_benchmark():
    def func(x):
        return x

    bench = jb.Benchmark(
        {"x": [1, 2]},
        solver=func,
    )

    experiment = jb.Experiment(name=jb.ID().data)
    res = experiment.apply(bench)
    columns = res.table.columns

    assert isinstance(res, jb.Experiment)
    assert "func_return[0]" in columns

    op1 = res.operator
    # ic()
    # ic(op1.inputs)
    assert op1 is not None
    assert isinstance(op1, jb.Benchmark)
    assert isinstance(op1.inputs[0], jb.Experiment)
    assert len(op1.inputs) == 1
    assert op1.inputs[0].table.empty


def test_benchmark_params_table():
    def func(x):
        return x

    bench = jb.Benchmark(
        {"x": [1, 2]},
        solver=func,
    )

    res = bench()


def test_benchmark_with_multi_return_solver():
    def func():
        return "a", 1

    bench = jb.Benchmark({"num_reads": [1, 2], "num_sweeps": [10]}, solver=func)
    res = bench()

    # assert res.table["solver"][0] == func.__name__
    assert res.table["func_return[0]"][0] == "a"
    assert res.table["func_return[1]"][0] == 1.0


# def test_benchmark_with_custom_solver_by_sync_False():
#     def func():
#         return "a", 1
#
#     bench = jb.Benchmark({"num_reads": [1, 2], "num_sweeps": [10]}, solver=func)
#     with pytest.raises(ConcurrentFailedError):
#         bench.run(sync=False)


def test_benchmark_with_custom_sample_model(
    problem,
    ph_value,
    ph_value_list,
    instance_data,
    instance_data_list,
):
    for d in [ph_value, ph_value_list, instance_data, instance_data_list][:1]:
        bench = jb.Benchmark(
            {
                "num_reads": [1, 2],
                "num_sweeps": [10],
                "multipliers": [{"knapsack_constraint": 1}],
                "problem": [problem],
                "instance_data": [d],
            },
            solver=sample_model,
        )
        res = bench()
        columns = res.table.columns

        print()
        print(res.data[1])
        print(res.table)
        print(res.operator)

        # assert res.table["solver"].iloc[0] == sample_model.__name__
        assert "sample_molel_return[0]" not in columns
        # assert res.table["problem_name"].iloc[0] == "knapsack"


def test_benchmark_with_custom_sample_model_for_multi_problem(
    problem_list, multi_ph_value_list, multi_instance_data_list
):
    for d in [multi_ph_value_list, multi_instance_data_list]:
        bench = jb.Benchmark(
            {
                "num_reads": [1, 2],
                "num_sweeps": [10],
                "problem": [problem_list],
                "instance_data": [d],
            },
            solver=sample_model,
        )
        res = bench()
        columns = res.table.columns

        # assert res.table["solver"].iloc[0] == sample_model.__name__
        assert "sample_model_return[0]" not in columns
        # assert res.table["problem_name"].iloc[0] == "knapsack"


def test_benchmark_with_any_problem_and_instance_data():
    problem = generate_problem()
    instance_data = {"d": [1 for _ in range(10)]}
    instance_data["d"][0] = -1

    bench = jb.Benchmark(
        {
            "num_reads": [1, 2],
            "num_sweeps": [10],
            "problem": [problem],
            "instance_data": [instance_data],
        },
        solver=sample_qubo,
    )
    res = bench()

    # assert res.table["problem_name"][0] == "problem"
    # assert res.table["instance_data_name"][0] == "Unnamed[0]"


def test_benchmark_with_callable_args():
    def rap_solver(N, sample_model):
        d_p = [1 for _ in range(N)]
        d_p[0] = -1
        s = time.time()
        dimod_sampleset = sample_qubo()
        dimod_sampleset.info["total_time"] = time.time() - s
        return dimod_sampleset, dimod_sampleset.info["total_time"]

    bench = jb.Benchmark(
        {
            "N": [10, 200],
            "sample_model": [sample_model],
        },
        solver=rap_solver,
    )

    res = bench()

    columns = res.table.columns
    # assert sample_model.__name__ in columns
    # assert isinstance(res.table[sample_model.__name__][0], str)


# def test_benchmark_with_multisolver():
#     def func1(x):
#         return 2 * x
#
#     def func2(x):
#         return 3 * x
#
#     bench = jb.Benchmark(params={"x": [1, 2, 3]}, solver=[func1, func2])
#     bench.run()
#
#     columns = bench.table.columns
#
#     assert "solver" in columns
#     assert "func1" in bench.table["solver"].values
#     assert "func2" in bench.table["solver"].values
#
#
# def test_load():
#     def func1(x):
#         return 2 * x
#
#     bench = jb.Benchmark(params={"x": [1, 2, 3]}, solver=func1, benchmark_id="test")
#     bench.run()
#
#     del bench
#
#     bench = jb.load(benchmark_id="test")
#
#     assert "func1" in bench.table["solver"].values
#
#
# def test_save():
#     def func1(x):
#         return 2 * x
#
#     import pathlib
#
#     save_dir = str(pathlib.PurePath(__file__).parent / ".my_result")
#
#     bench = jb.Benchmark(
#         params={"x": [1, 2, 3]}, solver=func1, benchmark_id="test", save_dir=save_dir
#     )
#     bench.run()
#
#     shutil.rmtree(save_dir)
#
#
# def test_benchmark_for_custom_solver_return_jm_sampleset():
#     def func():
#         jm_sampleset = jm.SampleSet.from_serializable(
#             {
#                 "record": {
#                     "solution": {
#                         "x": [
#                             (([0, 1], [0, 1]), [1, 1], (2, 2)),
#                             (([], []), [], (2, 2)),
#                         ]
#                     },
#                     "num_occurrences": [1, 1],
#                 },
#                 "evaluation": {
#                     "energy": [
#                         -3.8499999046325684,
#                         0.0,
#                     ],
#                     "objective": [3.0, 0.0],
#                     "constraint_violations": {},
#                     "penalty": None,
#                 },
#                 "measuring_time": {
#                     "solve": None,
#                     "system": None,
#                     "total": None,
#                 },
#             }
#         )
#         jm_sampleset.measuring_time.solve.solve = None
#         jm_sampleset.measuring_time.system.system = None
#         jm_sampleset.measuring_time.total = None
#         return jm_sampleset
#
#     bench = jb.Benchmark(params={"dummy": [1]}, solver=func)
#     bench.run()
#
#
# def test_benchmark_for_custom_solver_failed():
#     def custom_solver_failed():
#         raise Exception("solver is failed.")
#
#     bench = jb.Benchmark(params={"dummy": [1]}, solver=custom_solver_failed)
#     with pytest.raises(SolverFailedError):
#         bench.run()
#
#
# def test_benchmark_for_num_feasible():
#     bench = jb.Benchmark(
#         {
#             "N": [10, 200],
#             "sample_model": [sample_model],
#         },
#         solver=sample_model,
#     )
#     bench.run()
#     assert (bench.table["num_feasible"].values == 7).all()
#
#
# def test_benchmark_for_change_solver_return_name():
#     def solver():
#         return 1
#
#     bench = jb.Benchmark(
#         {
#             "N": [10, 200],
#             "sample_model": [sample_model],
#         },
#         solver=solver,
#         solver_return_name={"solver": ["return_1"]},
#     )
#     bench.run()
#     assert "return_1" in bench.table.columns
