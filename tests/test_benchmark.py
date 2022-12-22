import os, shutil, time

import dimod
import jijmodeling as jm
import numpy as np
import pytest

import jijbench as jb
from jijbench.exceptions import (
    SolverFailedError,
    ConcurrentFailedError,
    LoadFailedError,
)


@pytest.fixture(scope="function", autouse=True)
def pre_post_process():
    # preprocess
    yield
    # postprocess
    norm_path = os.path.normcase("./.jb_results")
    if os.path.exists(norm_path):
        shutil.rmtree(norm_path)


@pytest.fixture
def problem():
    return jb.get_problem("Knapsack")


@pytest.fixture
def problem_list():
    return [jb.get_problem("Knapsack"), jb.get_problem("TSP")]


@pytest.fixture
def instance_data():
    return jb.get_instance_data("Knapsack")[0]


@pytest.fixture
def instance_data_list():
    return jb.get_instance_data("Knapsack")


@pytest.fixture
def multi_instance_data_list():
    return [jb.get_instance_data("Knapsack")[0:2], jb.get_instance_data("TSP")[0:1]]


@pytest.fixture
def ph_value():
    return jb.get_instance_data("Knapsack")[0][1]


@pytest.fixture
def ph_value_list():
    instance_data = jb.get_instance_data("Knapsack")
    return [instance_data[0][1], instance_data[1][1]]


@pytest.fixture
def multi_ph_value_list():
    knapsack_instance_data = jb.get_instance_data("Knapsack")
    tsp_instance_data = jb.get_instance_data("TSP")
    return [
        [knapsack_instance_data[0][1], knapsack_instance_data[1][1]],
        [tsp_instance_data[0][1]],
    ]


def generate_problem():
    d = jm.Placeholder("d", dim=1)
    x = jm.Binary("x", shape=(d.shape[0].set_latex("n")))
    i = jm.Element("i", d.shape[0])
    problem = jm.Problem("problem")
    problem += jm.Sum(i, d[i] * x[i])
    problem += jm.Constraint("onehot1", jm.Sum(i, x[i]) == 1)
    return problem


def sample_qubo():
    dimod_sampleset = dimod.SampleSet.from_samples(
        samples_like=[
            {"x[0][0]": 1, "x[0][1]": 0, "x[1][0]": 0, "x[1][1]": 1},  # 最適解
            {
                "x[0][0]": 0,
                "x[0][1]": 1,
                "x[1][0]": 1,
                "x[1][1]": 0,
            },  # 実行可能解だけど最適解ではない
            {
                "x[0][0]": 0,
                "x[0][1]": 0,
                "x[1][0]": 0,
                "x[1][1]": 0,
            },  # 実行不可能解、目的関数値 < 最適値
            {
                "x[0][0]": 1,
                "x[0][1]": 0,
                "x[1][0]": 1,
                "x[1][1]": 0,
            },  # 制約onehot1だけ満たす
        ],
        vartype="BINARY",
        energy=[3, 24, 0, 20],
        num_occurrences=[4, 3, 2, 1],
    )
    dimod_sampleset.info["execution_time"] = 1.0
    return dimod_sampleset


def sample_model():
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


def test_set_problem_in_benchmark(problem, problem_list):
    bench = jb.Benchmark({"dummy": [1]}, solver="JijSASampler", problem=problem)

    assert isinstance(bench.problem, list)
    assert isinstance(bench.problem[0], jm.Problem)

    bench = jb.Benchmark({"dummy": [1]}, solver="JijSASampler", problem=problem_list)

    assert isinstance(bench.problem, list)
    assert isinstance(bench.problem[0], jm.Problem)


def test_set_instance_data_in_benchmark(
    ph_value,
    ph_value_list,
    multi_ph_value_list,
    instance_data,
    instance_data_list,
    multi_instance_data_list,
):
    # PH_VALUES_INTERFACE
    bench = jb.Benchmark({"dummy": [1]}, solver="JijSASampler", instance_data=ph_value)
    assert len(bench.instance_data) == 1
    assert isinstance(bench.instance_data[0], list)
    assert isinstance(bench.instance_data[0][0], tuple)

    # List[PH_VALUES_INTERFACE]
    bench = jb.Benchmark(
        {"dummy": [1]}, solver="JijSASampler", instance_data=ph_value_list
    )
    assert len(bench.instance_data) == 1
    assert isinstance(bench.instance_data[0], list)
    assert isinstance(bench.instance_data[0][0], tuple)

    # List[List[PH_VALUES_INTERFACE]]
    bench = jb.Benchmark(
        {"dummy": [1]}, solver="JijSASampler", instance_data=multi_ph_value_list
    )
    assert len(bench.instance_data) == 2
    assert isinstance(bench.instance_data[0], list)
    assert isinstance(bench.instance_data[0][0], tuple)

    # Tuple[str, PH_VALUES_INTERFACE]
    bench = jb.Benchmark(
        {"dummy": [1]}, solver="JijSASampler", instance_data=instance_data
    )
    assert len(bench.instance_data) == 1
    assert isinstance(bench.instance_data[0], list)
    assert isinstance(bench.instance_data[0][0], tuple)

    # List[Tuple[str, PH_VALUES_INTERFACE]]
    bench = jb.Benchmark(
        {"dummy": [1]}, solver="JijSASampler", instance_data=instance_data_list
    )

    assert len(bench.instance_data) == 1
    assert isinstance(bench.instance_data[0], list)
    assert isinstance(bench.instance_data[0][0], tuple)

    # List[List[Tuple[str, PH_VALUES_INTERFACE]]]
    bench = jb.Benchmark(
        {"dummy": [1]},
        solver="JijSASampler",
        instance_data=multi_instance_data_list,
    )
    assert len(bench.instance_data) == 2
    assert isinstance(bench.instance_data[0], list)
    assert isinstance(bench.instance_data[0][0], tuple)


def test_simple_benchmark(problem, instance_data):
    bench = jb.Benchmark(
        {"num_reads": [1, 2], "num_sweeps": [10]},
        solver=sample_qubo,
        problem=problem,
        instance_data=instance_data,
    )
    bench.run()

    columns = bench.table.columns

    assert "problem_name" in columns
    assert "instance_data_name" in columns
    assert "solver" in columns


def test_benchmark_with_custom_solver():
    def func():
        return "a", 1

    bench = jb.Benchmark({"num_reads": [1, 2], "num_sweeps": [10]}, solver=func)
    bench.run()

    assert bench.table["solver"][0] == func.__name__
    assert bench.table["solver_return_values[0]"][0] == "a"
    assert bench.table["solver_return_values[1]"][0] == 1.0


def test_benchmark_with_custom_solver_by_sync_False():
    def func():
        return "a", 1

    bench = jb.Benchmark({"num_reads": [1, 2], "num_sweeps": [10]}, solver=func)
    with pytest.raises(ConcurrentFailedError):
        bench.run(sync=False)


def test_benchmark_with_custom_sample_model(
    problem,
    ph_value,
    ph_value_list,
    instance_data,
    instance_data_list,
):
    for d in [ph_value, ph_value_list, instance_data, instance_data_list]:
        bench = jb.Benchmark(
            {
                "num_reads": [1, 2],
                "num_sweeps": [10],
                "multipliers": [{"knapsack_constraint": 1}],
            },
            solver=sample_model,
            problem=problem,
            instance_data=d,
        )
        bench.run()
        columns = bench.table.columns

        assert bench.table["solver"].iloc[0] == sample_model.__name__
        assert "solver_return_values[0]" not in columns
        assert bench.table["problem_name"].iloc[0] == "knapsack"


def test_benchmark_with_custom_sample_model_for_multi_problem(
    problem_list, multi_ph_value_list, multi_instance_data_list
):
    for d in [multi_ph_value_list, multi_instance_data_list]:
        bench = jb.Benchmark(
            {
                "num_reads": [1, 2],
                "num_sweeps": [10],
            },
            solver=sample_model,
            problem=problem_list,
            instance_data=d,
        )
        bench.run()
        columns = bench.table.columns

        assert bench.table["solver"].iloc[0] == sample_model.__name__
        assert "solver_return_values[0]" not in columns
        assert bench.table["problem_name"].iloc[0] == "knapsack"


def test_benchmark_with_custom_decode(
    problem,
    ph_value,
    ph_value_list,
    instance_data,
    instance_data_list,
):
    for d in [ph_value, ph_value_list, instance_data, instance_data_list]:
        bench = jb.Benchmark(
            {
                "num_reads": [1, 2],
                "num_sweeps": [10],
                "multipliers": [{"knapsack_constraint": 1}],
            },
            solver=sample_model,
            problem=problem,
            instance_data=d,
        )
        bench.run()
        columns = bench.table.columns

        assert bench.table["solver"].iloc[0] == sample_model.__name__
        assert "solver_return_values[0]" not in columns
        assert bench.table["problem_name"].iloc[0] == "knapsack"
        assert isinstance(bench.table["objective"].iloc[0], np.ndarray)


def test_benchmark_with_any_problem_and_instance_data():
    problem = generate_problem()
    instance_data = {"d": [1 for _ in range(10)]}
    instance_data["d"][0] = -1

    bench = jb.Benchmark(
        {"num_reads": [1, 2], "num_sweeps": [10]},
        solver=sample_qubo,
        problem=problem,
        instance_data=instance_data,
    )
    bench.run()

    assert bench.table["problem_name"][0] == "problem"
    assert bench.table["instance_data_name"][0] == "Unnamed[0]"


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

    bench.run()

    columns = bench.table.columns
    assert sample_model.__name__ in columns
    assert isinstance(bench.table[sample_model.__name__][0], str)


def test_benchmark_with_multisolver():
    def func1(x):
        return 2 * x

    def func2(x):
        return 3 * x

    bench = jb.Benchmark(params={"x": [1, 2, 3]}, solver=[func1, func2])
    bench.run()

    columns = bench.table.columns

    assert "solver" in columns
    assert "func1" in bench.table["solver"].values
    assert "func2" in bench.table["solver"].values


def test_load():
    def func1(x):
        return 2 * x

    bench = jb.Benchmark(params={"x": [1, 2, 3]}, solver=func1, benchmark_id="test")
    bench.run()

    del bench

    bench = jb.load(benchmark_id="test")

    assert "func1" in bench.table["solver"].values


def test_load_invalid_benchmark_id():
    INVALID_BENCHMARK_ID = "invalid_benchmark_id"

    def func1(x):
        return 2 * x

    bench = jb.Benchmark(params={"x": [1, 2, 3]}, solver=func1, benchmark_id="test")
    bench.run()

    del bench

    with pytest.raises(LoadFailedError):
        bench = jb.load(benchmark_id=INVALID_BENCHMARK_ID)


def test_load_invalid_experiment_id():
    INVALID_EXPERIMENT_ID = ["invalid_experiment_id"]

    def func1(x):
        return 2 * x

    bench = jb.Benchmark(params={"x": [1, 2, 3]}, solver=func1, benchmark_id="test")
    bench.run()

    del bench

    with pytest.raises(LoadFailedError):
        bench = jb.load(benchmark_id="test", experiment_id=INVALID_EXPERIMENT_ID)


def test_get_experiment_id_list():
    from jijbench.benchmark.benchmark import get_experiment_id_list
    from jijbench.components import ExperimentResultDefaultDir

    def func1(x):
        return 2 * x

    bench = jb.Benchmark(params={"x": [1, 2, 3]}, solver=func1, benchmark_id="test")
    bench.run()
    experiment_id_list = list(bench.table["experiment_id"].unique())

    experiment_id_list_load = get_experiment_id_list("test", ExperimentResultDefaultDir)

    assert sorted(experiment_id_list) == sorted(experiment_id_list_load)


def test_save():
    def func1(x):
        return 2 * x

    import pathlib

    save_dir = str(pathlib.PurePath(__file__).parent / ".my_result")

    bench = jb.Benchmark(
        params={"x": [1, 2, 3]}, solver=func1, benchmark_id="test", save_dir=save_dir
    )
    bench.run()

    shutil.rmtree(save_dir)


def test_benchmark_for_custom_solver_return_jm_sampleset():
    def func():
        jm_sampleset = jm.SampleSet.from_serializable(
            {
                "record": {
                    "solution": {
                        "x": [
                            (([0, 1], [0, 1]), [1, 1], (2, 2)),
                            (([], []), [], (2, 2)),
                        ]
                    },
                    "num_occurrences": [1, 1],
                },
                "evaluation": {
                    "energy": [
                        -3.8499999046325684,
                        0.0,
                    ],
                    "objective": [3.0, 0.0],
                    "constraint_violations": {},
                    "penalty": None,
                },
                "measuring_time": {
                    "solve": None,
                    "system": None,
                    "total": None,
                },
            }
        )
        jm_sampleset.measuring_time.solve.solve = None
        jm_sampleset.measuring_time.system.system = None
        jm_sampleset.measuring_time.total = None
        return jm_sampleset

    bench = jb.Benchmark(params={"dummy": [1]}, solver=func)
    bench.run()


def test_benchmark_for_custom_solver_failed():
    def custom_solver_failed():
        raise Exception("solver is failed.")

    bench = jb.Benchmark(params={"dummy": [1]}, solver=custom_solver_failed)
    with pytest.raises(SolverFailedError):
        bench.run()


def test_benchmark_for_num_feasible():
    bench = jb.Benchmark(
        {
            "N": [10, 200],
            "sample_model": [sample_model],
        },
        solver=sample_model,
    )
    bench.run()
    assert (bench.table["num_feasible"].values == 7).all()


def test_benchmark_for_change_solver_return_name():
    def solver():
        return 1

    bench = jb.Benchmark(
        {
            "N": [10, 200],
            "sample_model": [sample_model],
        },
        solver=solver,
        solver_return_name={"solver": ["return_1"]},
    )
    bench.run()
    assert "return_1" in bench.table.columns
