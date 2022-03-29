import os
import shutil
import jijbench as jb
import pytest


@pytest.fixture(scope="function", autouse=True)
def pre_post_process():
    # preprocess
    yield
    # postprocess
    if os.path.exists("./.jb_results"):
        shutil.rmtree("./.jb_results")
        pass


def test_simple_benchmark():
    bench = jb.Benchmark(
        {"num_reads": [1, 2], "num_sweeps": [10]},
        solvers="SASampler",
        targets="Knapsack",
    )
    bench.run()

    columns = bench.table.columns

    assert "problem_name" in columns
    assert "instance_name" in columns
    assert "solver" in columns


def test_custom_solver_benchmark():
    import jijbench as jb

    def func():
        return "a", 1

    bench = jb.Benchmark(
        {"num_reads": [1, 2], "num_sweeps": [10]}, solvers=func, targets="Knapsack"
    )
    bench.run()

    assert bench.table["solver"][0] == func.__name__
    assert bench.table["solver_return_values[0]"][0] == "a"
    assert bench.table["solver_return_values[1]"][0] == 1.0
