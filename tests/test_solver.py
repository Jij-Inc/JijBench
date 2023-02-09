import pytest

import jijbench as jb
from jijbench.exceptions.exceptions import SolverFailedError


def func1(x):
    return x


def custom_solver_failed():
    raise Exception("solver is failed.")


def test_simple_solver():
    solver = jb.Solver(func1)

    param = jb.Parameter(1, "x")
    record = solver([param])

    assert isinstance(record, jb.Record)
    assert record.data[0].data == 1
    assert record.operator is None


def test_CallebleSolver_solver_failed_error():
    solver = jb.Solver(custom_solver_failed)
    with pytest.raises(SolverFailedError):
        solver([])
