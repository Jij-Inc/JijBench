import pytest

import jijbench as jb
from jijbench.exceptions.exceptions import SolverFailedError


def func1(x):
    return x


def custom_solver_failed():
    raise Exception("solver is failed.")


def test_simple_solver():
    solver = jb.functions.Solver(func1)

    param = jb.Parameter(1, "x")
    ret = solver([param])
    
    print()
    print(ret.data)
    print(ret.operator)

    assert isinstance(ret, jb.Record)
    assert ret.operator is not None
    assert ret.operator.inputs[0].data == 1


def test_CallebleSolver_solver_failed_error():
    solver = jb.functions.Solver(custom_solver_failed)
    with pytest.raises(SolverFailedError):
        solver([])
