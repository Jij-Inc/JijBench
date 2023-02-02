import pytest

from jijbench.exceptions import SolverFailedError
from jijbench.solver import CallableSolver


OBJECT = "OBJECT"
UNSUPPORTED_SOLVER = 0


def custom_solver_failed():
    raise Exception("solver is failed.")


def test_simple_solver():
    solver = jb.functions.Solver(func1)

    param = jb.Parameter(1, "x")
    record = solver([param])

    assert isinstance(record, jb.Record)
    assert record.data[0].data == 1
    assert record.operator is None


def test_CallebleSolver_solver_failed_error():
    solver = CallableSolver(custom_solver_failed)
    with pytest.raises(SolverFailedError):
        solver()


def test_solver_for_unsupported_solver():
    with pytest.raises(TypeError):
        CallableSolver(UNSUPPORTED_SOLVER)
