import pytest

from jijbench.exceptions import SolverFailedError
from jijbench.solver import CallableSolver


def custom_solver_failed():
    raise Exception("solver is failed.")


def test_CallebleSolver_solver_failed_error():
    solver = CallableSolver(custom_solver_failed)
    with pytest.raises(SolverFailedError):
        solver()
