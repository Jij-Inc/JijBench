import pytest
import jijmodeling as jm

from jijbench.exceptions import SolverFailedError
from jijbench.solver import CallableSolver


def custom_solver_failed(problem, instance_data, multipliers):
    raise Exception("solver is failed.")


def test_CallebleSolver_solver_failed_error():
    solver = CallableSolver(custom_solver_failed)
    problem = jm.Problem("problem")
    with pytest.raises(SolverFailedError):
        solver(problem=problem, instance_data={}, multipliers={})
