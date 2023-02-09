import pytest

import jijbench as jb
import jijmodeling as jm
import numpy as np
import pytest

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


def test_instance_data():
    instance_data = jb.InstanceData(
        {
            "int": 1,
            "float": 1.0,
            "list": [1, 2.0],
            "ndarray": np.array([[1, 2], [3, 4]]),
        },
        "sample"
    )
    
    assert instance_data.data["int"] == 1
    assert instance_data.data["float"] == 1.0
    assert instance_data.data["list"] == [1, 2.0]
    assert isinstance(instance_data.data["ndarray"], np.ndarray)

def test_invalid_instance_data_keys():
    with pytest.raises(TypeError):
        instance_data = jb.InstanceData(
            {
                "int": 1,
                "float": 1.0,
                0: [1, 2.0],
                1: np.array([[1, 2], [3, 4]]),
            },
            "sample"
        )
    
def test_invalid_instance_data_values():
    with pytest.raises(TypeError):
        instance_data = jb.InstanceData(
            {
                "int": "a",
                "float": {"b": 1},
                "list": [1, 2.0],
                "ndarray": np.array([[1, 2], [3, 4]]),
            },
            "sample"
        )

def test_user_defined_model():
    problem = jm.Problem("sample")
    a = jm.Placeholder("a")
    b = jm.Placeholder("b")
    c = jm.Placeholder("c", 1)
    i = jm.Element("i", 5)
    x = jm.DecisionVariable("x", 5)
    
    problem = jm.Problem("sample")
    problem += a + b + jm.Sum(i, c[i] * x[i])
    problem
    
    
    