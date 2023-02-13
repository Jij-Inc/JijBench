import jijbench as jb
import jijmodeling as jm
import pytest


@pytest.fixture
def knapsack_problem() -> jm.Problem:
    return jb.get_problem("Knapsack")


@pytest.fixture
def tsp_problem() -> jm.Problem:
    return jb.get_problem("TSP")


@pytest.fixture
def knapsack_instance_data() -> jm.PH_VALUES_INTERFACE:
    return jb.get_instance_data("Knapsack")[0][1]


@pytest.fixture
def tsp_instance_data() -> jm.PH_VALUES_INTERFACE:
    return jb.get_instance_data("TSP")[0][1]


@pytest.fixture
def sampleset_dict() -> dict:
    return {
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


@pytest.fixture
def sampleset(sampleset_dict: dict) -> jm.SampleSet:
    s = jm.SampleSet.from_serializable(sampleset_dict)
    solving_time = jm.SolvingTime(
        **{"preprocess": 1.0, "solve": 1.0, "postprocess": 1.0}
    )
    s.measuring_time.solve = solving_time
    return s
