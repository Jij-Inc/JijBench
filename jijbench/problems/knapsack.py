import jijmodeling as jm
from .target import JijModelingTarget, InstanceMixin


def _problem(problem_name):
    w = jm.Placeholder("weights", dim=1)
    v = jm.Placeholder("values", dim=1)
    n = jm.Placeholder("num_items")
    c = jm.Placeholder("capacity")
    x = jm.Binary("x", shape=(n,))

    # i: itemの添字
    i = jm.Element("i", n)

    problem = jm.Problem(problem_name)

    # objective function
    obj = jm.Sum(i, v[i] * x[i])
    problem += -1 * obj

    # Constraint: knapsack 制約
    const = jm.Constraint("knapsack_constraint", jm.Sum(i, w[i] * x[i]) - c <= 0)
    problem += const

    return problem


class Knapsack(JijModelingTarget, InstanceMixin):
    problem_name = "knapsack"
    problem = _problem(problem_name)

    def __init__(self):
        super().__init__(self.problem, self.small_instance()[0])
