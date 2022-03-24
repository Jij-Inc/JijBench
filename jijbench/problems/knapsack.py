import jijmodeling as jm
from jijbench.problems.instance_loader import JijBenchInstance


def knapsack_instance() -> JijBenchInstance:
    return JijBenchInstance(problem_name="knapsack")


def knapsack():
    w = jm.Placeholder("weights", dim=1)
    v = jm.Placeholder("values", dim=1)
    n = jm.Placeholder("num_items")
    c = jm.Placeholder("capacity")
    x = jm.Binary("x", shape=(n,))

    # i: itemの添字
    i = jm.Element("i", n)

    problem = jm.Problem("knapsack")

    # objective function
    obj = jm.Sum(i, v[i] * x[i])
    problem += -1 * obj

    # Constraint: knapsack 制約
    const = jm.Constraint("knapsack_constraint", jm.Sum(i, w[i] * x[i]) - c <= 0)
    problem += const

    return problem
