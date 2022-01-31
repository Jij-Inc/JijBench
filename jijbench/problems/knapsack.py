import jijmodeling as jm


def knapsack():
    w = jm.Placeholder("weights", dim=1)
    v = jm.Placeholder("values", dim=1)
    n = jm.Placeholder("num_items")
    c = jm.Placeholder("capacity")
    x = jm.Binary("x", shape=(n, ))

    # i: itemの添字
    i = jm.Element("i", n)

    problem = jm.Problem("knapsack packing")

    # objective function
    obj = jm.Sum(i, v[i] * x[i])
    problem += -1 * obj

    # Constraint: knapsack 制約
    const = jm.Constraint("knapsack_constraint", jm.Sum(i, w[i] * x[i]) - c <= 0)
    problem += const

    return problem
