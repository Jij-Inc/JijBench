import jijmodeling as jm
from .target import JijModelingTarget, DefaultInstanceMixin


def _problem(problem_name):
    w = jm.Placeholder("w", dim=1)
    num_items = jm.Placeholder("num_items")
    c = jm.Placeholder("c")

    # y[j]: bin j を使用するかしないか
    y = jm.Binary("y", shape=(num_items,))
    # x[i][j]: item i を bin j に入れるとき1
    x = jm.Binary("x", shape=(num_items, num_items))

    # i: itemの添字
    i = jm.Element("i", num_items)
    # j: binの添字
    j = jm.Element("j", num_items)

    problem = jm.Problem(problem_name)

    # objective function
    obj = y[:]
    problem += obj

    # Constraint1: 各itemをちょうど1つのbinにぶち込む
    const1 = jm.Constraint("onehot_constraint", jm.Sum(j, x[i, j]) - 1 == 0, forall=i)
    problem += const1

    # Constraint2: knapsack制約
    const2 = jm.Constraint(
        "knapsack_constraint", jm.Sum(i, w[i] * x[i, j]) - y[j] * c <= 0, forall=j
    )
    problem += const2

    return problem


class BinPacking(JijModelingTarget, DefaultInstanceMixin):
    problem_name = "bin_packing"
    problem = _problem(problem_name)

    def __init__(self):
        super().__init__(self.problem, self.small_instance()[0])
