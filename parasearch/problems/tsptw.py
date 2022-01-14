import jijmodeling as jm


def travelling_salesman_with_time_windows():
    # 問題
    problem = jm.Problem('tsptw')

    # 距離行列
    dist = jm.Placeholder("dist", dim=2)  # 距離行列
    N = jm.Placeholder("N")
    e = jm.Placeholder("e", shape=(N, ))  # ready time
    l = jm.Placeholder("l", shape=(N, ))  # due time
    x = jm.Binary("x", shape=(N, N))
    t = jm.LogEncIntArray("t", shape=(N, ), lower=e, upper=l)

    i = jm.Element("i", N)
    j = jm.Element("j", N)

    # Objevtive Function: 距離の最小化
    sum_list = [i, (j, j != i)]
    obj = jm.Sum(sum_list, dist[i, j] * x[i, j])
    problem += obj

    # Const1: 都市iから出る辺は1つ
    term1 = jm.Sum((j, j != i), x[i, j])
    problem += jm.Constraint("onehot_constraint1", term1 == 1, forall=[i, ])

    # Const2: 都市iに入る辺は1つ
    term2 = jm.Sum((j, j != i), x[j, i])
    problem += jm.Constraint("onehot_constraint2", term2 == 1, forall=[i, ])

    # Const3: Time Windows制約
    term3 = t[i] + dist[i, j] - t[j]
    forall_list = [(j, j != 0), (i, (i != 0) & (i != j))]
    problem += jm.Constraint("time_window_constraint", term3 <= 0, forall=forall_list)

    return problem
