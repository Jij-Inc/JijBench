import jijmodeling as jm


def travelling_salesman():
    # 問題
    problem = jm.Problem('tsp')
    dist = jm.Placeholder("dist", dim=2)
    N = jm.Placeholder("N")
    
    x = jm.Binary("x", shape=(N, N))
    i = jm.Element("i", N)
    j = jm.Element("j", N)

    t = jm.Element("t", N)

    # Objective Funtion
    sum_list = [t, i, j]
    obj = jm.Sum(sum_list, dist[i, j] * x[t, i] * x[(t + 1) % N, j])
    problem += obj

    # const1: onehot for time
    const1 = x[t, :]
    problem += jm.Constraint("onehot_time", const1 == 1, forall=[t, ])

    # const2: onehot for location
    const2 = x[:, i]
    problem += jm.Constraint("onehot_location", const2 == 1, forall=[i, ])

    return problem