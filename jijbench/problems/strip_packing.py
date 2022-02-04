import jijmodeling as jm


def strip_packing():
    """
    :param I_n: 行数
    :param J_n: 列数
    :param K_n: 列内の行数
    :param L_n: アイテム数
    :param W: 素板幅寸法
    :param width: 製品幅寸法
    :param height: 製品高さ寸法
    :return:
    """

    # 問題
    problem = jm.Problem("strip_packing")

    # 定数
    I_n = jm.Placeholder("I_n")  # 行数
    J_n = jm.Placeholder("J_n")  # 列数
    K_n = jm.Placeholder("K_n")  # 行内の行数
    L_n = jm.Placeholder("L_n")  # アイテム数
    W = jm.Placeholder("W")  # 素板幅寸法

    h = jm.Placeholder("h", shape=(L_n,))
    w = jm.Placeholder("w", shape=(L_n,))

    # 決定変数
    x = jm.Binary("x", shape=(I_n, J_n, K_n, L_n, 2))

    # 要素
    i = jm.Element("i", I_n)
    j = jm.Element("j", J_n)
    k = jm.Element("k", K_n)
    l = jm.Element("l", L_n)
    r = jm.Element("r", 2)

    # cost(Hが最小)
    cost = jm.Sum([i, k, l], h[l] * x[i, 0, k, l, 0] + w[l] * x[i, 0, k, l, 1])
    problem += cost

    # const1(アイテムは必ず使う)
    const1 = jm.Constraint(
        "const1", jm.Sum([i, j, k, r], x[i, j, k, l, r]) == 1, forall=l
    )
    problem += const1

    # const2(同じ場所においていいのは一つのアイテムのみ)
    const2 = jm.Constraint(
        "const2", jm.Sum([l, r], x[i, j, k, l, r]) <= 1, forall=[i, j, k]
    )
    problem += const2

    # const3(一番左列の高さが一番高い)
    const3 = jm.Constraint(
        "const3",
        jm.Sum(
            [k, l],
            h[l] * x[i, j, k, l, 0]
            + w[l] * x[i, j, k, l, 1]
            - (h[l] * x[i, 0, k, l, 0] + w[l] * x[i, 0, k, l, 1]),
        )
        <= 0,
        forall=[i, j],
    )
    problem += const3

    # const4(各列の下段が幅広)
    const4 = jm.Constraint(
        "const4",
        jm.Sum(
            l,
            w[l] * x[i, j, k, l, 0]
            + h[l] * x[i, j, k, l, 1]
            - (w[l] * x[i, j, 0, l, 0] + h[l] * x[i, j, 0, l, 1]),
        )
        <= 0,
        forall=[i, j, k],
    )
    problem += const4

    # const5(母材幅Wを超えない(各列の下段の総和))
    const5 = jm.Constraint(
        "const5",
        jm.Sum([j, l], w[l] * x[i, j, 0, l, 0] + h[l] * x[i, j, 0, l, 1]) <= W,
        forall=i,
    )
    problem += const5

    return problem
