import jijmodeling as jm
from jijbench.problems.instance_loader import JijBenchInstance


def nurse_scheduling_instance() -> JijBenchInstance:
    return JijBenchInstance(problem_name="nurse_scheduling")


def nurse_scheduling_problem():
    # 問題
    problem = jm.Problem("nurse_scheduling")

    I = jm.Placeholder("I")  # 人の数
    D = jm.Placeholder("D")  # 日数 D%7=0 とし, 月曜日からstartするとする
    W = jm.Placeholder("W")  # 週の数
    T = jm.Placeholder("T")  # シフトのタイプ数

    N = jm.Placeholder("N", dim=2)  # N[i] = 人iが必ず休む日

    # シフトtの次の日に働ないtypeのシフト R = [type1, type2, type3, ... ] = [[1, 2], [0, ], [], ...]
    R = jm.Placeholder("R", dim=2)
    l = jm.Placeholder("l", shape=(I,))  # シフトtの労働時間
    m_max = jm.Placeholder("m_max", shape=(I, T))  # 従業員にシフトtを割当てられる最大回数
    b_min = jm.Placeholder("b_min", shape=(I,))  # 各従業員の労働時間の最小値
    b_max = jm.Placeholder("b_max", shape=(I,))  # 各従業員の労働時間の最大値
    c_min = jm.Placeholder("c_min", shape=(I,))  # 各従業員の最小連続勤務数
    c_max = jm.Placeholder("c_max", shape=(I,))  # 各従業員の最大連続勤務数
    o_min = jm.Placeholder("o_min", shape=(I,))  # 各従業員の連続最低休日日数
    a_max = jm.Placeholder("a_max", shape=(I,))  # 各従業員の週末働ける最大の回数
    # q[i, d, t] = 1, 人iは日にちdにtype tの仕事をしたい
    q = jm.Placeholder("q", shape=(I, D, T))
    # p[i, d, t] = 1, 人iは日にちdにtype tの仕事をしたくない
    p = jm.Placeholder("p", shape=(I, D, T))
    u = jm.Placeholder("u", shape=(D, T))  # 日にちd, type tの必要人数
    v_min = jm.Placeholder("v_min", shape=(D, T))  # 人員不足のペナルティーの重み
    v_max = jm.Placeholder("v_max", shape=(D, T))  # 人員過剰のペナルティーの重み

    len_R = R.shape[0]

    # Element
    i = jm.Element("i", (0, I))
    d = jm.Element("d", (0, D))
    t = jm.Element("t", (0, T))
    w = jm.Element("w", (0, W))
    dt = jm.Element("dt", (0, len_R))
    j = jm.Element("j", (0, D))
    s = jm.Element("s", (0, D))

    # 決定変数
    x = jm.Binary("x", shape=(I, D, T))  # 人iを日にちdにシフトtを割当てるかどうか
    k = jm.Binary("k", shape=(I, W))  # week wに働いたかどうか
    y = jm.LogEncInteger("y", lower=0, upper=u, shape=(D, T))  # 日dのシフトtの不足人数
    z = jm.LogEncInteger("z", lower=0, upper=u, shape=(D, T))  # 日dのシフトtの過剰人数

    # Objective Function
    term1 = jm.Sum([i, d, t], q[i, d, t] * (1 - x[i, d, t]))  # シフト入りの希望を叶える
    term2 = jm.Sum([i, d, t], p[i, d, t] * x[i, d, t])  # シフト休みの希望を叶える
    term3 = jm.Sum([d, t], y[d, t] * v_min[d, t])  # 人員不足のペナルティー
    term4 = jm.Sum([d, t], z[d, t] * v_max[d, t])  # 人員過剰のペナルティー
    problem += term1 + term2 + term3 + term4

    # Constraint1: 1人一つの仕事しか割り当てられない
    const1 = jm.Sum(t, x[i, d, t])
    problem += jm.Constraint("assign", const1 - 1 <= 0, forall=[i, d])

    # Constraint2: 特定の仕事を行った次の日に, 特定の仕事を行うことはできない
    problem += jm.Constraint(
        "shift_rotation",
        x[i, d, t] + x[i, (d + 1), dt] - 1 <= 0,
        forall=[i, (d, jm.neq(d, D - 1)), t, (dt, jm.neq(R[t, dt], -1))],
    )

    # Constraint3: 従業員に割り当てられる各タイプのシフトの最大数
    const3 = jm.Sum(d, x[i, d, t]) - m_max[i, t]
    problem += jm.Constraint("assign_max", const3 <= 0, forall=[i, t])

    # Constraint4: 最低・最高労働時間
    const4 = jm.Sum([d, t], l[t] * x[i, d, t])
    problem += jm.Constraint(
        "minimum_work_time",
        b_min[i] - const4 <= 0,
        forall=[
            i,
        ],
    )
    problem += jm.Constraint(
        "maximum_work_time",
        const4 - b_max[i] <= 0,
        forall=[
            i,
        ],
    )

    # Constraint5: 最大連続勤務
    const5 = jm.Sum([{j: (d, d + c_max[i] + 1)}, t], x[i, j, t])
    problem += jm.Constraint(
        "maximum_consecutive_shifts",
        const5 - c_max[i] <= 0,
        forall=[i, (d, d <= D - (c_max[i] + 1))],
    )

    # Constraint6: 最小連続勤務
    problem += jm.Constraint(
        "minimum_consecutive_shifts",
        jm.Sum(t, x[i, d, t])
        + (s - jm.Sum([{j: (d + 1, d + s + 1)}, t], x[i, j, t]))
        + jm.Sum(t, x[i, d + s + 1, t])
        - 1
        >= 0,
        forall=[i, {s: (1, c_min[i])}, (d, d < (D - (s + 1)))],
    )

    # Constraint7: 最低連続休暇日数
    problem += jm.Constraint(
        "minimum_consecutive_days_off",
        (1 - jm.Sum(t, x[i, d, t]))
        + jm.Sum([{j: (d + 1, d + s + 1)}, t], x[i, j, t])
        + (1 - jm.Sum(t, x[i, d + s + 1, t]))
        - 1
        >= 0,
        forall=[i, {s: (1, o_min[i])}, (d, d < (D - (s + 1)))],
    )

    # Constraint8: 週末休みの最大回数
    const8 = jm.Sum(t, x[i, 7 * (w + 1) - 2, t]) + jm.Sum(t, x[i, 7 * (w + 1) - 1, t])
    problem += jm.Constraint("variable_k_lower", k[i, w] - const8 <= 0, forall=[i, w])
    problem += jm.Constraint(
        "variable_k_upper", const8 - 2 * k[i, w] <= 0, forall=[i, w]
    )
    problem += jm.Constraint(
        "maximum_number_of_weekends",
        jm.Sum(w, k[i, w]) - a_max[i] <= 0,
        forall=[
            i,
        ],
    )

    # Constraint9: 働けない日の制約
    do = jm.Element("do", (0, D))
    problem += jm.Constraint(
        "days_off", x[i, do, t] == 0, forall=[i, ({do: N[i]}, jm.neq(do, -1)), t]
    )

    # Constraint10: 必要人数に関する制約
    const10 = jm.Sum(i, x[i, d, t]) - z[d, t] + y[d, t] - u[d, t]
    problem += jm.Constraint("cover_requirements", const10 == 0, forall=[d, t])

    return problem
