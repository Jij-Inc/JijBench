from __future__ import annotations

import jijbench as jb
import jijmodeling as jm
import openjij as oj

from jijbench.problem import TSP, TSPTW, Knapsack
from jijmodeling.transpiler.pyqubo import to_pyqubo


def test_tsptw():
    target = TSPTW()

    assert len(target.small_instance()) > 0
    assert len(target.medium_instance()) > 0

    instance_name = target.instance_names("small")[0]
    ins_data = target.get_instance("small", instance_name)
    assert isinstance(ins_data, dict)


def test_tsptw_time_window_constraint():
    target = TSPTW()
    instance_name = target.instance_names("small")[0]
    instance_data = target.get_instance("small", instance_name)
    print(vars(target))  # あとで消す
    print(instance_data)  # あとで消す

    model,cache = jm.transpiler.pyqubo.to_pyqubo(target,instance_data,{})  # ここが問題となっているので、この行以降確認できていない
    multipliers = {"onehot-constraint1":1.0, "onehot-constraint1":1.0, "time-window-constraint":1.0}
    Q,_ = model.compile().to_qubo(feed_dict = multipliers)

    sampler = oj.SASampler(num_reads=10)
    res = sampler.sample_qubo(Q=Q)
    result = cache.decode(res)

    constraint_violations = result.evaluation.constraint_violations
    print(constraint_violations)  # あとで消す
    assert min(constraint_violations["time-window-constraint"]) == 0


def test_tsp():
    target = TSP()

    assert len(target.small_instance()) > 0
    assert len(target.medium_instance()) > 0

    instance_name = target.instance_names("small")[0]
    ins_data = target.get_instance("small", instance_name)
    assert isinstance(ins_data, dict)


def test_knapsack():
    target = Knapsack()

    assert len(target.small_instance()) > 0
    assert len(target.medium_instance()) > 0

    instance_name = target.instance_names("small")[0]
    ins_data = target.get_instance("small", instance_name)
    assert isinstance(ins_data, dict)


def test_get_default_problem():
    problem = jb.get_problem("TSP")
    assert problem.name == "travelling-salesman"

    problem = jb.get_problem("TSPTW")
    assert problem.name == "travelling-salesman-with-time-windows"

    problem = jb.get_problem("Knapsack")
    assert problem.name == "knapsack"

    problem = jb.get_problem("BinPacking")
    assert problem.name == "bin-packing"

    problem = jb.get_problem("NurseScheduling")
    assert problem.name == "nurse-scheduling"


def test_get_default_instance_data():
    instance_data = jb.get_instance_data("TSP")

    assert isinstance(instance_data[0], tuple)
    assert isinstance(instance_data[0][0], str)
    assert isinstance(instance_data[0][1], dict)

    instance_data = jb.get_instance_data("NurseScheduling")
    assert isinstance(instance_data[0], tuple)
    assert isinstance(instance_data[0][0], str)
    assert isinstance(instance_data[0][1], dict)
