from __future__ import annotations

import jijbench as jb

from jijbench.datasets import TSP, TSPTW, Knapsack


def test_tsptw():
    target = TSPTW()

    assert len(target.small_instance()) > 0
    assert len(target.medium_instance()) > 0

    instance_name = target.instance_names("small")[0]
    ins_data = target.get_instance("small", instance_name)
    assert isinstance(ins_data, dict)


def test_tsptw_time_window_constraint():
    problem = jb.get_problem("TSPTW")
    print(str(problem.constraints["time-window-constraint"]))
    assert "1*20*1 - 1*x[i,j]" in str(problem.constraints["time-window-constraint"])


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
