from gettext import install
from jijbench.problems.tsptw import (
    travelling_salesman_with_time_windows,
    tsptw_instance,
)
from jijbench.problems.tsp import travelling_salesman, tsp_instance
from jijbench.problems.knapsack import knapsack, knapsack_instance


def test_tsptw():
    instance = tsptw_instance()
    small_list = instance.small_list()
    assert len(small_list) > 0

    medium_list = instance.medium_list()
    assert len(medium_list) > 0

    ins_data = instance.get_instance("small", small_list[0])
    assert isinstance(ins_data, dict)


def test_tsp():
    tsp_ins = tsp_instance()
    small_list = tsp_ins.small_list()
    assert len(small_list) > 0

    medium_list = tsp_ins.medium_list()
    assert len(medium_list) > 0

    ins_data = tsp_ins.get_instance("small", small_list[0])
    assert isinstance(ins_data, dict)


def test_knapsack():
    instance = knapsack_instance()
    small_list = instance.small_list()
    assert len(small_list) > 0

    medium_list = instance.medium_list()
    assert len(medium_list) > 0

    ins_data = instance.get_instance("small", small_list[0])
    assert isinstance(ins_data, dict)
