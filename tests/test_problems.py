from jijbench.problems import TSP, TSPTW, Knapsack


def test_tsptw():
    target = TSPTW()
    
    assert len(target.small_instance()) > 0
    assert len(target.medium_instance()) > 0

    instance_name = target.instance_names("small")[0]
    ins_data = target.get_instance("small", instance_name)
    assert isinstance(ins_data, dict)


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

