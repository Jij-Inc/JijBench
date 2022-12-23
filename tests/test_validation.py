import pytest
from jijbench.benchmark import validation


OBJECT = "OBJECT"
UNSUPPORTED_PROBLEM = "unsupported_problem"
UNSUPPORTED_INSTANCE_DATA = "unsupported_instance_data"
UNSUPPORTED_INSTANCE_DATA_TUPLE = ()
UNSUPPORTED_INSTANCE_DATA_LIST_0 = [[()]]
UNSUPPORTED_INSTANCE_DATA_LIST_1 = [[""]]
UNSUPPORTED_INSTANCE_DATA_LIST_2 = [()]
UNSUPPORTED_INSTANCE_DATA_LIST_3 = [""]


@validation.on_problem
def get_problem_function(obj, problem):
    pass


def test_on_problem_for_unsupported_problem():
    with pytest.raises(TypeError):
        get_problem_function(OBJECT, UNSUPPORTED_PROBLEM)


@validation.on_instance_data
def get_instance_data_function(obj, instance_data):
    pass


def test_on_instance_data_for_unsupported_instance_data():
    with pytest.raises(TypeError):
        get_instance_data_function(OBJECT, UNSUPPORTED_INSTANCE_DATA)


def test_tuple_to_instance_data_for_unsupported_instance_data_tuple():
    with pytest.raises(TypeError):
        validation._tuple_to_instance_data(UNSUPPORTED_INSTANCE_DATA_TUPLE)


params = {
    "the first is list with tuple": UNSUPPORTED_INSTANCE_DATA_LIST_0,
    "the first is list with except tuple": UNSUPPORTED_INSTANCE_DATA_LIST_1,
    "the first is tuple": UNSUPPORTED_INSTANCE_DATA_LIST_2,
    "the first is others": UNSUPPORTED_INSTANCE_DATA_LIST_3,
}


@pytest.mark.parametrize(
    "instance_data",
    list(params.values()),
    ids=list(params.keys()),
)
def test_list_to_instance_data_for_unsupported_instance_data_list(instance_data):
    with pytest.raises(TypeError):
        validation._list_to_instance_data(instance_data)
