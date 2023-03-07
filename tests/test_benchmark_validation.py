# import pytest
# import jijmodeling as jm
# 
# from jijbench.benchmark import validation
# 
# OBJECT = "OBJECT"
# 
# UNSUPPORTED_PROBLEM = "unsupported_problem"
# 
# UNSUPPORTED_INSTANCE_DATA = "unsupported_instance_data"
# UNSUPPORTED_INSTANCE_DATA_TUPLE = ()
# UNSUPPORTED_INSTANCE_DATA_LIST_0 = [[()]]
# UNSUPPORTED_INSTANCE_DATA_LIST_1 = [[""]]
# UNSUPPORTED_INSTANCE_DATA_LIST_2 = [()]
# UNSUPPORTED_INSTANCE_DATA_LIST_3 = [""]
# 
# 
# class GetSolver:
#     def __init__(self):
#         self.solver = None
# 
#     @validation.on_solver
#     def receive_solver(self, solver):
#         self.solver = solver
# 
# 
# def callable_function():
#     pass
# 
# 
# params = {
#     "list_case": ([callable_function], list),
#     "tuple_case": ((callable_function,), tuple),
#     "callable_case": (callable_function, list),
# }
# 
# 
# @pytest.mark.parametrize(
#     "solver, expect_type", list(params.values()), ids=list(params.keys())
# )
# def test_on_solver(solver, expect_type):
#     obj = GetSolver()
#     obj.receive_solver(solver)
#     assert type(obj.solver) == expect_type
# 
# 
# class GetProblem:
#     def __init__(self):
#         self.problem = None
# 
#     @validation.on_problem
#     def receive_problem(self, problem):
#         self.problem = problem
# 
# 
# params = {
#     "problem": (jm.Problem("test"), list),
#     "problem_list": ([jm.Problem("test")], list),
#     "problem_tuple": ((jm.Problem("test"),), tuple),
#     "problem_None": (None, type(None)),
# }
# 
# 
# @pytest.mark.parametrize(
#     "problem, expect_type", list(params.values()), ids=list(params.keys())
# )
# def test_on_problem(problem, expect_type):
#     obj = GetProblem()
#     obj.receive_problem(problem)
#     assert type(obj.problem) == expect_type
# 
# 
# def test_on_problem_for_unsupported_problem():
#     obj = GetProblem()
#     with pytest.raises(TypeError):
#         obj.receive_problem(UNSUPPORTED_PROBLEM)
# 
# 
# class GetInstanceData:
#     def __init__(self):
#         self.instance_data = None
# 
#     @validation.on_instance_data
#     def receive_instance_data(self, instance_data):
#         self.instance_data = instance_data
# 
# 
# params = {
#     "tuple_case": (("label", {"a": 1}), [[("label", {"a": 1})]]),
#     "list_case": ([[{"a": 1}]], [[("Unnamed[0][0]", {"a": 1})]]),
#     "dict_case": ({"a": 1}, [[("Unnamed[0]", {"a": 1})]]),
#     "None_case": (None, None),
# }
# 
# 
# @pytest.mark.parametrize(
#     "instance_data, expect", list(params.values()), ids=list(params.keys())
# )
# def test_on_instance_data(instance_data, expect):
#     obj = GetInstanceData()
#     obj.receive_instance_data(instance_data)
#     assert obj.instance_data == expect
# 
# 
# def test_on_instance_data_for_unsupported_instance_data():
#     obj = GetInstanceData()
#     with pytest.raises(TypeError):
#         obj.receive_instance_data(UNSUPPORTED_INSTANCE_DATA)
# 
# 
# params = {
#     "tuple_with_str_and_dict": (("label", {"a": 1}), True),
#     "invalid_tuple": (("element_1", "element_2"), False),
#     "not_tuple_case": ("invalid_input", False),
# }
# 
# 
# @pytest.mark.parametrize(
#     "instance_data, expect", list(params.values()), ids=list(params.keys())
# )
# def test_is_tuple_to_instance_data(instance_data, expect):
#     actual = validation._is_tuple_to_instance_data(instance_data)
#     assert actual == expect
# 
# 
# def test_tuple_to_instance_data():
#     instance_data = ("label", {"a": 1})
#     instance_data = validation._tuple_to_instance_data(instance_data)
#     assert type(instance_data) == list
# 
# 
# def test_tuple_to_instance_data_for_unsupported_instance_data_tuple():
#     with pytest.raises(TypeError):
#         validation._tuple_to_instance_data(UNSUPPORTED_INSTANCE_DATA_TUPLE)
# 
# 
# params = {
#     "multiple_list_with_data_label": ([[("label", {"a": 1})]], [[("label", {"a": 1})]]),
#     "multiple_list_no_data_label": ([[{"a": 1}]], [[("Unnamed[0][0]", {"a": 1})]]),
#     "single_list_with_data_label": ([("label", {"a": 1})], [[("label", {"a": 1})]]),
#     "single_list_no_data_label": ([{"a": 1}], [[("Unnamed[0]", {"a": 1})]]),
# }
# 
# 
# @pytest.mark.parametrize(
#     "instance_data, expect", list(params.values()), ids=list(params.keys())
# )
# def test_list_to_instance_data(instance_data, expect):
#     actual = validation._list_to_instance_data(instance_data)
#     assert actual == expect
# 
# 
# params = {
#     "the first is list with tuple": UNSUPPORTED_INSTANCE_DATA_LIST_0,
#     "the first is list with except tuple": UNSUPPORTED_INSTANCE_DATA_LIST_1,
#     "the first is tuple": UNSUPPORTED_INSTANCE_DATA_LIST_2,
#     "the first is others": UNSUPPORTED_INSTANCE_DATA_LIST_3,
# }
# 
# 
# @pytest.mark.parametrize(
#     "instance_data",
#     list(params.values()),
#     ids=list(params.keys()),
# )
# def test_list_to_instance_data_for_unsupported_instance_data_list(instance_data):
#     with pytest.raises(TypeError):
#         validation._list_to_instance_data(instance_data)
