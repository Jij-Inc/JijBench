from __future__ import annotations

import functools

import jijmodeling as jm

from jijbench.solver import CallableSolver

from jijbench.exceptions import (
    UnsupportedProblemError,
    UnsupportedInstanceDataError,
)

__all__ = []


def on_solver(fn):
    @functools.wraps(fn)
    def wrapper(obj, *args, **kwargs):
        (solver,) = args
        if isinstance(solver, (list, tuple)):
            print("1_on_solver_if")
            callable_solvers = []
            for s in solver:
                callable_solvers.append(CallableSolver(s))
            if isinstance(solver, tuple):
                callable_solvers = tuple(callable_solvers)
                print("2_on_solver_if_if")
        else:
            print("3_on_solver_else")
            callable_solvers = [CallableSolver(solver)]
        fn(obj, callable_solvers)

    return wrapper


def on_problem(fn):
    @functools.wraps(fn)
    def wrapper(obj, *args, **kwargs):
        (problem,) = args
        if isinstance(problem, jm.Problem):
            print("4_on_problem_if")
            fn(obj, [problem])
        elif isinstance(problem, (list, tuple)):
            print("5_on_problem_elif")
            fn(obj, problem)
        else:
            if problem is None:
                print("6_on_problem_else_if")
                fn(obj, problem)
            else:
                raise UnsupportedProblemError("problem of this type is not supported.")

    return wrapper


def on_instance_data(fn):
    @functools.wraps(fn)
    def wrapper(obj, *args, **kwargs):
        (instance_data,) = args
        if isinstance(instance_data, tuple):
            print("7_on_instance_data_tuple")
            fn(obj, [_tuple_to_instance_data(instance_data)])
        elif isinstance(instance_data, list):
            print("8_on_instance_data_list")
            fn(obj, _list_to_instance_data(instance_data))
        elif isinstance(instance_data, dict):
            print("9_on_instance_data_dict")
            fn(obj, [[("Unnamed[0]", instance_data)]])
        else:
            if instance_data is None:
                print("10_on_instance_data_None")
                fn(obj, instance_data)
            else:
                raise UnsupportedInstanceDataError(
                    "problem of this type is not supported."
                )

    return wrapper


def _is_tuple_to_instance_data(d):
    if len(d) == 2:
        if isinstance(d[0], str) and isinstance(d[1], dict):
            print("11_is_tuple_to_instance_data_len=2_valid")
            return True
        else:
            print("12_is_tuple_to_instance_data_len=2_invalid")
            False
    else:
        False


def _tuple_to_instance_data(d):
    if _is_tuple_to_instance_data(d):
        return [d]
    else:
        raise UnsupportedInstanceDataError(
            "instance_data of this type is not supported."
        )


def _list_to_instance_data(d):
    if isinstance(d[0], list):
        if isinstance(d[0][0], tuple):
            if _is_tuple_to_instance_data(d[0][0]):
                return d
            else:
                raise UnsupportedInstanceDataError(
                    "instance_data of this type is not supported."
                )
        elif isinstance(d[0][0], dict):
            return [
                [(f"Unnamed[{i}][{j}]", dj) for j, dj in enumerate(di)]
                for i, di in enumerate(d)
            ]
        else:
            raise UnsupportedInstanceDataError(
                "instance_data of this type is not supported."
            )
    elif isinstance(d[0], tuple):
        if _is_tuple_to_instance_data(d[0]):
            return [d]
        else:
            raise UnsupportedInstanceDataError(
                "instance_data of this type is not supported."
            )
    elif isinstance(d[0], dict):
        return [[(f"Unnamed[{i}]", di) for i, di in enumerate(d)]]
    else:
        raise UnsupportedInstanceDataError(
            "instance_data of this type is not supported."
        )
