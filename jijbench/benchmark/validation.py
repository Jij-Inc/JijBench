import functools
import jijmodeling as jm
from jijbench.solver import CallableSolver


def on_solver(fn):
    @functools.wraps(fn)
    def wrapper(obj, *args, **kwargs):
        (solver,) = args
        if isinstance(solver, (list, tuple)):
            callable_solvers = []
            for s in solver:
                callable_solvers.append(CallableSolver(s))
            if isinstance(solver, tuple):
                callable_solvers = tuple(callable_solvers)
        else:
            callable_solvers = [CallableSolver(solver)]
        fn(obj, callable_solvers)

    return wrapper


def on_problem(fn):
    @functools.wraps(fn)
    def wrapper(obj, *args, **kwargs):
        (problem,) = args
        if isinstance(problem, jm.Problem):
            fn(obj, [problem])
        elif isinstance(problem, (list, tuple)):
            fn(obj, problem)
        else:
            if problem is None:
                fn(obj, problem)
            else:
                raise Exception("対応していない問題のtypeです。")

    return wrapper


def on_instance_data(fn):
    @functools.wraps(fn)
    def wrapper(obj, *args, **kwargs):
        (instance_data,) = args
        if isinstance(instance_data, tuple):
            fn(obj, [_tuple_to_instance_data(instance_data)])
        elif isinstance(instance_data, list):
            fn(obj, _list_to_instance_data(instance_data))
        elif isinstance(instance_data, dict):
            fn(obj, [[("Unnamed[0]", instance_data)]])
        else:
            if instance_data is None:
                fn(obj, instance_data)
            else:
                raise Exception("対応していないインスタンスデータのtypeです")

    return wrapper


def _is_tuple_to_instance_data(d):
    if len(d) == 2:
        if isinstance(d[0], str) and isinstance(d[1], dict):
            return True
        else:
            False
    else:
        False


def _tuple_to_instance_data(d):
    if _is_tuple_to_instance_data(d):
        return [d]
    else:
        raise Exception("このフォーマットのインスタンスデータには対応していません")


def _list_to_instance_data(d):
    if isinstance(d[0], list):
        if isinstance(d[0][0], tuple):
            if _is_tuple_to_instance_data(d[0][0]):
                return d
            else:
                raise Exception("このフォーマットのインスタンスデータには対応していません")
        elif isinstance(d[0][0], dict):
            return [
                [(f"Unnamed[{i}][{j}]", dj) for j, dj in enumerate(di)]
                for i, di in enumerate(d)
            ]
        else:
            Exception("このフォーマットのインスタンスデータには対応していません")
    elif isinstance(d[0], tuple):
        if _is_tuple_to_instance_data(d[0]):
            return [d]
        else:
            raise Exception("このフォーマットのインスタンスデータには対応していません")
    elif isinstance(d[0], dict):
        return [[(f"Unnamed[{i}]", di) for i, di in enumerate(d)]]
    else:
        raise Exception("このフォーマットのインスタンスデータには対応していません")
