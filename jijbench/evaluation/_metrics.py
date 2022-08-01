from __future__ import annotations

from typing import Any, Union, Callable

import numpy as np


class Scorer:
    def __init__(self, score_func: Callable, kwargs):
        self._score_func = score_func
        self._kwargs = kwargs

    def __call__(self, x: Any):
        return self._score_func(x, **self._kwargs)


def make_scorer(score_func: Callable, **kwargs):
    return Scorer(score_func, kwargs)


def time_to_solution(
    x,
    opt_value: Union[int, float],
    pr: float,
    solution_type: str,
):
    if solution_type == "optimal":
        ps = success_probability(x, opt_value)
    elif solution_type == "feasible":
        ps = feasible_rate(x)
    elif solution_type == "derived":
        ps = success_probability(x, x.obj_min)
    else:
        ps = np.nan

    if ps >= 1e-16:
        ps -= 1e-16
    else:
        ps += 1e-16
    return np.log(1 - pr) / np.log(1 - ps) * x.execution_time


def success_probability(x, opt_value: Union[int, float]):
    if np.isnan(opt_value):
        return np.nan
    else:
        constraint_violations = np.array(x[x.index.str.contains("violations")][0])
        return (
            np.nan
            if np.isnan(x.objective).all()
            else ((x.objective <= opt_value) & (constraint_violations == 0)).sum()
            / x.num_occurances.sum()
        )


def feasible_rate(x):
    return x.num_feasible / x.num_occurances.sum()


def residual_energy(x, opt_value: Union[int, float]):
    constraint_violations = np.array(x[x.index.str.contains("violations")][0])
    obj = x.objective[constraint_violations == 0]
    return (obj.mean() - opt_value) / np.abs(opt_value)
