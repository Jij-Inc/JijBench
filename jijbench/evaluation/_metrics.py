from __future__ import annotations
from re import A

from typing import Any, Union, Callable

import functools
import numpy as np


class Scorer:
    def __init__(self, score_func: Callable, kwargs):
        self._score_func = score_func
        self._kwargs = kwargs

    def __call__(self, x: Any):
        return self._score_func(x, **self._kwargs)


def make_scorer(score_func: Callable, **kwargs):
    return Scorer(score_func, kwargs)


def optimal_time_to_solution(
    x,
    opt_value: Union[int, float],
    pr: float,
):
    ps = success_probability(x, opt_value)

    if ps:
        return np.log(1 - pr) / np.log(1 - ps) * x.execution_time
    else:
        return np.inf


def feasible_time_to_solution(
    x,
    pr: float,
):
    ps = feasible_rate(x)

    if ps:
        return np.log(1 - pr) / np.log(1 - ps) * x.execution_time
    else:
        return np.inf


def derived_time_to_solution(
    x,
    pr: float,
):
    ps = success_probability(x, x.obj_min)

    if ps:
        return np.log(1 - pr) / np.log(1 - ps) * x.execution_time
    else:
        return np.inf


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
    return (obj.mean() - opt_value)
