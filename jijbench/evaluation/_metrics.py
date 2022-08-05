from __future__ import annotations
from re import A

from typing import Any, Union, Callable

import numpy as np
import warnings


class Scorer:
    def __init__(self, score_func: Callable, kwargs):
        self._score_func = score_func
        self._kwargs = kwargs

    def __call__(self, x: Any):
        if np.isnan([x.objective]).any():
            warnings.warn(
                'TTS cannot be calculated because "objective" is not stored in table attribute of jijbench.Benchmark instance.'
            )
            return np.nan

        if np.isnan([x.execution_time]).any():
            warnings.warn(
                'TTS cannot be calculated because "execution_time" is not stored in table attribute of jijbench.Benchmark instance.'
            )
            return np.nan

        if np.isnan([x.num_occurrences]).any():
            warnings.warn(
                'TTS cannot be calculated because "num_occurrences" is not stored in table attribute of jijbench.Benchmark instance.'
            )
            return np.nan

        if np.isnan([x.num_feasible]).any():
            warnings.warn(
                'TTS cannot be calculated because "num_feasible" is not stored in table attribute of jijbench.Benchmark instance.'
            )
            return np.nan

        if np.isnan(list(self._kwargs.values())).any():
            warnings.warn(
                "TTS cannot be calculated because np.nan exists in scoring method."
            )
            return np.nan

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
    feas = _to_bool_values_for_feasible(x)
    if feas.any():
        opt_value = x.objective[feas].min()
        ps = success_probability(x, opt_value)
    else:
        ps = 0.

    if ps:
        return np.log(1 - pr) / np.log(1 - ps) * x.execution_time
    else:
        return np.inf


def success_probability(x, opt_value: Union[int, float]):
    success = _to_bool_values_for_success(x, opt_value)
    num_success = (success * x.num_occurrences).sum()

    return num_success / x.num_occurrences.sum()


def feasible_rate(x):
    return x.num_feasible / x.num_occurrences.sum()


def residual_energy(x, opt_value: Union[int, float]):
    feas = _to_bool_values_for_feasible(x)
    if np.all(~feas):
        return np.nan
    else:
        obj = x.objective * feas * x.num_occurrences
        mean = obj.sum() / (feas * x.num_occurrences).sum()
        return mean - opt_value


def _to_bool_values_for_feasible(x):
    constraint_violations = np.array([v for v in x[x.index.str.contains("violations")]])
    return constraint_violations.sum(axis=0) == 0


def _to_bool_values_for_success(x, opt_value):
    feas = _to_bool_values_for_feasible(x)
    return (x.objective <= opt_value) & feas
