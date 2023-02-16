from __future__ import annotations

import jijmodeling as jm
import numpy as np
import warnings


from jijbench.elements.base import Number
from jijbench.solver.jijzept import SampleSet
from jijbench.node.base import FunctionNode


def _is_success_list(sampleset: jm.SampleSet, opt_value: int | float) -> list[bool]:
    is_feas = _is_feasible_list(sampleset)
    objective = np.array(sampleset.evaluation.objective)
    return list((objective <= opt_value) & is_feas)


def _is_feasible_list(sampleset: jm.SampleSet) -> list[bool]:
    if sampleset.evaluation.constraint_violations is None:
        raise ValueError(
            "The value of sampleset.evaluation.constraint_violations is None. This SampleSet object is not evaluated."
        )
    else:
        constraint_violations = np.array(
            list(sampleset.evaluation.constraint_violations.values())
        )
        return constraint_violations.sum(axis=0) == 0


class Metrics(FunctionNode[SampleSet, Number]):
    pass

    # def __init__(self):
    # self._score_func = score_func
    # self._kwargs = kwargs
    # self._is_warning = is_warning
    #    pass

    # def __call__(self, inputs: list[Experiment]) -> Number:
    #     if self._is_warning:

    #         def _generate_warning_msg(metrics):
    #             return f'{self._score_func.__name__} cannot be calculated because "{metrics}" is not stored in table attribute of jijbench.Benchmark instance.'

    #         if np.isnan([x.objective]).any():
    #             warnings.warn(_generate_warning_msg("objective"))
    #             return np.nan

    #         if np.isnan([x.execution_time]).any():
    #             warnings.warn(_generate_warning_msg("execution_time"))
    #             return np.nan

    #         if np.isnan([x.num_occurrences]).any():
    #             warnings.warn(_generate_warning_msg("num_occurrences"))
    #             return np.nan

    #         if np.isnan([x.num_feasible]).any():
    #             warnings.warn(_generate_warning_msg("num_feasible"))
    #             return np.nan

    #         if np.isnan(list(self._kwargs.values())).any():
    #             warnings.warn(
    #                 f"{self._score_func.__name__} cannot be calculated because NaN exists in args for scoring method."
    #             )
    #             return np.nan

    #     self._score_func(x, **self._kwargs)

    #     return super().__call__(inputs, **kwargs)


# def make_scorer(score_func: Callable, is_warning=False, **kwargs) -> Scorer:
#    return Scorer(score_func, kwargs, is_warning)


class TimeToSolution(Metrics):
    def operate(
        self, inputs: list[SampleSet], opt_value: int | float, pr: float
    ) -> Number:
        res = jm.concatenate([node.data for node in inputs])
        num_occurrences = np.array(res.record.num_occurrences)
        time = res.measuring_time.solve.solve
        if sum(num_occurrences) == 1:
            warnings.warn("num_reads = 1; should be increased to measure TTS")

        node = SampleSet(res, "sampleset")
        f_ps = SuccessProbability()
        ps = f_ps([node])
        
        if ps.data == 1:
            data = 0.0
        elif ps:
            data = np.log(1 - pr) / np.log(1 - ps) * x.execution_time

        return super().operate(inputs)


class SuccessProbability(Metrics):
    def operate(self, inputs: list[SampleSet], opt_value: int | float) -> Number:
        res = jm.concatenate([node.data for node in inputs])
        num_occurrences = np.array(res.record.num_occurrences)
        num_success = sum(_is_success_list(res, opt_value) * num_occurrences)
        data = num_success / sum(num_occurrences)
        return Number(data, "success_probability")


class FeasibleRate(Metrics):
    def operate(self, inputs: list[SampleSet]) -> Number:
        res = jm.concatenate([node.data for node in inputs])
        num_occurrences = np.array(res.record.num_occurrences)
        num_feasible = sum(_is_feasible_list(res) * num_occurrences)
        data = num_feasible / sum(num_occurrences)
        return Number(data, "feasible_rate")


class ResidualEnergy(Metrics):
    def operate(self, inputs: list[SampleSet]) -> Number:
        return super().operate(inputs)


def optimal_time_to_solution(
    x,
    opt_value: int | float,
    pr: float,
):
    if x.num_reads == 1:
        warnings.warn("num_reads = 1; should be increased to measure optimal TTS")

    ps = success_probability(x, opt_value)

    if ps == 1:
        return np.nan
    elif ps:
        return np.log(1 - pr) / np.log(1 - ps) * x.execution_time
    else:
        return np.inf


def feasible_time_to_solution(
    x,
    pr: float,
):
    if x.num_reads == 1:
        warnings.warn("num_reads = 1; should be increased to measure feasible TTS")

    ps = feasible_rate(x)

    if ps == 1:
        return np.nan
    elif ps:
        return np.log(1 - pr) / np.log(1 - ps) * x.execution_time
    else:
        return np.inf


def derived_time_to_solution(
    x,
    pr: float,
):
    if x.num_reads == 1:
        warnings.warn("num_reads = 1; should be increased to measure derived TTS")

    feas = _to_bool_values_for_feasible(x)
    if feas.any():
        opt_value = x.objective[feas].min()
        ps = success_probability(x, opt_value)
    else:
        ps = 0.0

    if ps == 1:
        return np.nan
    elif ps:
        return np.log(1 - pr) / np.log(1 - ps) * x.execution_time
    else:
        return np.inf


def success_probability(x, opt_value: int | float):
    success = _to_bool_values_for_success(x, opt_value)
    num_success = (success * x.num_occurrences).sum()

    return num_success / x.num_occurrences.sum()


def feasible_rate(x):
    return x.num_feasible / x.num_occurrences.sum()


def residual_energy(x, opt_value: int | float):
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
