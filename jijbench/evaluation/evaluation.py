from typing import Any, Callable, Union

import numpy as np
import pandas as pd

from jijbench.experiment import Experiment


class Evaluator:
    def __init__(self, experiment: Experiment):
        self.experiment = experiment

    @property
    def table(self):
        return self.experiment.table

    @property
    def artifact(self):
        return self.experiment.artifact

    @property
    def ps(self):
        return self.success_probability

    @property
    def tts(self):
        return self.time_to_solution

    @property
    def fr(self):
        return self.feasible_rate

    @property
    def re(self):
        return self.residual_energy

    def calc_typical_metrics(self, opt_value=None, pr=0.99, expand=True):
        metrics = pd.DataFrame()
        metrics["success_probability"] = self.ps(opt_value=opt_value, expand=expand)
        metrics["feasible_rate"] = self.fr(expand=expand)
        metrics["residual_energy"] = self.re(opt_value=opt_value, expand=expand)
        metrics["TTS(optimal)"] = self.tts(
            opt_value=opt_value, pr=pr, solution_type="optimal", expand=expand
        )
        metrics["TTS(feasible)"] = self.tts(
            pr=pr, solution_type="feasible", expand=expand
        )
        metrics["TTS(derived)"] = self.tts(
            pr=pr, solution_type="derived", expand=expand
        )
        return metrics

    def apply(self, func, column, expand=True, axis=1, **kwargs):
        func = self.make_scorer(func, **kwargs)
        metrics = self.table.apply(func, axis=axis)
        if expand:
            self.table[column] = metrics
        return metrics

    def success_probability(
        self,
        opt_value: Union[int, float] = None,
        column: str = "success_probability",
        expand: bool = True,
    ):
        scorer = Evaluator.make_scorer(
            _Metrics.success_probability, opt_value=self._none_to_value(opt_value)
        )
        return self.apply(func=scorer, column=column, expand=expand, axis=1)

    def time_to_solution(
        self,
        opt_value: Union[int, float] = None,
        pr: float = 0.99,
        solution_type: str = "optimal",
        column: str = "TTS",
        expand: bool = True,
    ):
        scorer = Evaluator.make_scorer(
            _Metrics.time_to_solution,
            opt_value=self._none_to_value(opt_value),
            pr=pr,
            solution_type=solution_type,
        )
        return self.apply(
            func=scorer, column=f"{column}({solution_type})", expand=expand, axis=1
        )

    def feasible_rate(self, column: str = "feasible_rate", expand: bool = True):
        scorer = Evaluator.make_scorer(_Metrics.feasible_rate)
        return self.apply(func=scorer, column=column, expand=expand, axis=1)

    def residual_energy(
        self,
        opt_value: Union[int, float] = None,
        column: str = "residual_energy",
        expand: bool = True,
    ):
        scorer = Evaluator.make_scorer(
            _Metrics.residual_energy, opt_value=self._none_to_value(opt_value)
        )
        return self.apply(func=scorer, column=column, expand=expand, axis=1)

    @staticmethod
    def make_scorer(score_func: Callable, **kwargs):
        return _Scorer(score_func, kwargs)

    @staticmethod
    def _none_to_value(value: Union[int, float]):
        return value if value else np.nan


class _Scorer:
    def __init__(self, score_func: Callable, kwargs):
        self._score_func = score_func
        self._kwargs = kwargs

    def __call__(self, x: Any):
        return self._score_func(x, **self._kwargs)


class _Metrics:
    def time_to_solution(
        x: Any,
        opt_value: Union[int, float],
        *,
        pr: float = 0.99,
        solution_type: str = "optimal",
    ):
        if solution_type == "optimal":
            ps = _Metrics.success_probability(x, opt_value)
        elif solution_type == "feasible":
            ps = _Metrics.feasible_rate(x)
        elif solution_type == "derived":
            ps = _Metrics.success_probability(x, x.energy_min)
        else:
            ps = np.nan
        return np.log(1 - pr) / np.log(1 - ps) * x.execution_time

    def success_probability(x: Any, opt_value: Union[int, float]):
        if np.isnan(opt_value):
            return np.nan
        else:
            return (
                np.nan
                if np.isnan(x.energy).all()
                else (x.energy <= opt_value).sum() / x.num_reads
            )

    def feasible_rate(x: Any):
        return x.num_feasible / x.num_samples

    def residual_energy(x: Any, opt_value: Union[int, float]):
        return (x.energy_mean - opt_value) / np.abs(opt_value)
