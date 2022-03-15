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

    def calc_typical_metircs(self, opt_value=None, pr=0.99, expand=True):
        metrics = pd.DataFrame()
        metrics["success_probability"] = self.ps(opt_value=opt_value, expand=expand)
        metrics["feasible_rate"] = self.fr(expand=expand)
        metrics["residual_energy"] = self.re(opt_value=opt_value, expand=expand)
        metrics["TTS(optimal)"] = self.tts(
            opt_value=opt_value, pr=pr, solution="optimal", expand=expand
        )
        metrics["TTS(feasible)"] = self.tts(pr=pr, solution="feasible", expand=expand)
        metrics["TTS(derived)"] = self.tts(pr=pr, solution="derived", expand=expand)
        return metrics

    def apply(self, func, column, expand=True, axis=0):
        metrics = self.table.apply(func, axis=axis)
        if expand:
            self.table[column] = metrics
        return metrics

    def success_probability(
        self, opt_value=None, column="success_probability", expand=True
    ):
        scorer = Evaluator.make_scorer(
            _Metrics.success_probability, opt_value=self._to_available(opt_value)
        )
        return self.apply(func=scorer, column=column, expand=expand, axis=1)

    def time_to_solution(
        self, opt_value=None, pr=0.99, solution="optimal", column="TTS", expand=True
    ):
        scorer = Evaluator.make_scorer(
            _Metrics.time_to_solution,
            opt_value=self._to_available(opt_value),
            pr=pr,
            solution=solution,
        )
        return self.apply(
            func=scorer, column=f"{column}({solution})", expand=expand, axis=1
        )

    def feasible_rate(self, column="feasible_rate", expand=True):
        scorer = Evaluator.make_scorer(_Metrics.feasible_rate)
        return self.apply(func=scorer, column=column, expand=expand, axis=1)

    def residual_energy(self, opt_value=None, column="residual_energy", expand=True):
        scorer = Evaluator.make_scorer(
            _Metrics.residual_energy, opt_value=self._to_available(opt_value)
        )
        return self.apply(func=scorer, column=column, expand=expand, axis=1)

    @staticmethod
    def make_scorer(score_func, **kwargs):
        return _Scorer(score_func, kwargs)

    @staticmethod
    def _to_available(value):
        if value is None:
            value = np.nan
        return value


class _Scorer:
    def __init__(self, score_func, kwargs):
        self._score_func = score_func
        self._kwargs = kwargs

    def __call__(self, x):
        return self._score_func(x, **self._kwargs)


class _Metrics:
    def time_to_solution(x, opt_value=None, pr=0.99, solution="optimal"):
        if solution == "optiaml":
            ps = _Metrics.success_probability(x, opt_value) + 1e-16
        elif solution == "feasible":
            ps = _Metrics.feasible_rate(x) + 1e-16
        elif solution == "derived":
            ps = _Metrics.success_probability(x, x.energy_min) + 1e-16
        else:
            ps = np.nan
        return (
            np.log(1 - pr) / np.log(1 - ps) * x.execution_time
            if ps < pr
            else x.execution_time
        )

    def success_probability(x, opt_value):
        return (x.energy <= opt_value).sum() / len(x.energy)

    def feasible_rate(x):
        return x.num_feasible / x.num_samples

    def residual_energy(x, opt_value):
        return (x.energy_mean - opt_value) / np.abs(opt_value)
