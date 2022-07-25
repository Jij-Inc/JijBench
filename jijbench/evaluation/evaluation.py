from __future__ import annotations

from typing import Union, Optional

import numpy as np
import pandas as pd

from jijbench.experiment import Experiment
from jijbench.evaluation._metrics import (
    make_scorer,
    time_to_solution,
    success_probability,
    feasible_rate,
    residual_energy,
)


class Evaluator:
    def __init__(self, experiment: Experiment):
        self.experiment = experiment

    @property
    def table(self):
        return self.experiment.table

    @property
    def artifact(self):
        return self.experiment.artifact

    def calc_typical_metrics(
        self, opt_value: Optional[float] = None, pr: float = 0.99, expand: bool = True
    ):
        """_summary_

        Args:
            opt_value (_type_, optional): _description_. Defaults to None.
            pr (float, optional): _description_. Defaults to 0.99.
            expand (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        opt_value = np.nan if opt_value is None else opt_value

        metrics = pd.DataFrame()
        metrics["success_probability"] = self.success_probability(
            opt_value=opt_value, expand=expand
        )
        metrics["feasible_rate"] = self.feasible_rate(expand=expand)
        metrics["residual_energy"] = self.residual_energy(
            opt_value=opt_value, expand=expand
        )
        metrics["TTS(optimal)"] = self.time_to_solution(
            opt_value=opt_value, pr=pr, solution_type="optimal", expand=expand
        )
        metrics["TTS(feasible)"] = self.time_to_solution(
            opt_value=opt_value, pr=pr, solution_type="feasible", expand=expand
        )
        metrics["TTS(derived)"] = self.time_to_solution(
            opt_value=opt_value, pr=pr, solution_type="derived", expand=expand
        )
        return metrics

    def apply(self, func, column, expand=True, axis=1, **kwargs):
        func = make_scorer(func, **kwargs)
        metrics = self.table.apply(func, axis=axis)
        if expand:
            self.table[column] = metrics
        return metrics

    def success_probability(
        self,
        opt_value: Union[int, float],
        column: str = "success_probability",
        expand: bool = True,
    ):
        scorer = make_scorer(success_probability, opt_value=opt_value)
        return self.apply(func=scorer, column=column, expand=expand, axis=1)

    def time_to_solution(
        self,
        opt_value: Union[int, float],
        pr: float = 0.99,
        solution_type: str = "optimal",
        column: str = "TTS",
        expand: bool = True,
    ):
        scorer = make_scorer(
            time_to_solution,
            opt_value=opt_value,
            pr=pr,
            solution_type=solution_type,
        )
        return self.apply(
            func=scorer, column=f"{column}({solution_type})", expand=expand, axis=1
        )

    def feasible_rate(self, column: str = "feasible_rate", expand: bool = True):
        scorer = make_scorer(feasible_rate)
        return self.apply(func=scorer, column=column, expand=expand, axis=1)

    def residual_energy(
        self,
        opt_value: Union[int, float],
        column: str = "residual_energy",
        expand: bool = True,
    ):
        scorer = make_scorer(residual_energy, opt_value=opt_value)
        return self.apply(func=scorer, column=column, expand=expand, axis=1)
