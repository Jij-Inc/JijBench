from __future__ import annotations

from typing import Callable, Union, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

from jijbench.evaluation._metrics import (
    make_scorer,
    optimal_time_to_solution,
    feasible_time_to_solution,
    derived_time_to_solution,
    success_probability,
    feasible_rate,
    residual_energy,
)

if TYPE_CHECKING:
    from jijbench.benchmark.benchmark import Benchmark
    from jijbench.experiment.experiment import Experiment


class Evaluator:
    """Evaluate becnhmark results.

    Args:
        experiment (Union[jijbench.Experiment, jijbench.Benchmark]): Experiment or Benchmark object.
    Attibutes:
        table (pandas.DataFrame): Table that store experiment and evaluation resutls.
        artifact (dict): Dict that store experiment results.
    """

    def __init__(self, experiment: Union[Benchmark, Experiment]):
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
        """Calculate typincal metrics for benchmark

        Args:
            opt_value (float, optional): Optimal value for instance_data. Defaults to None.
            pr (float, optional): Probability of obtaining optimal value. Defaults to 0.99.
            expand (bool, optional): If True, expand table with evaluation results. Defaults to None.

        Returns:
            pandas.Dataframe: pandas.Dataframe object for evalution results.
            columns: ["success_probability", "feasible_rate", "residual_energy", "TTS(optimal)", "TTS(feasible)", "TTS(derived)"]
            - success_probability: Solution that is feasible and less than or equal to opt_value is counted as success, which is NaN if `opt_value` is not given.
            - feasible_rate: Rate of feasible solutions out of all solutions.
            - residual_energy: Difference between average objective of feasible solutions and `opt_value`, which is NaN if `opt_value` is not given.
            - TTS(optimal): Time to obtain opt_value with probability `pr`, which is NaN if opt_value is not given.
            - TTS(feasible): Time to obtain feasible solutions with probability `pr`.
            - TTS(derived): Time to obtain minimum objective among feasible solutions with probability `pr`.
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
        metrics["TTS(optimal)"] = self.optimal_time_to_solution(
            opt_value=opt_value, pr=pr, expand=expand
        )
        metrics["TTS(feasible)"] = self.feasible_time_to_solution(pr=pr, expand=expand)
        metrics["TTS(derived)"] = self.derived_time_to_solution(pr=pr, expand=expand)
        return metrics

    def apply(self, func: Callable, column: str, expand=True, axis=1, **kwargs):
        """Apply evaluation function to table

        Args:
            func (Callable): Callable object to calculate evaluation metrics.
            column (str, optional): Column name for metircs table that is pandas.Dataframe.
            expand (bool, optional): Whether to expand table with evaluation results. Defaults to True.
            axis (int, optional): Axis along which the `func` is applied:
                - 0: apply `func` to column direction.
                - 1(default): apply `func` to row direction.

        Returns:
            pandas.Series: Evalution results.
        """
        func = make_scorer(func, **kwargs)
        metrics = self.table.apply(func, axis=axis)
        if expand:
            self.table[column] = metrics
        return metrics

    def success_probability(
        self,
        opt_value: float,
        column: str = "success_probability",
        expand: bool = True,
    ):
        """Success probability

        Args:
            opt_value (float): Optimal value for instance_data.
            column (str, optional): Column name for metircs table that is pandas.Dataframe. Defaults to "success_probability".
            expand (bool, optional): Whether to expand table with evaluation results. Defaults to True.

        Returns:
            pandas.Series: Success Probability.
        """
        scorer = make_scorer(success_probability, opt_value=opt_value)
        return self.apply(func=scorer, column=column, expand=expand, axis=1)

    def optimal_time_to_solution(
        self,
        opt_value: float,
        pr: float = 0.99,
        column: str = "TTS(optimal)",
        expand: bool = True,
    ):
        """Time to solution for optimal value.

        Args:
            opt_value (float, optional): Optimal value for instance_data.
            pr (float, optional): Probability of obtaining optimal value. Defaults to 0.99.
            column (str, optional): Column name for metircs table that is pandas.Dataframe. Defaults to "TTS(optimal)".
            expand (bool, optional): Whether to expand table with evaluation results. Defaults to True.

        Returns:
            pandas.Series: Time to Solution for optimal value.
        """

        scorer = make_scorer(
            optimal_time_to_solution,
            opt_value=opt_value,
            pr=pr,
        )
        return self.apply(
            func=scorer, column=f"{column}", expand=expand, axis=1
        )

    def feasible_time_to_solution(
        self,
        pr: float = 0.99,
        column: str = "TTS(feasible)",
        expand: bool = True,
    ):
        """Time to solution for feasible.

        Args:
            pr (float, optional): Probability of obtaining optimal value. Defaults to 0.99.
            column (str, optional): Column name for metircs table that is pandas.Dataframe. Default to "TTS(feasible)"
            expand (bool, optional): Whether to expand table with evaluation results. Defaults to True.

        Returns:
            pandas.Series: Time to Solution for feasible.
        """
        scorer = make_scorer(feasible_time_to_solution, pr=pr)
        return self.apply(
            func=scorer, column=f"{column}", expand=expand, axis=1
        )

    def derived_time_to_solution(
        self,
        pr: float = 0.99,
        column: str = "TTS(derived)",
        expand: bool = True,
    ):
        """Time to solution for min value among obtained objective.

        Args:
            pr (float, optional): Probability of obtaining optimal value. Defaults to 0.99.
            column (str, optional): Column name for metircs table that is pandas.Dataframe. Default to "TTS(derived)"
            expand (bool, optional): Whether to expand table with evaluation results. Defaults to True.

        Returns:
            pandas.Series: Time to Solution for min value among obtained objective.
        """
        scorer = make_scorer(
            derived_time_to_solution,
            pr=pr,
        )
        return self.apply(
            func=scorer, column=f"{column}", expand=expand, axis=1
        )

    def feasible_rate(self, column: str = "feasible_rate", expand: bool = True):
        """Feasible rate

        Args:
            column (str, optional): Column name for metircs table that is pandas.Dataframe. Defaults to "feasible_rate".
            expand (bool, optional): Whether to expand table with evaluation results. Defaults to True.

        Returns:
            pandas.Series: Feasible rate.
        """
        scorer = make_scorer(feasible_rate)
        return self.apply(func=scorer, column=column, expand=expand, axis=1)

    def residual_energy(
        self,
        opt_value: float,
        column: str = "residual_energy",
        expand: bool = True,
    ):
        if solution_type == "optimal":
            ps = _Metrics.success_probability(x, opt_value)
        elif solution_type == "feasible":
            ps = _Metrics.feasible_rate(x)
        elif solution_type == "derived":
            ps = _Metrics.success_probability(x, x.energy_min)
        else:
            ps = np.nan
        if ps >= 1e-16:
            ps -= 1e-16
        else:
            ps += 1e-16
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
