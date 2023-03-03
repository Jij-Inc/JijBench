from __future__ import annotations


import numpy as np
import pandas as pd
import typing as tp


from jijbench.experiment.experiment import Experiment
from jijbench.functions.concat import Concat
from jijbench.functions.factory import RecordFactory
from jijbench.functions.metrics import (
    TimeToSolution,
    SuccessProbability,
    FeasibleRate,
    ResidualEnergy,
)
from jijbench.node.base import FunctionNode

if tp.TYPE_CHECKING:
    from jijbench.mappings.mappings import Record


class Evaluation(FunctionNode[Experiment, Experiment]):
    """Evaluate the benchmark results."""

    def __call__(
        self,
        inputs: list[Experiment],
        opt_value: float | None = None,
        pr: float = 0.99,
    ) -> Experiment:
        return super().__call__(inputs, opt_value=opt_value, pr=pr)

    def operate(
        self,
        inputs: list[Experiment],
        opt_value: float,
        pr: float = 0.99,
    ) -> Experiment:
        """Calculate the typincal metrics of benchmark results.

        The metrics are as follows:
            - success_probability: Solution that is feasible and less than or equal to opt_value is counted as success, which is NaN if `opt_value` is not given.
            - feasible_rate: Rate of feasible solutions out of all solutions.
            - residual_energy: Difference between average objective of feasible solutions and `opt_value`, which is NaN if `opt_value` is not given.
            - TTS(optimal): Time to obtain opt_value with probability `pr`, which is NaN if opt_value is not given.
            - TTS(feasible): Time to obtain feasible solutions with probability `pr`.
            - TTS(derived): Time to obtain minimum objective among feasible solutions with probability `pr`.

        Args:
            opt_value (float, optional): Optimal value for instance_data.
            pr (float, optional): Probability of obtaining optimal value. Defaults to 0.99.

        Returns:
            Experiment: Experiment object included evalution results.
        """
        from icecream import ic
        from jijbench.solver.jijzept import SampleSet
        from jijbench.mappings.mappings import Table

        def f(x: pd.Series, opt_value: float, pr: float) -> pd.Series:
            inputs: list[SampleSet] = x.tolist()
            node = Concat()(inputs)
            metrics = [
                SuccessProbability()([node], opt_value=opt_value),
                FeasibleRate()([node]),
                ResidualEnergy()([node], opt_value=opt_value),
                TimeToSolution()([node], pr=pr, opt_value=opt_value, base="optimal"),
                TimeToSolution()([node], pr=pr, base="feasible"),
                TimeToSolution()([node], pr=pr, base="derived"),
            ]
            record = RecordFactory()(metrics)
            return record.data

        experiment = Concat()(inputs)

        artifact, table = experiment.data
        
        # TODO artifactの格納

        sampleset_columns = [
            c
            for c in table.columns
            if all(map(lambda x: isinstance(x, SampleSet), table.data[c]))
        ]

        data = table.data[sampleset_columns].apply(
            f, opt_value=opt_value, pr=pr, axis=1
        )
        metrics_table = Table(data)

        table = Concat()([table, metrics_table], axis=1)
        ic(table.view())

        return experiment

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
        is_warning = kwargs.pop("is_warning", False)
        func = make_scorer(func, is_warning, **kwargs)
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

        return self.apply(
            func=success_probability,
            opt_value=opt_value,
            column=column,
            expand=expand,
            axis=1,
            is_warning=True,
        )

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

        return self.apply(
            func=optimal_time_to_solution,
            opt_value=opt_value,
            pr=pr,
            column=f"{column}",
            expand=expand,
            axis=1,
            is_warning=True,
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

        return self.apply(
            func=feasible_time_to_solution,
            pr=pr,
            column=f"{column}",
            expand=expand,
            axis=1,
            is_warning=True,
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

        return self.apply(
            func=derived_time_to_solution,
            pr=pr,
            column=f"{column}",
            expand=expand,
            axis=1,
            is_warning=True,
        )

    def feasible_rate(self, column: str = "feasible_rate", expand: bool = True):
        """Feasible rate

        Args:
            column (str, optional): Column name for metircs table that is pandas.Dataframe. Defaults to "feasible_rate".
            expand (bool, optional): Whether to expand table with evaluation results. Defaults to True.

        Returns:
            pandas.Series: Feasible rate.
        """

        return self.apply(
            func=feasible_rate,
            column=column,
            expand=expand,
            axis=1,
            is_warning=True,
        )

    def residual_energy(
        self,
        opt_value: float,
        column: str = "residual_energy",
        expand: bool = True,
    ):
        """Residual energy

        Args:
            opt_value (float, optional): Optimal value for instance_data.
            column (str, optional): Column name for metircs table that is pandas.Dataframe. Defaults to "residual_energy".
            expand (bool, optional): Whether to expand table with evaluation results. Defaults to True.

        Returns:
            pandas.Series: Residual energy.
        """

        return self.apply(
            func=residual_energy,
            opt_value=opt_value,
            column=column,
            expand=expand,
            axis=1,
            is_warning=True,
        )
