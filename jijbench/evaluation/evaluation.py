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
from jijbench.mappings.mappings import Artifact, Table
from jijbench.node.base import FunctionNode
from jijbench.solver.jijzept import SampleSet


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

        sampleset_columns = [
            col
            for col in table.columns
            if all(map(lambda x: isinstance(x, SampleSet), table.data[col]))
        ]

        data = table.data[sampleset_columns].apply(
            f, opt_value=opt_value, pr=pr, axis=1
        )
        metrics_artifact = Artifact(data.to_dict("index"))
        artifact = Concat()([artifact, metrics_artifact])

        metrics_table = Table(data)
        table = Concat()([table, metrics_table], axis=1)

        experiment.data = (artifact, table)
        return experiment
