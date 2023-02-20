from __future__ import annotations

import jijmodeling as jm
import numpy as np
import typing as tp
import warnings


from jijbench.elements.base import Number
from jijbench.functions.concat import Concat
from jijbench.solver.jijzept import SampleSet
from jijbench.node.base import FunctionNode


def _is_success_list(sampleset: SampleSet, opt_value: int | float) -> list[bool]:
    is_feas = _is_feasible_list(sampleset)
    objective = np.array(sampleset.evaluation.objective)
    return list((objective <= opt_value) & is_feas)


def _is_feasible_list(sampleset: SampleSet) -> list[bool]:
    if sampleset.evaluation.constraint_violations is None:
        raise ValueError(
            "The value of sampleset.evaluation.constraint_violations is None. This SampleSet object is not evaluated or has not constraints."
        )
    else:
        constraint_violations = np.array(
            list(sampleset.evaluation.constraint_violations.values())
        )
        return constraint_violations.sum(axis=0) == 0


class Metrics(FunctionNode[SampleSet, Number]):
    """A base class for metrics."""

    def __call__(self, inputs: list[SampleSet], **kwargs: tp.Any) -> Number:
        """Calculate the metric.

        Args:
            inputs (list[SampleSet]): A list of SampleSet objects.
        """

        def _generate_warning_msg(metrics):
            return f'{self.__class__.__name__} cannot be calculated because "{metrics}" is not stored in table of Experiment object.'

        concat: Concat[SampleSet] = Concat()
        node = concat(inputs, "sampleset")

        objective = node.evaluation.objective
        if objective is None:
            raise ValueError("This SampleSet object is not evaluated.")
        if np.isnan(objective).any():
            warnings.warn(_generate_warning_msg("objective"))

        time = node.time.solve
        if time is None:
            warnings.warn(_generate_warning_msg("execution_time"))
        else:
            if time.solve is None:
                warnings.warn(_generate_warning_msg("execution_time"))

        num_occurrences = node.record.num_occurrences
        if np.isnan(num_occurrences).any():
            warnings.warn(_generate_warning_msg("num_occurrences"))

        return super().__call__([node], **kwargs)


class TimeToSolution(Metrics):
    """Time to solution.
    The time to solution is defined as the time to obtain a solution with probability `pr`.
    The solution is defined as the solution that is feasible and less than or equal to `opt_value`.
    There are three types of time to solution:
        - optimal: The solution is the optimal solution.
        - feasible: The solution is feasible.
        - derived: The solution is the best solution among the obtained solutions.
    """

    @tp.overload
    def __call__(
        self,
        inputs: list[SampleSet],
        pr: float,
        *,
        opt_value: int | float,
        base: tp.Literal["optimal"] = "optimal",
    ) -> Number:
        ...

    @tp.overload
    def __call__(
        self, inputs: list[SampleSet], pr: float, *, base: tp.Literal["feasible"]
    ) -> Number:
        ...

    @tp.overload
    def __call__(
        self, inputs: list[SampleSet], pr: float, *, base: tp.Literal["derived"]
    ) -> Number:
        ...

    def __call__(
        self,
        inputs: list[SampleSet],
        pr: float,
        *,
        opt_value: int | float | None = None,
        base: tp.Literal["optimal", "feasible", "derived"] = "optimal",
    ) -> Number:
        """Calculate time to solution.

        Args:
            inputs (list[SampleSet]): A list of SampleSet objects.
            pr (float): The probability of obtaining a solution.
            opt_value (int or float): The optimal value.
            base (str): The type of time to solution.

        Returns:
            Number: The time to solution.
        """
        return super().__call__(inputs, pr=pr, opt_value=opt_value, base=base)

    @tp.overload
    def operate(
        self,
        inputs: list[SampleSet],
        pr: float,
        *,
        opt_value: int | float,
        base: tp.Literal["optimal"] = "optimal",
    ) -> Number:
        ...

    @tp.overload
    def operate(
        self, inputs: list[SampleSet], pr: float, *, base: tp.Literal["feasible"]
    ) -> Number:
        ...

    @tp.overload
    def operate(
        self, inputs: list[SampleSet], pr: float, *, base: tp.Literal["derived"]
    ) -> Number:
        ...

    def operate(
        self,
        inputs: list[SampleSet],
        pr: float,
        *,
        opt_value: int | float | None = None,
        base: tp.Literal["optimal", "feasible", "derived"] = "optimal",
    ) -> Number:
        """Calculate time to solution.

        Args:
            inputs (list[SampleSet]): A list of SampleSet objects.
            pr (float): The probability of obtaining a solution.
            opt_value (int or float): The optimal value.
            base (str): The type of time to solution.

        Returns:
            Number: The time to solution.
        """

        node = inputs[0]

        num_occurrences = np.array(node.record.num_occurrences)
        if sum(num_occurrences) == 1:
            warnings.warn("num_reads = 1; should be increased to measure TTS")

        if base == "optimal":
            f_ps = SuccessProbability()
            ps = f_ps([node], opt_value=opt_value or np.nan)
        elif base == "feasible":
            f_ps = FeasibleRate()
            ps = f_ps([node])
        elif base == "derived":
            f_ps = SuccessProbability()
            is_feas = _is_feasible_list(node)
            if any(is_feas):
                opt_value = np.array(node.evaluation.objective)[is_feas].min()
                ps = f_ps([node], opt_value=opt_value or np.nan)
            else:
                ps = f_ps([node], opt_value=np.inf)

        time = np.nan if node.time.solve is None else node.time.solve.solve
        time = np.nan if time is None else time

        if ps.data == 1:
            data = 0.0
        elif ps:
            data = np.log(1 - pr) / np.log(1 - ps.data) * time
        else:
            data = np.inf
        return Number(data, f"TTS[{base}]")


class SuccessProbability(Metrics):
    """Success probability.
    The success probability is defined as the probability of obtaining a solution less than or equal to `opt_value`.
    """

    def __call__(self, inputs: list[SampleSet], opt_value: int | float) -> Number:
        """Calculate success probability.

        Args:
            inputs (list[SampleSet]): A list of SampleSet objects.
            opt_value (int or float): The optimal value.

        Returns:
            Number: The success probability.
        """
        return super().__call__(inputs, opt_value=opt_value)

    def operate(self, inputs: list[SampleSet], opt_value: int | float) -> Number:
        """Calculate success probability.

        Args:
            inputs (list[SampleSet]): A list of SampleSet objects.
            opt_value (int or float): The optimal value.

        Returns:
            Number: The success probability.
        """
        node = inputs[0]
        num_occurrences = np.array(node.record.num_occurrences)
        num_success = sum(_is_success_list(node, opt_value) * num_occurrences)
        data = num_success / sum(num_occurrences)
        return Number(data, "success_probability")


class FeasibleRate(Metrics):
    """Feasible rate.
    The feasible rate is defined as the probability of obtaining a feasible solution.
    """

    def __call__(self, inputs: list[SampleSet]) -> Number:
        """Calculate feasible rate.

        Args:
            inputs (list[SampleSet]): A list of SampleSet objects.

        Returns:
            Number: The feasible rate.

        """
        return super().__call__(inputs)

    def operate(self, inputs: list[SampleSet]) -> Number:
        node = inputs[0]
        num_occurrences = np.array(node.record.num_occurrences)
        num_feasible = sum(_is_feasible_list(node) * num_occurrences)
        data = num_feasible / sum(num_occurrences)
        return Number(data, "feasible_rate")


class ResidualEnergy(Metrics):
    """Residual energy.
    The residual energy is defined as the mean of the objective function minus the optimal value.
    """
    def __call__(self, inputs: list[SampleSet], opt_value: int | float) -> Number:
        return super().__call__(inputs, opt_value=opt_value)

    def operate(self, inputs: list[SampleSet], opt_value: int | float) -> Number:
        node = inputs[0]

        is_feas = np.array(_is_feasible_list(node))
        if all(~is_feas):
            data = np.nan
        else:
            num_occurrences = np.array(node.record.num_occurrences)
            objective = np.array(node.evaluation.objective) * is_feas * num_occurrences
            mean = objective.sum() / (is_feas * num_occurrences).sum()
            data = float(mean - opt_value)
        return Number(data, "residual_energy")
