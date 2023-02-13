from __future__ import annotations

import jijmodeling as jm
import typing as tp
import inspect

from dataclasses import dataclass
from jijbench.elements.base import Element
from jijbench.exceptions.exceptions import SolverFailedError
from jijbench.node.base import FunctionNode
from jijbench.mappings.mappings import Record
from jijbench.functions.factory import RecordFactory
from jijbench.typing import T


@dataclass
class Parameter(Element[T]):
    """A parameter for a solver function.

    Attributes:
        data (Any): The data in the node.
        name (str): The name of the parameter.
    """

    @classmethod
    def validate_data(cls, data: tp.Any) -> tp.Any:
        """A class method to validate the data before setting it.

        Args:
            data (Any): The data to be validated.

        Returns:
            Any: The validated data.
        """
        return data


@dataclass
class Return(Element[T]):
    """A return value of a solver function.

    Attributes:
        data (Any): The data in the node.
        name (str): The name of the return value.
    """

    @classmethod
    def validate_data(cls, data: tp.Any) -> tp.Any:
        """A class method to validate the data before setting it.

        Args:
            data (Any): The data to be validated.

        Returns:
            Any: The validated data.
        """
        return data


class Solver(FunctionNode[Parameter, Record]):
    """A solver function that takes a list of Parameter and returns a Record.

    Attributes:
        name (str): The name of the solver function.
        function (Callable): The actual function to be executed.
    """

    def __init__(self, function: tp.Callable, name: str | None = None) -> None:
        """The constructor of the `Solver` class.

        Args:
            function (Callable): The actual function to be executed.
            name (str, optional): The name of the solver function. Defaults to None.
        """
        if name is None:
            name = function.__name__
        super().__init__(name)
        self.function = function

    def operate(
        self,
        inputs: list[Parameter],
        is_parsed_sampleset: bool = True,
    ) -> Record:
        """The main operation of the solver function.

        Args:
            inputs (list[Parameter]): The list of input `Parameter` for the solver function.
            is_parsed_sampleset (bool, optional): Whether the sample set is parsed. Defaults to True.

        Raises:
            SolverFailedError: If an error occurs inside the solver function.

        Returns:
            Record: The result of the solver function as a `Record`.
        """
        parameters = inspect.signature(self.function).parameters
        solver_args = {
            node.name: node.data for node in inputs if node.name in parameters
        }
        try:
            rets = self.function(**solver_args)
            if not isinstance(rets, tuple):
                rets = (rets,)
        except Exception as e:
            msg = f'An error occurred inside your solver. Please check implementation of "{self.name}". -> {e}'
            raise SolverFailedError(msg)

        solver_return_names = [f"{self.name}_return[{i}]" for i in range(len(rets))]

        rets = [Return(data, name) for data, name in zip(rets, solver_return_names)]
        factory = RecordFactory()
        return factory(rets, is_parsed_sampleset=is_parsed_sampleset)
