from __future__ import annotations


import typing as tp
import inspect

from jijbench.exceptions.exceptions import SolverFailedError
from jijbench.node.base import FunctionNode
from jijbench.data.mapping import Record
from jijbench.data.elements.base import Parameter, Return
from jijbench.functions.factory import RecordFactory


class Solver(FunctionNode[Parameter, Record]):
    def __init__(self, function: tp.Callable, name: str | None = None) -> None:
        if name is None:
            name = function.__name__
        super().__init__(name)
        self.function = function

    def operate(
        self,
        inputs: list[Parameter],
        is_parsed_sampleset: bool = True,
    ) -> Record:
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
