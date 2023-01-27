from __future__ import annotations


import typing as tp
import inspect

from jijbench.exceptions.exceptions import SolverFailedError
from jijbench.node.base import DataNode, FunctionNode
from jijbench.data.mapping import Record
from jijbench.data.elements.values import Parameter
from jijbench.functions.factory import RecordFactory


class Solver(FunctionNode[Parameter, Record]):
    def __init__(self, function: tp.Callable, name: str = "") -> None:
        if not name:
            name = function.__name__
        super().__init__(name)
        self.function = function

    def operate(
        self, inputs: list[Parameter], is_parsed_sampleset: bool = True
    ) -> Record:
        parameters = inspect.signature(self.function).parameters
        solver_args = {
            node.name: node.data for node in inputs if node.name in parameters
        }
        try:
            ret = self.function(**solver_args)
            if not isinstance(ret, tuple):
                ret = (ret,)
        except Exception as e:
            msg = f'An error occurred inside your solver. Please check implementation of "{self.name}". -> {e}'
            raise SolverFailedError(msg)

        solver_return_names = [f"{self.name}_return[{i}]" for i in range(len(ret))]

        ret_nodes = [
            DataNode(data, name) for data, name in zip(ret, solver_return_names)
        ]
        factory = RecordFactory()
        return factory.operate(ret_nodes, is_parsed_sampleset=is_parsed_sampleset)
