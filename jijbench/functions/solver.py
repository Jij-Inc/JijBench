from __future__ import annotations


import typing as tp
import inspect

from jijbench.exceptions.exceptions import SolverFailedError
from jijbench.node.base import DataNode, FunctionNode
from jijbench.data.mapping import Record
from jijbench.functions.factory import RecordFactory


class Solver(FunctionNode[Record]):
    def __init__(self, function: tp.Callable, name: str = "") -> None:
        if not name:
            name = function.__name__
        super().__init__(name)
        self.function = function

    # TODO インターフェースを統一
    def __call__(
        self, is_parsed_sampleset: bool = True, **solver_args: tp.Any
    ) -> Record:
        parameters = inspect.signature(self.function).parameters
        is_kwargs = any([p.kind == 4 for p in parameters.values()])
        solver_args = (
            solver_args
            if is_kwargs
            else {k: v for k, v in solver_args.items() if k in parameters}
        )
        try:
            ret = self.function(**solver_args)
            if not isinstance(ret, tuple):
                ret = (ret,)
        except Exception as e:
            msg = f'An error occurred inside your solver. Please check implementation of "{self.name}". -> {e}'
            raise SolverFailedError(msg)

        solver_return_names = [f"{self.name}_return[{i}]" for i in range(len(ret))]

        inputs = [DataNode(data, name) for data, name in zip(ret, solver_return_names)]
        node = super().__call__(inputs, is_parsed_sampleset=is_parsed_sampleset)
        node.operator = self
        return node

    def operate(
        self, inputs: list[DataNode], is_parsed_sampleset: bool = True
    ) -> Record:
        factory = RecordFactory()
        return factory(inputs, is_parsed_sampleset=is_parsed_sampleset)
