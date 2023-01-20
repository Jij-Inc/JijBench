from __future__ import annotations


import typing as tp
import inspect

from jijbench.exceptions.exceptions import SolverFailedError
from jijbench.node.base import DataNode, FunctionNode
from jijbench.data.elements.values import Any
from jijbench.data.record import Record
from jijbench.functions.factory import RecordFactory


class Solver(FunctionNode[Any, Record]):
    def __init__(self, function: tp.Callable, name: str = "") -> None:
        if name is None:
            name = function.__name__
        super().__init__(name)
        self.function = function

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
        nodes = [
            Any(data=data, name=name) for data, name in zip(ret, solver_return_names)
        ]
        return RecordFactory().apply(nodes, is_parsed_sampleset=is_parsed_sampleset)