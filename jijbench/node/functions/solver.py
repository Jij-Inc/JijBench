from __future__ import annotations


import typing as tp
import inspect

from jijbench.exceptions import SolverFailedError
from jijbench.node.base import DataNode, FunctionNode
import jijbench.node.functions.factory as _factory

if tp.TYPE_CHECKING:
    from jijbench.node.data.record import Record


class Solver(FunctionNode[DataNode, Record]):
    def __init__(self, function: tp.Callable) -> None:
        super().__init__()
        self.function = function

    def __call__(self, extract: bool = True, **kwargs: tp.Any) -> Record:
        parameters = inspect.signature(self.function).parameters
        is_kwargs = any([p.kind == 4 for p in parameters.values()])
        kwargs = (
            kwargs
            if is_kwargs
            else {k: v for k, v in kwargs.items() if k in parameters}
        )
        try:
            ret = self.function(**kwargs)
            if not isinstance(ret, tuple):
                ret = (ret,)
        except Exception as e:
            msg = f'An error occurred inside your solver. Please check implementation of "{self.name}". -> {e}'
            raise SolverFailedError(msg)

        solver_return_names = [f"{self.name}_return[{i}]" for i in range(len(ret))]
        nodes = [
            DataNode(data=data, name=name)
            for data, name in zip(ret, solver_return_names)
        ]
        return _factory.RecordFactory().apply(nodes, extract=extract)

    @property
    def name(self) -> str:
        return self.function.__name__
