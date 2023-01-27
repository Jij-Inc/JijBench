from __future__ import annotations

import abc
import copy
import typing as tp

from dataclasses import dataclass
from jijbench.typing import DataNodeIT, DataNodeIT_co, DataNodeOT_co


@dataclass
class DataNode:
    data: tp.Any
    name: str

    def __post_init__(self) -> None:
        self.operator: FunctionNode | None = None

    @property
    def dtype(self) -> type:
        return type(self.data)

    def apply(
        self,
        f: FunctionNode,
        others: list[DataNodeIT] | None = None,
        **kwargs: tp.Any,
    ) -> DataNode:
        inputs = [tp.cast("DataNodeIT", copy.deepcopy(self))] + others if others else []
        node = f(inputs, **kwargs)
        return node


class FunctionNode(tp.Generic[DataNodeIT_co, DataNodeOT_co], metaclass=abc.ABCMeta):
    def __init__(self, name: str | None = None) -> None:
        if name is None:
            name = self.__class__.__name__
        self._name = name
        self.inputs: list[DataNodeIT_co] = []

    def __call__(self, inputs: list[DataNodeIT_co], **kwargs: tp.Any) -> DataNodeOT_co:
        self.inputs += inputs
        node = self.operate(inputs, **kwargs)
        node.operator = self
        return node

    @property
    def name(self) -> str | None:
        return self._name

    @abc.abstractmethod
    def operate(self, inputs: list[DataNodeIT_co], **kwargs: tp.Any) -> DataNodeOT_co:
        pass
