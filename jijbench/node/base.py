from __future__ import annotations

import abc
import copy
import typing as tp

from dataclasses import dataclass
from jijbench.typing import DataNodeT, DataNodeT2


@dataclass
class DataNode(metaclass=abc.ABCMeta):
    data: tp.Any
    name: str

    def __post_init__(self) -> None:
        self.operator: FunctionNode | None = None

    @property
    def dtype(self) -> type:
        return type(self.data)

    def apply(
        self,
        f: FunctionNode[DataNodeT, DataNodeT2],
        others: list[DataNodeT] | None = None,
        **kwargs: tp.Any,
    ) -> DataNodeT2:
        inputs = [tp.cast("DataNodeT", copy.deepcopy(self))] + (
            others if others else []
        )
        node = f(inputs, **kwargs)
        return node


class FunctionNode(tp.Generic[DataNodeT, DataNodeT2], metaclass=abc.ABCMeta):
    def __init__(self, name: str | None = None) -> None:
        if name is None:
            name = self.__class__.__name__
        self._name = name
        self.inputs: list[DataNodeT] = []

    def __call__(self, inputs: list[DataNodeT], **kwargs: tp.Any) -> DataNodeT2:
        node = self.operate(inputs, **kwargs)
        self.inputs += inputs
        node.operator = self
        return node

    @property
    def name(self) -> str | None:
        return self._name

    @abc.abstractmethod
    def operate(self, inputs: list[DataNodeT], **kwargs: tp.Any) -> DataNodeT2:
        pass
