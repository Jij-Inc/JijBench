from __future__ import annotations

import abc
import copy
import typing as tp

from dataclasses import dataclass, field
from jijbench.typing import T, DataNodeT, DataNodeT2


@dataclass
class DataNode(tp.Generic[T], metaclass=abc.ABCMeta):
    data: T
    name: str
    operator: FunctionNode | None = field(default=None, repr=False)

    @property
    def dtype(self) -> type:
        return type(self.data)

    def apply(
        self,
        f: FunctionNode[DataNodeT, DataNodeT2],
        others: list[DataNodeT] | None = None,
        **kwargs: tp.Any,
    ) -> DataNodeT2:
        inputs = [tp.cast("DataNodeT", copy.copy(self))] + (others if others else [])
        node = f(inputs, **kwargs)
        node.operator = f
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
        # node.operator = self
        return node

    @property
    def name(self) -> str:
        return self._name

    @abc.abstractmethod
    def operate(self, inputs: list[DataNodeT], **kwargs: tp.Any) -> DataNodeT2:
        pass
