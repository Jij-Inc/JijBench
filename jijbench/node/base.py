from __future__ import annotations

import abc
import copy
import typing as tp

from dataclasses import dataclass


DNodeT_co = tp.TypeVar("DNodeT_co", bound="DataNode", covariant=True)
FNodeT_co = tp.TypeVar("FNodeT_co", bound="FunctionNode", covariant=True)


@dataclass
class DataNode:
    data: tp.Any
    name: str = ""

    def __post_init__(self) -> None:
        self.operator: FunctionNode | None = None

    @property
    def dtype(self) -> type:
        return type(self.data)

    def apply(
        self,
        f: FunctionNode,
        others: list[DataNode] | None = None,
        **kwargs: tp.Any,
    ) -> DataNode:
        if others is None:
            others = []
        inputs = others + [copy.deepcopy(self)]
        node = f(inputs[::-1], **kwargs)
        node.operator = f
        return node


class FunctionNode(tp.Generic[DNodeT_co], metaclass=abc.ABCMeta):
    def __init__(self, name: str | None = None) -> None:
        if name is None:
            name = self.__class__.__name__
        self._name = name
        self.inputs: list[DataNode] = []

    def __call__(self, inputs: list[DataNode], **kwargs: tp.Any) -> DNodeT_co:
        self.inputs += inputs
        node = self.operate(inputs, **kwargs)
        return node

    @property
    def name(self) -> str | None:
        return self._name

    @abc.abstractmethod
    def operate(self, inputs: list[DataNode], **kwargs: tp.Any) -> DNodeT_co:
        pass
