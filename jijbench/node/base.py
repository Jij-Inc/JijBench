from __future__ import annotations

import abc
import typing as tp

from dataclasses import dataclass


DNodeIT_co = tp.TypeVar("DNodeIT_co", bound="DataNode", covariant=True)
DNodeOT_co = tp.TypeVar("DNodeOT_co", bound="DataNode", covariant=True)
FNodeT_co = tp.TypeVar("FNodeT_co", bound="FunctionNode", covariant=True)


@dataclass
class DataNode(tp.Generic[DNodeIT_co, DNodeOT_co]):
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
        others: list[DNodeIT_co] | None = None,
        **kwargs: tp.Any,
    ) -> DNodeOT_co:
        inputs = [self]
        if others:
            inputs += others
        node = f(inputs, **kwargs)
        node.operator = f
        return node


class FunctionNode(tp.Generic[DNodeIT_co, DNodeOT_co], metaclass=abc.ABCMeta):
    def __init__(self, name: str | None = None) -> None:
        if name is None:
            name = self.__class__.__name__
        self._name = name
        self.inputs: list[DNodeIT_co] = []

    def __call__(self, inputs: list[DNodeIT_co], **kwargs: tp.Any) -> DNodeOT_co:
        self.inputs += inputs
        node = self.operate(inputs, **kwargs)
        return node

    @property
    def name(self) -> str | None:
        return self._name

    @abc.abstractmethod
    def operate(self, inputs: list[DNodeIT_co], **kwargs: tp.Any) -> DNodeOT_co:
        pass
