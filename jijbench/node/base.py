from __future__ import annotations

import abc
import typing as tp

from dataclasses import dataclass


DataNodeKT_co = tp.TypeVar("DataNodeKT_co", bound="DataNode", covariant=True)
DataNodeVT_co = tp.TypeVar("DataNodeVT_co", bound="DataNode", covariant=True)


@dataclass
class DataNode:
    data: tp.Any
    name: str | None = None

    def __post_init__(self) -> None:
        self.operator: FunctionNode | None = None


class FunctionNode(tp.Generic[DataNodeKT_co, DataNodeVT_co], metaclass=abc.ABCMeta):
    def __init__(self, name: str | None = None) -> None:
        self._name = name
        self.inputs: list[DataNode] = []

    @abc.abstractmethod
    def __call__(self, inputs: list[DataNodeKT_co], **kwargs: tp.Any) -> DataNodeVT_co:
        pass

    @property
    def name(self) -> str | None:
        return self._name

    def apply(self, inputs: list[DataNodeKT_co], **kwargs: tp.Any) -> DataNodeVT_co:
        self.inputs += inputs
        node = self(inputs, **kwargs)
        node.operator = self
        return node
