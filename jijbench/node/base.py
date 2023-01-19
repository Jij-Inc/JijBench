from __future__ import annotations

import typing as tp
from dataclasses import dataclass


DNodeInType = tp.TypeVar("DNodeInType", bound="DataNode")
DNodeOutType = tp.TypeVar("DNodeOutType", bound="DataNode")


@dataclass
class DataNode:
    data: tp.Any
    name: str | None = None

    def __post_init__(self) -> None:
        self.operator: FunctionNode | None = None


class FunctionNode(tp.Generic[DNodeInType, DNodeOutType]):
    def __init__(self, name: str | None = None) -> None:
        self._name = name
        self.inputs: list[DataNode] = []

    def __call__(self, inputs: list[DNodeInType], **kwargs: tp.Any) -> DNodeOutType:
        raise NotImplementedError

    @property
    def name(self) -> str | None:
        return self._name

    def apply(self, inputs: list[DNodeInType], **kwargs: tp.Any) -> DNodeOutType:
        self.inputs += inputs
        node = self(inputs, **kwargs)
        node.operator = self
        return node
