from __future__ import annotations

import numpy as np

from dataclasses import dataclass
from jijbench.node import DataNode, FunctionNode


@dataclass
class Array(DataNode):
    data: np.ndarray
    name: str

    def min(self) -> Array:
        return Min().apply([self])

    def max(self) -> Array:
        return Max().apply([self])

    def mean(self) -> Array:
        return Mean().apply([self])

    def std(self) -> Array:
        return Std().apply([self])


class Min(FunctionNode["Array", "Array"]):
    def __call__(self, inputs: list[Array]) -> Array:
        data = inputs[0].data.min()
        name = inputs[0].name + f"_{self.name}"
        node = Array(data=data, name=name)
        return node

    @property
    def name(self) -> str:
        return "min"


class Max(FunctionNode["Array", "Array"]):
    def __call__(self, inputs: list[Array]) -> Array:
        data = inputs[0].data.max()
        name = inputs[0].name + f"_{self.name}"
        node = Array(data=data, name=name)
        return node

    @property
    def name(self) -> str:
        return "max"


class Mean(FunctionNode["Array", "Array"]):
    def __call__(self, inputs: list[Array]) -> Array:
        data = inputs[0].data.mean()
        name = inputs[0].name + f"_{self.name}"
        node = Array(data=data, name=name)
        return node

    @property
    def name(self) -> str:
        return "mean"


class Std(FunctionNode["Array", "Array"]):
    def __call__(self, inputs: list[Array]) -> Array:
        data = inputs[0].data.std()
        name = inputs[0].name + f"_{self.name}"
        node = Array(data=data, name=name)
        return node

    @property
    def name(self) -> str:
        return "std"
