from __future__ import annotations

from jijbench.node.base import FunctionNode
from jijbench.data.elements.array import Array


class Min(FunctionNode[Array]):
    def operate(self, inputs: list[Array]) -> Array:
        data = inputs[0].data.min()
        name = inputs[0].name + f"_{self.name}"
        node = Array(data, name)
        return node


class Max(FunctionNode[Array]):
    def operate(self, inputs: list[Array]) -> Array:
        data = inputs[0].data.max()
        name = inputs[0].name + f"_{self.name}"
        node = Array(data, name)
        return node


class Mean(FunctionNode[Array]):
    def operate(self, inputs: list[Array]) -> Array:
        data = inputs[0].data.mean()
        name = inputs[0].name + f"_{self.name}"
        node = Array(data, name)
        return node


class Std(FunctionNode[Array]):
    def operate(self, inputs: list[Array]) -> Array:
        data = inputs[0].data.std()
        name = inputs[0].name + f"_{self.name}"
        node = Array(data, name)
        return node

