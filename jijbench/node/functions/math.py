from __future__ import annotations

from jijbench.node.base import FunctionNode
import jijbench.node.data.array as _array


class Min(FunctionNode[_array.Array, _array.Array]):
    def __call__(self, inputs: list[_array.Array]) -> _array.Array:
        data = inputs[0].data.min()
        name = inputs[0].name + f"_{self.name}"
        node = _array.Array(data=data, name=name)
        return node

    @property
    def name(self) -> str:
        return "min"


class Max(FunctionNode[_array.Array, _array.Array]):
    def __call__(self, inputs: list[_array.Array]) -> _array.Array:
        data = inputs[0].data.max()
        name = inputs[0].name + f"_{self.name}"
        node = _array.Array(data=data, name=name)
        return node

    @property
    def name(self) -> str:
        return "max"


class Mean(FunctionNode[_array.Array, _array.Array]):
    def __call__(self, inputs: list[_array.Array]) -> _array.Array:
        data = inputs[0].data.mean()
        name = inputs[0].name + f"_{self.name}"
        node = _array.Array(data=data, name=name)
        return node

    @property
    def name(self) -> str:
        return "mean"


class Std(FunctionNode[_array.Array, _array.Array]):
    def __call__(self, inputs: list[_array.Array]) -> _array.Array:
        data = inputs[0].data.std()
        name = inputs[0].name + f"_{self.name}"
        node = _array.Array(data=data, name=name)
        return node

    @property
    def name(self) -> str:
        return "std"
