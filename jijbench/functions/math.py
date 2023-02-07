from __future__ import annotations

import numpy as np
import typing as tp

from jijbench.elements.base import Number
from jijbench.node.base import FunctionNode

if tp.TYPE_CHECKING:
    from jijbench.elements.array import Array


class Min(FunctionNode["Array", Number]):
    def operate(self, inputs: list[Array]) -> Number:
        return _operate_array(inputs, np.min)


class Max(FunctionNode["Array", Number]):
    def operate(self, inputs: list[Array]) -> Number:
        return _operate_array(inputs, np.max)


class Mean(FunctionNode["Array", Number]):
    def operate(self, inputs: list[Array]) -> Number:
        return _operate_array(inputs, np.mean)


class Std(FunctionNode["Array", Number]):
    def operate(self, inputs: list[Array]) -> Number:
        return _operate_array(inputs, np.std)


def _operate_array(inputs: list[Array], f: tp.Callable) -> Number:
    data = f(inputs[0].data)
    if "int" in str(data.dtype):
        data = int(data)
    else:
        data = float(data)
    name = inputs[0].name + f"{f.__name__}"
    return Number(data, name)
