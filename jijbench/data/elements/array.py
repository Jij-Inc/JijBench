from __future__ import annotations

import numpy as np
import typing as tp

from dataclasses import dataclass
from jijbench.node.base import DataNode, FunctionNode

if tp.TYPE_CHECKING:
    from jijbench.functions.math import Min, Max, Mean, Std


@dataclass
class Array(DataNode["Array", "Array"]):
    data: np.ndarray

    @tp.overload
    def apply(self, f: Min) -> Array:
        ...

    @tp.overload
    def apply(self, f: Max) -> Array:
        ...

    @tp.overload
    def apply(self, f: Mean) -> Array:
        ...

    @tp.overload
    def apply(self, f: Std) -> Array:
        ...

    def apply(self, f: FunctionNode) -> Array:
        return super().apply(f)

    def min(self) -> Array:
        from jijbench.functions.math import Min

        return self.apply(Min())

    def max(self) -> Array:
        from jijbench.functions.math import Max

        return self.apply(Max())

    def mean(self) -> Array:
        from jijbench.functions.math import Mean

        return self.apply(Mean())

    def std(self) -> Array:
        from jijbench.functions.math import Std

        return self.apply(Std())
