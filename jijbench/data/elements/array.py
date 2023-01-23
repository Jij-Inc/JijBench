from __future__ import annotations

import numpy as np

from dataclasses import dataclass
from jijbench.node.base import DataNode


@dataclass
class Array(DataNode):
    data: np.ndarray
    name: str

    def min(self) -> Array:
        from jijbench.functions.math import Min

        return Min().apply([self])

    def max(self) -> Array:
        from jijbench.functions.math import Max

        return Max().apply([self])

    def mean(self) -> Array:
        from jijbench.functions.math import Mean

        return Mean().apply([self])

    def std(self) -> Array:
        from jijbench.functions.math import Std

        return Std().apply([self])
