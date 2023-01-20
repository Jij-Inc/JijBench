from __future__ import annotations

import numpy as np

from dataclasses import dataclass
from jijbench.node.base import DataNode
import jijbench.node.functions.math as _math


@dataclass
class Array(DataNode):
    data: np.ndarray
    name: str

    def min(self) -> Array:
        return _math.Min().apply([self])

    def max(self) -> Array:
        return _math.Max().apply([self])

    def mean(self) -> Array:
        return _math.Mean().apply([self])

    def std(self) -> Array:
        return _math.Std().apply([self])
