from __future__ import annotations

import numpy as np

from dataclasses import dataclass
from jijbench.node.base import DataNode

# import jijbench.node.functions.math as _math


@dataclass
class Array(DataNode):
    data: np.ndarray
    name: str

    def min(self) -> Array:
        from jijbench.node.functions.math import Min

        return Min().apply([self])

    def max(self) -> Array:
        from jijbench.node.functions.math import Max

        return Max().apply([self])

    def mean(self) -> Array:
        from jijbench.node.functions.math import Mean
        
        return Mean().apply([self])

    def std(self) -> Array:
        from jijbench.node.functions.math import Std
        
        return Std().apply([self])
