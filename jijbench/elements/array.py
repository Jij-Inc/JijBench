from __future__ import annotations

import numpy as np


from dataclasses import dataclass
from jijbench.elements.base import Element, Number
from jijbench.functions.math import Min, Max, Mean, Std


@dataclass
class Array(Element[np.ndarray]):
    def min(self) -> Number:
        return self.apply(Min())

    def max(self) -> Number:
        return self.apply(Max())

    def mean(self) -> Number:
        return self.apply(Mean())

    def std(self) -> Number:
        return self.apply(Std())

    @classmethod
    def validate_data(cls, data: np.ndarray) -> np.ndarray:
        return cls._validate_dtype(data, (np.ndarray,))
