from __future__ import annotations
from abc import ABCMeta, abstractmethod

import numpy as np


class Node(metaclass=ABCMeta):
    def __init__(self):
        pass

    @property
    @abstractmethod
    def data(self) -> np.ndarray:
        pass

    @data.setter
    def data(self, data: np.ndarray) -> None:
        pass

    @property
    @abstractmethod
    def grad(self) -> np.ndarray:
        pass

    @grad.setter
    def grad(self, grad: np.ndarray) -> None:
        pass

    @property
    @abstractmethod
    def children(self) -> list[Node]:
        pass
