from __future__ import annotations

import abc
import copy
import typing as tp

from dataclasses import dataclass, field
from jijbench.typing import T, DataNodeT, DataNodeT2


@dataclass
class DataNode(tp.Generic[T], metaclass=abc.ABCMeta):
    data: T
    name: tp.Hashable
    operator: FunctionNode | None = field(default=None, repr=False)

    def __setattr__(self, name: str, value: tp.Any) -> None:
        if name == "data":
            value = self.validate_data(value)
        return super().__setattr__(name, value)

    @property
    def dtype(self) -> type:
        return type(self.data)

    @classmethod
    @abc.abstractmethod
    def validate_data(cls, data: T) -> T:
        pass

    @classmethod
    def _validate_dtype(cls, data: T, cls_tuple: tuple) -> T:
        if isinstance(data, cls_tuple):
            return data
        else:
            dtype_str = " or ".join(
                map(
                    lambda x: x.__name__ if hasattr(x, "__name__") else str(x),
                    cls_tuple,
                )
            )
            raise TypeError(
                f"Attribute data of class {cls.__name__} must be type {dtype_str}."
            )

    def apply(
        self,
        f: FunctionNode[DataNodeT, DataNodeT2],
        others: list[DataNodeT] | None = None,
        **kwargs: tp.Any,
    ) -> DataNodeT2:
        inputs = [tp.cast("DataNodeT", copy.copy(self))] + (others if others else [])
        node = f(inputs, **kwargs)
        node.operator = f
        return node


class FunctionNode(tp.Generic[DataNodeT, DataNodeT2], metaclass=abc.ABCMeta):
    def __init__(self, name: tp.Hashable = None) -> None:
        if name is None:
            name = self.__class__.__name__
        self._name = name
        self.inputs: list[DataNodeT] = []

    def __call__(self, inputs: list[DataNodeT], **kwargs: tp.Any) -> DataNodeT2:
        node = self.operate(inputs, **kwargs)
        self.inputs += inputs
        # node.operator = self
        return node

    @property
    def name(self) -> tp.Hashable:
        return self._name

    @name.setter
    def name(self, name: tp.Hashable) -> None:
        if not isinstance(name, tp.Hashable):
            raise TypeError(f"{self.__class__.__name__} name must be hashable.")
        self._name = name

    @abc.abstractmethod
    def operate(self, inputs: list[DataNodeT], **kwargs: tp.Any) -> DataNodeT2:
        pass
