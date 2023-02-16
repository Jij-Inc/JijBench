from __future__ import annotations

import abc
import copy
import typing as tp

from dataclasses import dataclass
from jijbench.typing import T, DataNodeT, DataNodeT2


@dataclass
class DataNode(tp.Generic[T], metaclass=abc.ABCMeta):
    """A base class for all data nodes in a computation graph.

    Attributes:
        data: The data in the node.
        name (Hashable): The name of the node, must be hashable.
        operator (FunctionNode | None, optional): The operator that was applied to the node, or None if the node is a leaf.
    """

    data: T
    name: tp.Hashable

    def __post_init__(self) -> None:
        self.operator: FunctionNode | None = None

    def __setattr__(self, name: str, value: tp.Any) -> None:
        """Set the value of an attribute.

        Args:
            name (str): The name of the attribute.
            value (Any): The new value of the attribute.
        """
        if name == "data":
            value = self.validate_data(value)
        return super().__setattr__(name, value)

    @property
    def dtype(self) -> type:
        """Return the type of the data."""
        return type(self.data)

    @classmethod
    @abc.abstractmethod
    def validate_data(cls, data: T) -> T:
        """Validate the data in the node.
        This method must be implemented by subclasses, and it should raise a TypeError if the
        data is not of a valid type.

        Args:
            data (T): The data to be validated.

        Returns:
            T: The data, if it is valid.
        """
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

    def _init_attrs(self, node: DataNode) -> None:
        """Refresh the attributes in DataNode object.

        Args:
            node (DataNode): DataNode object.
        """
        operator = node.__dict__.pop("operator")
        self.__init__(**node.__dict__)
        self.operator = operator

    def apply(
        self,
        f: FunctionNode[DataNodeT, DataNodeT2],
        others: list[DataNodeT] | None = None,
        **kwargs: tp.Any,
    ) -> DataNodeT2:
        """Apply a function `f` on the data stored in the `DataNode` instance and other input `DataNode` instances.

        Args:
            f (FunctionNode[DataNodeT, DataNodeT2]): The function to be applied on the data.
            others (list[DataNodeT] | None, optional): A list of other `DataNode` instances to be used as inputs to the function. Defaults to None. Defaults to None.
            **kwargs: Arbitrary keyword arguments to be passed to the function.

        Returns:
            DataNodeT2: The result of applying the function on the data.
        """
        inputs = [tp.cast("DataNodeT", copy.copy(self))] + (others if others else [])
        node = f(inputs, **kwargs)
        node.operator = f
        return node


class FunctionNode(tp.Generic[DataNodeT, DataNodeT2], metaclass=abc.ABCMeta):
    """A base class for all function nodes to operate DataNode objects.

    Attributes:
        name (Hashable): The name of the function.
        inputs (list[DataNodeT]): A list of input `DataNode` objects that the function will operate.
    """

    def __init__(self, name: tp.Hashable = None) -> None:
        """Initialize the function node with a name and an empty list of inputs.

        Args:
            name (Hashable, optional): The name of the function. Defaults to None.
        """
        if name is None:
            name = self.__class__.__name__
        self._name = name
        self.inputs: list[DataNodeT] = []

    def __call__(self, inputs: list[DataNodeT], **kwargs: tp.Any) -> DataNodeT2:
        """Operate the inputs to produce a new `DataNode` object.

        Args:
            inputs (list[DataNodeT]): A list of input `DataNode` objects.
            kwargs (Any): Keyword arguments for the operation.

        Returns:
            DataNodeT2: A new `DataNode` object that is the result of the operation.
        """
        node = self.operate(inputs, **kwargs)
        self.inputs += inputs
        return node

    @property
    def name(self) -> tp.Hashable:
        """The name of the function."""
        return self._name

    @name.setter
    def name(self, name: tp.Hashable) -> None:
        """Set the name of the function.

        Args:
            name (Hashable): The new name for the function.

        Raises:
            TypeError: If the specified name is not hashable.
        """
        if not isinstance(name, tp.Hashable):
            raise TypeError(f"{self.__class__.__name__} name must be hashable.")
        self._name = name

    @abc.abstractmethod
    def operate(self, inputs: list[DataNodeT], **kwargs: tp.Any) -> DataNodeT2:
        """Perform the operation on the inputs.
        This method must be implemented by subclasses.

        Args:
            inputs (list[DataNodeT]): A list of input `DataNode` objects.
            kwargs (Any): Keyword arguments for the operation.

        Returns:
            DataNodeT2: A new `DataNode` object that is the result of the operation.
        """
        pass
