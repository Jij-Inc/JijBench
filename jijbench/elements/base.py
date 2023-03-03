from __future__ import annotations


import typing as tp

from dataclasses import dataclass
from jijbench.node.base import DataNode
from jijbench.typing import NumberTypes, T


@dataclass
class Element(DataNode[T]):
    """A generic class representing a element node.

    Attributes:
        data: The data in the node.
        name (str): The name of the element.
    """

    name: str


@dataclass
class Number(Element[NumberTypes]):
    """A class representing a numerical element.

    Attributes:
        data (NumberTypes): The int or float object.
        name (str): The name of the element.
    """

    @classmethod
    def validate_data(cls, data: NumberTypes) -> NumberTypes:
        """Validate the data to be a number.

        Args:
            data (NumberTypes): The data to be validated.

        Raises:
            TypeError: If the data is not a number.

        Returns:
            NumberTypes: The validated data.
        """
        return cls._validate_dtype(data, (int, float))


@dataclass
class String(Element[str]):
    """A class representing a string element.

    Attributes:
        data (str): The string object.
        name (str): The name of the element.
    """

    @classmethod
    def validate_data(cls, data: str) -> str:
        """Validate the data to be a string.

        Args:
            data (str): The data to be validated.

        Raises:
            TypeError: If the data is not a string.

        Returns:
            str: The validated data.
        """
        return cls._validate_dtype(data, (str,))


@dataclass
class Callable(Element[tp.Callable]):
    """A class representing a callable element.

    Attributes:
        data (Callable): The callable object.
        name (str): The name of the element.
    """

    @classmethod
    def validate_data(cls, data: tp.Callable) -> tp.Callable:
        """Validate the data to be a callable.

        Args:
            data (Callable): The data to be validated.

        Raises:
            TypeError: If the data is not a callable.

        Returns:
            Callable: The validated data.

        """
        return cls._validate_dtype(data, (tp.Callable,))
