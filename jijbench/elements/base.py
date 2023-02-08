from __future__ import annotations


import typing as tp

from dataclasses import dataclass
from jijbench.node.base import DataNode
from jijbench.typing import NumberTypes, T


@dataclass
class Element(DataNode[T]):
    name: str


@dataclass
class Number(Element[NumberTypes]):
    @classmethod
    def validate_data(cls, data: NumberTypes) -> NumberTypes:
        return cls._validate_dtype(data, (int, float))


@dataclass
class String(Element[str]):
    @classmethod
    def validate_data(cls, data: str) -> str:
        return cls._validate_dtype(data, (str,))


@dataclass
class Callable(Element[tp.Callable]):
    @classmethod
    def validate_data(cls, data: tp.Callable) -> tp.Callable:
        return cls._validate_dtype(data, (tp.Callable,))
