from __future__ import annotations

import typing as tp

from dataclasses import dataclass
from jijbench.node.base import DataNode


@dataclass
class Number(DataNode):
    data: int | float


@dataclass
class String(DataNode):
    data: str


@dataclass
class Callable(DataNode):
    data: tp.Callable


@dataclass
class Return(DataNode):
    data: tp.Any
