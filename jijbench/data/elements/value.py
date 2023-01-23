from __future__ import annotations

from dataclasses import dataclass
from jijbench.node.base import DataNode


@dataclass
class Value(DataNode):
    data: int | float
