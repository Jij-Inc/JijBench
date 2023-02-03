from __future__ import annotations

import uuid

from dataclasses import dataclass, field
from jijbench.node.base import DataNode


@dataclass
class ID(DataNode[str]):
    data: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str | None = None

    @classmethod
    def validate_data(cls, data: str) -> str:
        return cls._validate_dtype(data, (str,))
