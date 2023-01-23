from __future__ import annotations

import uuid

from dataclasses import dataclass, field
from jijbench.node.base import DataNode


@dataclass
class ID(DataNode):
    data: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self) -> None:
        self.data = str(self.data)
