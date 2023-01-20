from __future__ import annotations

import pandas as pd

from dataclasses import dataclass, field
from jijbench.node.base import DataNode


@dataclass
class Date(DataNode):
    data: str | pd.Timestamp = field(default_factory=pd.Timestamp.now)
    name: str = "timestamp"

    def __post_init__(self) -> None:
        if isinstance(self.data, str):
            self.data = pd.Timestamp(self.data)
