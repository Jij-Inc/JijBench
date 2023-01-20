from __future__ import annotations

import pandas as pd

from dataclasses import dataclass, field
from jijbench.node.base import DataNode


@dataclass
class Record(DataNode):
    data: pd.Series = field(default_factory=lambda: pd.Series(dtype="object"))
