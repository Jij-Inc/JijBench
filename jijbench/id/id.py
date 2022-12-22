from __future__ import annotations

import uuid
import datetime as dt
import networkx as nx
import pandas as pd

from dataclasses import dataclass


__all__ = []


@dataclass
class ID:
    """ID template."""

    benchmark: str | None = None
    experiment: str | None = None
    run: str | None = None
    timestamp: dt.datetime | pd.Timestamp | None = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = pd.Timestamp.now()

    def update(self, *, kind: str = "experiment") -> None:
        setattr(self, f"{kind}", str(uuid.uuid4()))
