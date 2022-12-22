from __future__ import annotations

import uuid
import datetime as dt
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
        if self.benchmark is None:
            self.benchmark = str(uuid.uuid4())

        if self.experiment is None:
            self.experiment = str(uuid.uuid4())

        if self.run is None:
            self.run = str(uuid.uuid4())

        if self.timestamp is None:
            self.timestamp = pd.Timestamp.now()

    def update(self, *, kind: str = "experiment") -> None:
        setattr(self, f"{kind}", str(uuid.uuid4()))
