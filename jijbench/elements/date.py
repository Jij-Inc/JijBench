from __future__ import annotations

import datetime
import pandas as pd


from dataclasses import dataclass, field
from jijbench.elements.base import Element
from jijbench.typing import DateTypes


@dataclass
class Date(Element[DateTypes]):
    data: DateTypes = field(default_factory=pd.Timestamp.now)
    name: str = "timestamp"

    def __post_init__(self) -> None:
        if isinstance(self.data, str):
            self.data = pd.Timestamp(self.data)

        if isinstance(self.data, datetime.datetime):
            self.data = pd.Timestamp(self.data)

    @classmethod
    def validate_data(cls, data: DateTypes) -> DateTypes:
        data = cls._validate_dtype(data, (str, datetime.datetime, pd.Timestamp))
        if isinstance(data, str):
            try:
                pd.Timestamp(data)
            except Exception:
                raise ValueError(f"Date string '{data}' is invalid for data attribute.")
        return data
