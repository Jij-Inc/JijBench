from __future__ import annotations

import datetime
import pandas as pd


from dataclasses import dataclass, field
from jijbench.node.base import DataNode
from jijbench.typing import DateTypes


@dataclass
class Date(DataNode[DateTypes]):
    """A class for date information.

    This class has date information in the `data` attribute.
    The data can be stored as a string, datetime.datetime object or pandas.Timestamp object.
    If the data is stored as a string, the class will try to convert it to pandas.Timestamp.

    Attributes:
        data (DateTypes): The date information stored in the node.
        name (str): The name of the node.
    """

    data: DateTypes = field(default_factory=pd.Timestamp.now)
    name: str = "timestamp"

    def __post_init__(self) -> None:
        if isinstance(self.data, str):
            self.data = pd.Timestamp(self.data)

        if isinstance(self.data, datetime.datetime):
            self.data = pd.Timestamp(self.data)

    @classmethod
    def validate_data(cls, data: DateTypes) -> DateTypes:
        """Validate data attribute and make sure it's a string, datetime.datetime object or pandas.Timestamp object.

        Raises:
            ValueError: If `data` attribute is a string and not a valid date string.

        Returns:
            DateTypes: The validated date data.
        """
        data = cls._validate_dtype(data, (str, datetime.datetime, pd.Timestamp))
        if isinstance(data, str):
            try:
                pd.Timestamp(data)
            except Exception:
                raise ValueError(f"Date string '{data}' is invalid for data attribute.")
        return data
