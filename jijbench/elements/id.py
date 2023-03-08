from __future__ import annotations

import uuid

from dataclasses import dataclass, field
from jijbench.node.base import DataNode


@dataclass
class ID(DataNode[str]):
    """A class for ID.

    Attributes:
        data (str): The ID data. This is generated as a unique identifier if not specified at the time of instantiation.
        name (str): The name of the node.
    """

    data: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "id"

    @classmethod
    def validate_data(cls, data: str) -> str:
        """
        Validate the data for the ID.

        Args:
            data: The data to be stored in the instance.

        Raises:
            TypeError: If the data is not of the expected type.

        Returns:
            The validated data.
        """
        return cls._validate_dtype(data, (str,))
